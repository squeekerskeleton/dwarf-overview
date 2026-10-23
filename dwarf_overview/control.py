"""Simple telescope control: one persistent link to the DWARF for goto, joystick and capture.

Speaks the same WebSocket/protobuf protocol as Dwarfium (port 9900), with the
command numbers and message layouts from dwarfii_api. Like Dwarfium, it asks
for control of the telescope (master lock) when it connects.

Safety: joystick movement must be refreshed by the page at least every
1.2 seconds or the link sends a joystick stop by itself, so a lost key-up or a
closed tab can't leave the telescope slewing.
"""
from __future__ import annotations

import json
import os
import struct
import threading
import time
import urllib.request

from . import config, scope

CLIENT_SUFFIX = scope.CLIENT_SUFFIX
JOYSTICK_TIMEOUT = 1.2


class ControlError(Exception):
    pass


# ---------------------------------------------------------------- protobuf helpers

def f_int(no: int, v: int) -> bytes:
    return scope._varint(no << 3) + scope._varint(int(v))


def f_bool(no: int, v: bool) -> bytes:
    return f_int(no, 1 if v else 0)


def f_double(no: int, v: float) -> bytes:
    return scope._varint(no << 3 | 1) + struct.pack("<d", float(v))


def f_bytes(no: int, v: bytes | str) -> bytes:
    if isinstance(v, str):
        v = v.encode("utf-8")
    return scope._varint(no << 3 | 2) + scope._varint(len(v)) + v


def parse(buf: bytes) -> dict[int, list]:
    """Decode one protobuf message into {field: [values]} (repeated-safe)."""
    out: dict[int, list] = {}
    i = 0
    while i < len(buf):
        key, i = scope._read_varint(buf, i)
        no, wt = key >> 3, key & 7
        if wt == 0:
            v, i = scope._read_varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            ln, i = scope._read_varint(buf, i)
            v, i = buf[i:i + ln], i + ln
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            break
        out.setdefault(no, []).append(v)
    return out


def one(fields: dict, no: int, default=None):
    v = fields.get(no)
    return v[0] if v else default


def sint(v) -> int:
    v = v or 0
    return v - (1 << 64) if v >= 1 << 63 else v


def text(v) -> str:
    return v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else (v or "")


# ---------------------------------------------------------------- command log

_LOG_LOCK = threading.Lock()


def log_path():
    return config.data_dir() / "dwarf-commands.log"


def _log(direction: str, module: int, cmd: int, typ: int, data: bytes) -> None:
    """Every command and reply, so problems on a real DWARF can be read afterwards. Data is cut to 48 bytes,
    and system-module payloads (which can hold Wi-Fi details) are never written."""
    try:
        fields = parse(data) if data else {}
        code = sint(one(fields, 2 if module == 6 else 1, 0)) if typ in (1, 3) and fields else ""
        shown = "" if module == 4 else data[:48].hex()
        line = f"{time.strftime('%H:%M:%S')} {direction} {module}/{cmd} type={typ} code={code} {shown}\n"
        with _LOG_LOCK:
            path = log_path()
            if path.exists() and path.stat().st_size > 2_000_000:
                path.replace(path.with_suffix(".old.log"))
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:  # logging must never break control
        pass


def log_tail(lines: int = 150) -> list[str]:
    try:
        return log_path().read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


# ---------------------------------------------------------------- the link

# Battery 15201, charging 15202, storage 15203, body temperature 15243, sensor temperature 15292, and the
# device-state snapshot 16405 that holds all of them.
TELEMETRY_CMDS = {15201, 15202, 15203, 15243, 15292, 16405}


def _storage(f: dict) -> dict:
    if not one(f, 4, 0):  # is_valid
        return {"storage_valid": False}
    return {"storage_valid": True, "storage_free_gb": one(f, 1, 0), "storage_total_gb": one(f, 2, 0)}


class Link:
    def __init__(self, ip: str, hardware_id: int):
        self.ip = ip
        self.client_id = scope.CLIENT_IDS.get(hardware_id, "0000DAF3") + CLIENT_SUFFIX
        self.ws = scope._WebSocket(ip, 9900, timeout=5)
        self.ws.sock.settimeout(None)
        self.send_lock = threading.Lock()
        self.cv = threading.Condition()
        self.replies: dict[tuple[int, int], list[dict]] = {}
        self.closed = False
        self.last_rx = self.last_tx = time.monotonic() - 4.5  # first device-state read half a second in
        self.goto = {}            # latest one-click goto progress
        self.events: list[dict] = []
        self.joy_deadline = 0.0
        self.telemetry: dict = {}  # battery, charging, storage, temperature, as the DWARF reports them
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._keeper, daemon=True).start()

    def _reader(self):
        try:
            while not self.closed:
                msg = self.ws.recv()
                if not msg:
                    continue
                self.last_rx = time.monotonic()
                f = parse(msg)
                module, cmd, typ = one(f, 4, 0), one(f, 5, 0), one(f, 6, 0)
                data = one(f, 7, b"")
                if cmd not in (16405,):  # skip the 5 s heartbeat
                    _log("<-", module, cmd, typ, data)
                if cmd in (11013, 11014) and typ in (1, 3):
                    d = parse(data)
                    self.goto = {"step": sint(one(d, 1, 0)), "code": sint(one(d, 2, 0)), "done": bool(one(d, 3, 0)),
                                 "at": time.time()}
                if cmd in TELEMETRY_CMDS and typ != 0:
                    try:
                        self._telemetry(cmd, data)
                    except Exception:  # a reading we can't decode is skipped
                        pass
                if typ == 2:
                    self.events = (self.events + [{"cmd": cmd, "at": time.time()}])[-50:]
                with self.cv:
                    self.replies.setdefault((module, cmd), []).append({"type": typ, "data": data})
                    self.cv.notify_all()
        except (OSError, ConnectionError) as e:
            _log("xx", 0, 0, 0, b"")
            with _LOG_LOCK:
                try:
                    with open(log_path(), "a", encoding="utf-8") as fh:
                        fh.write(f"{time.strftime('%H:%M:%S')} link closed: {type(e).__name__}: {e}\n")
                except OSError:
                    pass
        finally:
            self.closed = True
            with self.cv:
                self.cv.notify_all()

    def _telemetry(self, cmd: int, data: bytes) -> None:
        t = dict(self.telemetry)
        if cmd == 16405:  # device state snapshot (the heartbeat reply)
            f = parse(data)
            if sint(one(f, 7, 0)) != 0:
                return
            state = parse(one(f, 6, b"")) if one(f, 6) is not None else {}
            if one(state, 9) is not None:
                t["battery"] = sint(one(parse(one(state, 9)), 1, 0))
            if one(state, 3) is not None:
                t["charging"] = sint(one(parse(one(state, 3)), 1, 0))
            if one(state, 4) is not None:
                t.update(_storage(parse(one(state, 4))))
            if one(state, 7) is not None:
                tf = parse(one(state, 7))
                if sint(one(tf, 1, 0)) == 0:
                    t["temperature"] = sint(one(tf, 2, 0))
            tele = parse(one(f, 2)) if one(f, 2) is not None else {}
            if one(tele, 7) is not None:
                cmos = parse(one(tele, 7))
                if one(cmos, 1) is not None:
                    t["sensor_temperature"] = sint(one(cmos, 1))
        else:
            f = parse(data)
            if cmd == 15201:
                t["battery"] = sint(one(f, 1, 0))
            elif cmd == 15202:
                t["charging"] = sint(one(f, 1, 0))
            elif cmd == 15203:
                t.update(_storage(f))
            elif cmd == 15243 and sint(one(f, 1, 0)) == 0:
                t["temperature"] = sint(one(f, 2, 0))
            elif cmd == 15292 and one(f, 1) is not None:
                t["sensor_temperature"] = sint(one(f, 1))
        if "battery" in t and not 0 <= t["battery"] <= 100:
            t.pop("battery")
        t["at"] = time.time()
        self.telemetry = t

    def _keeper(self):
        """Heartbeat like Dwarfium (device state every 5 s when idle) and the joystick dead-man stop."""
        while not self.closed:
            time.sleep(0.25)
            now = time.monotonic()
            if self.joy_deadline and now > self.joy_deadline:
                self.joy_deadline = 0.0
                try:
                    self.send(6, 14008)
                except OSError:
                    pass
            if now - max(self.last_tx, self.last_rx) > 5:
                try:
                    self.send(14, 16405)
                except OSError:
                    self.closed = True

    def send(self, module: int, cmd: int, data: bytes = b"") -> None:
        if self.closed:
            raise ControlError("The link to the DWARF dropped. Connect again.")
        with self.send_lock:
            self.ws.send(scope.packet(module, cmd, data, self.client_id))
            self.last_tx = time.monotonic()
        if cmd != 16405:
            _log("->", module, cmd, 0, data)

    def request(self, module: int, cmd: int, data: bytes = b"", timeout: float = 10, code_field: int = 1) -> dict:
        with self.cv:
            self.replies.pop((module, cmd), None)
        self.send(module, cmd, data)
        end = time.monotonic() + timeout
        with self.cv:
            while True:
                got = [r for r in self.replies.get((module, cmd), []) if r["type"] in (1, 3)]
                if got:
                    fields = parse(got[0]["data"])
                    return {"code": sint(one(fields, code_field, 0)), "fields": fields}
                left = end - time.monotonic()
                if left <= 0 or self.closed:
                    raise ControlError("The DWARF didn't answer in time." if not self.closed else
                                       "The link to the DWARF dropped. Connect again.")
                self.cv.wait(left)

    def close(self):
        self.closed = True
        self.ws.close()


_LINK: Link | None = None
_LINK_LOCK = threading.Lock()


def link(connect: bool = False) -> Link:
    global _LINK
    cfg = config.load()
    ip = (cfg.get("dwarf_ip") or "").strip()
    if not scope._is_local(ip):
        raise ControlError("Find your DWARF first (Find my DWARF), so the app knows its address.")
    with _LINK_LOCK:
        if _LINK and (_LINK.closed or _LINK.ip != ip):
            _LINK = None
        if _LINK is None:
            if not connect:
                raise ControlError("Not connected. Click Connect.")
            try:
                _LINK = Link(ip, int(cfg.get("dwarf_hardware_id") or 2))
            except OSError as e:
                raise ControlError(f"Couldn't reach the DWARF at {ip} ({e}).")
        return _LINK


def _ok(r: dict, what: str) -> dict:
    if r["code"] != 0:
        raise ControlError(f"The DWARF declined {what} (code {r['code']}).")
    return r


# ---------------------------------------------------------------- actions

# ---------------------------------------------------------------- auto-reconnect

RETRIES, RETRY_SPACING, PROBE_TIMEOUT = 5, 1.0, 0.8
_WANTED = False      # the user connected and hasn't clicked Disconnect
_RECON = {"state": "idle", "attempt": 0, "error": None, "at": None, "drops": []}
_RECON_LOCK = threading.Lock()
_RUNNING = threading.Lock()


def _probe_once(saved: str) -> dict | None:
    """One quick search: the last-known address first, then the hotspot, recent devices and the whole local network."""
    from concurrent.futures import ThreadPoolExecutor
    if scope._is_local(saved) and scope._port_open(saved, 8082, PROBE_TIMEOUT):
        d = scope.device_info(saved, timeout=PROBE_TIMEOUT)
        if d:
            return d
    mine = scope.local_addresses()
    cands = [scope.HOTSPOT_IP] + scope._neighbours()
    for net in scope.local_networks(mine)[:4]:
        cands += [str(h) for h in net.hosts()]
    cands = [c for c in dict.fromkeys(cands) if c not in mine and c != saved]
    with ThreadPoolExecutor(max_workers=256) as pool:
        hosts = [h for h, ok in zip(cands, pool.map(lambda h: scope._port_open(h, 8082, 0.5), cands)) if ok]
        for d in pool.map(lambda h: scope.device_info(h, timeout=PROBE_TIMEOUT), hosts):
            if d:
                return d
    return None


def _reconnect_loop() -> None:
    global _LINK, _WANTED
    now = time.time()
    with _RECON_LOCK:
        drops = [t for t in _RECON["drops"] if now - t < 120] + [now]
        _RECON.update(state="reconnecting", attempt=0, error=None, at=now, drops=drops)
    saved = (config.load().get("dwarf_ip") or "").strip()
    if len(drops) >= 3 and scope.device_info(saved, timeout=PROBE_TIMEOUT):
        # The DWARF is still there but keeps dropping us: another app is most likely taking control.
        _WANTED = False
        with _RECON_LOCK:
            _RECON.update(state="failed", error="The DWARF dropped this app's link 3 times in 2 minutes while still on the "
                          "network. Another app (the DWARFLAB phone app or Dwarfium) is most likely taking control. "
                          "Close the other app, then click Connect.")
        return
    last = ""
    for attempt in range(1, RETRIES + 1):
        if not _WANTED:
            with _RECON_LOCK:
                _RECON.update(state="idle")
            return
        start = time.monotonic()
        with _RECON_LOCK:
            _RECON["attempt"] = attempt
        try:
            d = _probe_once(saved)
            if d:
                if d["ip"] != saved:
                    config.save({"dwarf_ip": d["ip"], "dwarf_hardware_id": d["hardware_id"]})
                with _LINK_LOCK:
                    _LINK = Link(d["ip"], d["hardware_id"])
                _LINK.request(4, 13004, f_bool(1, True), timeout=4)
                with _RECON_LOCK:
                    _RECON.update(state="ok", error=None, at=time.time(), reconnected_to=d["ip"])
                _watch()
                return
            last = "no DWARF answered"
        except (OSError, ControlError) as e:
            last = str(e)
        time.sleep(max(0.0, RETRY_SPACING - (time.monotonic() - start)))
    _WANTED = False
    with _RECON_LOCK:
        _RECON.update(state="failed", error=f"Lost the DWARF: searched {RETRIES} times, once a second, and it didn't answer "
                      f"({last}). Check it's switched on and on your Wi-Fi (or its hotspot), then click Connect.")


def _watch() -> None:
    """Start a watcher for the current link; when it drops unexpectedly, run the reconnect loop."""
    l = _LINK
    if l is None or getattr(l, "watched", False):
        return
    l.watched = True

    def run():
        while not l.closed:
            time.sleep(0.25)
        if _WANTED and _LINK is l and _RUNNING.acquire(blocking=False):
            try:
                _reconnect_loop()
            finally:
                _RUNNING.release()

    threading.Thread(target=run, daemon=True).start()


def connect() -> dict:
    global _WANTED
    l = link(connect=True)
    _WANTED = True
    with _RECON_LOCK:
        _RECON.update(state="idle", attempt=0, error=None)
    _watch()
    info = scope.device_info(l.ip) or {}
    try:
        lock = l.request(4, 13004, f_bool(1, True), timeout=6)["code"]
    except ControlError:
        lock = None
    return {"ip": l.ip, "name": info.get("name"), "model": info.get("model"), "control": lock == 0,
            "control_code": lock}


def open_preview(camera: str = "tele", go_live: bool = False) -> dict:
    """Switch the camera's video stream on (what Dwarfium does when its camera button is pressed).

    enterCamera (14/16404), optionally goLive for that camera (tele 3/11010, wide 3/11020, which a cold DWARF
    needs before it streams; it also ends live stacking, so it's only sent when the plain way gave no picture),
    then the preview-quality request (tele 1/10050, wide 2/12036, level 1). Nothing moves or focuses.
    """
    l = link(connect=True)
    if not _WANTED:
        connect()
    r = l.request(14, 16404, f_bytes(3, f_int(1, 1)), timeout=8)
    if r["code"] != 0:
        raise ControlError(f"The DWARF wouldn't open its camera (code {r['code']}). If the DWARFLAB phone app or "
                           "Dwarfium is in control, close it and try again.")
    live_code = None
    if go_live:
        try:
            live_code = l.request(3, 11010 if camera == "tele" else 11020, b"", timeout=8)["code"]
        except ControlError:
            live_code = "no reply"
    module, cmd = (1, 10050) if camera == "tele" else (2, 12036)
    try:
        code = l.request(module, cmd, f_int(1, 1), timeout=5)["code"]
    except ControlError:  # older firmware doesn't know this request; the camera is already on
        code = None
    return {"camera": camera, "enter_code": r["code"], "live_code": live_code, "preview_code": code}


# ---------------------------------------------------------------- AWAY (stow) and reboot

# Motor reset, the same four steps Dwarfium's "Reset motors" sends: each axis runs to its end stop and back
# to the DWARF's home position, where the lens is tucked away, as after power-on.
PARK_STEPS = [(1, True), (1, False), (2, False), (2, True)]


def _await_final(l: "Link", module: int, cmd: int, data: bytes, timeout: float, code_field: int = 2) -> int:
    """Send a request and wait for its completion notice (type 3); fail early on an error reply."""
    with l.cv:
        l.replies.pop((module, cmd), None)
    l.send(module, cmd, data)
    end = time.monotonic() + timeout
    with l.cv:
        while True:
            got = l.replies.get((module, cmd), [])
            for r in got:
                code = sint(one(parse(r["data"]), code_field, 0))  # motor replies (ResMotor) carry id in 1, code in 2
                if r["type"] == 3 or code != 0:
                    return code
            left = end - time.monotonic()
            if left <= 0 or l.closed:
                raise ControlError("The DWARF didn't finish moving in time." if not l.closed else
                                   "The link to the DWARF dropped while it was moving.")
            l.cv.wait(left)


MOTORS = {1: "rotation", 2: "tilt"}
AWAY_MOVES = False  # off until the move is checked on a real DWARF 3


def _position_from(data: bytes) -> tuple[int, float]:
    f = parse(data)  # ResMotorPosition: 1 id, 2 code, 3 position (degrees, double)
    code = sint(one(f, 2, 0))
    if code != 0:
        raise ControlError(f"The DWARF couldn't report its position (code {code}).")
    raw = one(f, 3, b"\0" * 8)
    return sint(one(f, 1, 0)), struct.unpack("<d", raw)[0] if isinstance(raw, (bytes, bytearray)) else 0.0


def read_positions(ip: str | None = None, hardware_id: int | None = None) -> dict:
    """Read both motors' positions. Read-only: nothing moves. Uses the app's link if it's up, else a short one."""
    out = {}
    l = _LINK if _LINK and not _LINK.closed else None
    if l:
        for mid in MOTORS:
            r = l.request(6, 14011, f_int(1, mid), timeout=5, code_field=2)
            if r["code"] != 0:
                raise ControlError(f"The DWARF couldn't report its {MOTORS[mid]} position (code {r['code']}).")
            raw = one(r["fields"], 3, b"\0" * 8)
            out[mid] = struct.unpack("<d", raw)[0]
        return out
    cfg = config.load()
    ip = ip or (cfg.get("dwarf_ip") or "").strip()
    hw = hardware_id or int(cfg.get("dwarf_hardware_id") or 2)
    if not scope._is_local(ip):
        raise ControlError("Find your DWARF first.")
    client_id = scope.CLIENT_IDS.get(hw, "0000DAF3") + CLIENT_SUFFIX
    ws = scope._WebSocket(ip, 9900, timeout=4)
    try:
        for mid in MOTORS:
            ws.send(scope.packet(6, 14011, f_int(1, mid), client_id))
        end = time.monotonic() + 5
        while len(out) < len(MOTORS) and time.monotonic() < end:
            ws.sock.settimeout(max(0.1, end - time.monotonic()))
            msg = ws.recv()
            if not msg:
                continue
            f = parse(msg)
            if one(f, 5, 0) == 14011 and one(f, 6, 0) in (1, 3):
                mid, pos = _position_from(one(f, 7, b""))
                out[mid or (len(out) + 1)] = pos
    except (OSError, ConnectionError) as e:
        raise ControlError(f"Couldn't read the DWARF's position ({e}).")
    finally:
        ws.close()
    if len(out) < len(MOTORS):
        raise ControlError("The DWARF didn't report its position.")
    return out


def record_away(ip: str | None = None, hardware_id: int | None = None) -> dict:
    """Save where the telescope is right now as its AWAY position."""
    pos = read_positions(ip, hardware_id)
    saved = {"rotation": round(pos[1], 3), "tilt": round(pos[2], 3),
             "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    config.save({"away_position": saved, "away_pending": False})
    return saved


def away(progress=None) -> dict:
    """Stop everything, then move the telescope to the AWAY position you set (lens tucked away)."""
    cfg = config.load()
    ip, hw = (cfg.get("dwarf_ip") or "").strip(), int(cfg.get("dwarf_hardware_id") or 2)
    stopped = scope.stop_all(ip, hw)
    if not AWAY_MOVES:
        # Moving was switched off after it turned the wrong way on a real DWARF 3 (0.6.1). AWAY only stops.
        return {"stopped": stopped.get("acknowledged"), "parked": False, "position": "none"}
    if progress:
        progress("Stopped everything; sending the telescope home…")
    time.sleep(1.0)
    l = link(connect=True)
    if not _WANTED:
        connect()
    target = cfg.get("away_position") or None
    if target:
        # Same move Dwarfium uses for its preset positions: 10 degrees/s, ramping 100, resolution 3.
        for mid, key in ((1, "rotation"), (2, "tilt")):
            if progress:
                progress(f"Moving {MOTORS[mid]} to your AWAY position ({float(target[key]):.1f}°)…")
            data = f_int(1, mid) + f_double(2, float(target[key])) + f_double(3, 10) + f_int(4, 100) + f_int(5, 3)
            code = _await_final(l, 6, 14001, data, timeout=90)
            if code != 0:
                raise ControlError(f"The {MOTORS[mid]} motor couldn't reach your AWAY position (code {code}). "
                                   "Everything else is stopped.")
        return {"stopped": stopped.get("acknowledged"), "parked": True, "position": "yours",
                "saved_at": target.get("saved_at")}
    for i, (motor, direction) in enumerate(PARK_STEPS, 1):
        if progress:
            progress(f"No AWAY position saved yet: using the factory home position ({i} of {len(PARK_STEPS)})…")
        code = _await_final(l, 6, 14003, f_int(1, motor) + f_bool(2, direction), timeout=90)
        if code != 0:
            raise ControlError(f"The DWARF couldn't reach its factory home position (step {i}, code {code}). "
                               "Everything is stopped. Set your own AWAY position in Settings.")
    return {"stopped": stopped.get("acknowledged"), "parked": True, "position": "factory"}


def reboot() -> dict:
    """Restart the DWARF. Its connection drops for a minute or two; the app then looks for it again."""
    global _WANTED
    l = link(connect=True)
    was_connected = True
    _WANTED = False  # don't treat the restart as a dropped link
    l.send(5, 13505)
    with _RECON_LOCK:
        _RECON.update(state="rebooting", attempt=0, error=None, at=time.time())

    def wait_for_it():
        time.sleep(20)
        saved = (config.load().get("dwarf_ip") or "").strip()
        for i in range(36):  # up to 3 more minutes
            with _RECON_LOCK:
                _RECON["attempt"] = i + 1
            d = None
            try:
                d = _probe_once(saved)
            except Exception:
                pass
            if d:
                if d["ip"] != saved:
                    config.save({"dwarf_ip": d["ip"], "dwarf_hardware_id": d["hardware_id"]})
                time.sleep(5)  # let it finish starting up
                try:
                    if was_connected:
                        disconnect()
                        connect()
                    with _RECON_LOCK:
                        _RECON.update(state="ok", error=None, at=time.time(), reconnected_to=d["ip"])
                except ControlError as e:
                    with _RECON_LOCK:
                        _RECON.update(state="failed", error=f"The DWARF restarted but the app couldn't connect: {e}")
                return
            time.sleep(5)
        with _RECON_LOCK:
            _RECON.update(state="failed", error="The DWARF didn't come back within 3 minutes of restarting. Check it's on "
                          "and on your Wi-Fi, then use Find my DWARF.")

    threading.Thread(target=wait_for_it, daemon=True).start()
    return {"rebooting": True}


def status() -> dict:
    l = _LINK
    with _RECON_LOCK:
        recon = {k: v for k, v in _RECON.items() if k != "drops"}
    cfg = config.load()
    away_info = {"pending": bool(cfg.get("away_pending")), "position": cfg.get("away_position") or None}
    if not l or l.closed:
        return {"connected": False, "reconnect": recon, "away": away_info}
    return {"connected": True, "ip": l.ip, "goto": l.goto, "seconds_since_heard": round(time.monotonic() - l.last_rx, 1),
            "reconnect": recon, "away": away_info, "telemetry": l.telemetry}


_STORAGE_CACHE: dict = {}


def health() -> dict:
    """Battery, temperature and storage for Plan & Control.

    Battery and temperature come over the control link (the DWARF reports them while connected). Storage also
    comes from device info over plain HTTP, so it shows even when another app is in control. Only the SD card
    numbers are taken from device info; the rest of it (which holds Wi-Fi details) is ignored."""
    l = _LINK
    out = dict(l.telemetry) if l and not l.closed else {}
    out["connected"] = bool(l and not l.closed)
    if "storage_total_gb" not in out:
        ip = (config.load().get("dwarf_ip") or "").strip()
        cached = _STORAGE_CACHE.get(ip)
        if cached and time.time() - cached["at"] < 30:
            out.update(cached["data"])
        elif scope._is_local(ip):
            try:
                req = urllib.request.Request(f"http://{ip}:8082/deviceInfo", data=b"{}", method="POST",
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=2) as resp:
                    sd = (json.loads(resp.read()).get("data") or {}).get("sdCardInfo") or {}
                data = {"storage_free_gb": sd.get("availableSize"), "storage_total_gb": sd.get("totalSize"),
                        "storage_valid": sd.get("totalSize") is not None}
                _STORAGE_CACHE[ip] = {"at": time.time(), "data": data}
                out.update(data)
            except (OSError, ValueError, AttributeError):
                pass
    return out


def away_watcher() -> None:
    """While an AWAY position is waiting to be recorded, record it the next time the DWARF connects.

    "Next time" means the DWARF goes from off (or out of reach) to reachable, so a position set by hand
    while it's off is what gets saved, not wherever it was when the app started.
    """
    def run():
        was_online = None
        while True:
            time.sleep(4)
            try:
                cfg = config.load()
                ip = (cfg.get("dwarf_ip") or "").strip()
                if not cfg.get("away_pending"):
                    was_online = None
                    continue
                online = scope._is_local(ip) and scope._port_open(ip, 8082, 1.0)
                if online and was_online is False:
                    time.sleep(8)  # let it finish starting up
                    record_away()
                was_online = online if was_online is not None or not online else True
            except Exception:  # try again on the next pass
                pass
    threading.Thread(target=run, daemon=True).start()


def disconnect() -> dict:
    global _LINK, _WANTED
    _WANTED = False
    with _RECON_LOCK:
        _RECON.update(state="idle", attempt=0, error=None)
    if _LINK:
        _LINK.close()
        _LINK = None
    return {"connected": False}


def goto(ra_hours: float, dec: float, name: str) -> dict:
    """One-click goto: the DWARF calibrates if it needs to, slews, and starts tracking."""
    if not (0 <= ra_hours < 24 and -90 <= dec <= 90):
        raise ControlError("RA must be 0 to 24 hours and Dec -90 to 90 degrees.")
    cfg = config.load()
    try:
        lat, lon = float(cfg.get("site_lat")), float(cfg.get("site_lon"))
    except (TypeError, ValueError):
        raise ControlError("Add your site's latitude and longitude in Settings first; the DWARF needs them to calibrate.")
    l = link()
    l.goto = {}
    body = (f_double(1, ra_hours) + f_double(2, dec) + f_bytes(3, name[:60]) + f_double(4, lon) + f_double(5, lat)
            + f_int(6, 2))
    r = l.request(3, 11013, body, timeout=15, code_field=2)
    _ok(r, "the goto")
    return {"started": True, "target": name}


def stop_goto() -> dict:
    l = link()
    l.send(3, 11015)
    l.send(3, 11004)
    return {"stopped": True}


def joystick(angle: float, length: float) -> dict:
    """angle: 0 right, 90 down, 180 left, 270 up (degrees clockwise on screen); length 0..1."""
    l = link()
    angle, length = float(angle) % 360, max(0.0, min(1.0, float(length)))
    l.send(6, 14006, f_double(1, angle) + f_double(2, length))
    l.joy_deadline = time.monotonic() + JOYSTICK_TIMEOUT
    return {"ok": True}


def joystick_stop() -> dict:
    l = link()
    l.joy_deadline = 0.0
    l.send(6, 14008)
    return {"ok": True}


# ---------------------------------------------------------------- camera settings and capture

def camera_catalog(ip: str, mode_id: int = 2) -> dict:
    req = urllib.request.Request(f"http://{ip}:8082/shootingMode/getParamAndSetting", method="POST",
                                 data=json.dumps({"modeId": mode_id}).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read(4 * 1024 * 1024))   # Python ints keep the 64-bit parameter ids exact
    except (OSError, ValueError) as e:
        raise ControlError(f"Couldn't read the camera settings from the DWARF ({e}).")
    if body.get("code") != 0 or not isinstance(body.get("data"), dict):
        raise ControlError(f"The DWARF didn't share its camera settings (code {body.get('code')}). "
                           "Its firmware may be older than this feature needs; update it in the DWARFLAB app.")
    return body["data"]


def _decode_pid(pid: int) -> dict:
    return {"mode": (pid >> 56) & 255, "category": (pid >> 48) & 255, "camera": (pid >> 44) & 15, "index": pid & 255}


def _params(data: dict, camera_id: int) -> list[dict]:
    out = []
    for cam in data.get("cameraParams") or []:
        if cam.get("cameraId") != camera_id:
            continue
        for key, p in (cam.get("specialParams") or {}).items():
            out.append({**p, "key": key})
        for p in cam.get("generalParams") or []:
            out.append({**p, "key": p.get("name")})
    for tech in data.get("shootingTechSettings") or []:
        if tech.get("cameraId") == camera_id:
            out += [{**p, "key": p.get("name")} for p in tech.get("generalParams") or []]
    return out


def _seconds(label: str) -> float | None:
    import re
    m = re.match(r"^\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?\s*(?:s|sec)?\s*$", str(label), re.I)
    if not m:
        return None
    return float(m.group(1)) / (float(m.group(2)) if m.group(2) else 1)


def camera_options(camera_id: int = 0) -> dict:
    """The exposures and gains the DWARF advertises for deep-sky stacking, for the settings dropdowns."""
    l = link()
    data = camera_catalog(l.ip)
    params = _params(data, camera_id)
    exp = next((p for p in params if p["key"] == "exp"), None)
    gain = next((p for p in params if p["key"] == "gain"), None)
    exposures = [{"seconds": _seconds(v.get("name")), "label": v.get("name")} for v in (exp or {}).get("values", [])
                 if isinstance(v, dict) and _seconds(v.get("name"))]
    gains = [v.get("value") if isinstance(v, dict) else v for v in (gain or {}).get("values", [])]
    if not gains and gain and "minValue" in gain:
        gains = list(range(int(gain["minValue"]), int(gain["maxValue"]) + 1, 10))
    # Stacking (technique 2) can narrow the range, e.g. gain 40 to 240 on the DWARF 3 telephoto.
    for r in (gain or {}).get("range") or []:
        if r.get("techId") == 2:
            gains = [g for g in gains if r.get("minValue", g) <= g <= r.get("maxValue", g)]
    return {"exposures": [e for e in exposures if e["seconds"] >= 1], "gains": gains,
            "filters": [{"index": 0, "label": "VIS"}, {"index": 1, "label": "Astro"}, {"index": 2, "label": "Duo-band"}]}


def start_capture(filter_index: int, exposure: float, gain: int, frames: int, camera_id: int = 0) -> dict:
    """Configure and start deep-sky live stacking, following the same steps as Dwarfium."""
    if filter_index not in (0, 1, 2):
        raise ControlError("Pick a filter.")
    frames = int(frames)
    if not 1 <= frames <= 999:
        raise ControlError("Frames must be between 1 and 999.")
    l = link()
    data = camera_catalog(l.ip)
    params = _params(data, camera_id)

    def find(key):
        hits = [p for p in params if p["key"] == key]
        if len(hits) != 1 or "paramId" not in hits[0]:
            raise ControlError(f"The DWARF didn't advertise a {key} setting.")
        return hits[0]

    exp_p, gain_p = find("exp"), find("gain")
    exp_opt = [v for v in exp_p.get("values", []) if isinstance(v, dict) and _seconds(v.get("name")) is not None
               and abs(_seconds(v["name"]) - exposure) < 1e-6]
    if len(exp_opt) != 1:
        raise ControlError(f"{exposure:g} s isn't one of the DWARF's exposure choices.")
    gain_values = [v.get("value") if isinstance(v, dict) else v for v in gain_p.get("values", [])]
    for rng in gain_p.get("range") or []:  # stacking can narrow the gain range (40-240 on the DWARF 3)
        if rng.get("techId") == 2:
            gain_values = [g for g in gain_values if rng.get("minValue", g) <= g <= rng.get("maxValue", g)]
    if gain_values and gain not in gain_values:
        raise ControlError(f"Gain {gain} isn't one of the DWARF's gain choices for stacking.")
    count = [p for p in params if "paramId" in p and _decode_pid(int(p["paramId"]))["category"] == 2
             and _decode_pid(int(p["paramId"]))["index"] == 16]
    if len(count) != 1:
        raise ControlError("The DWARF didn't advertise a frame-count setting.")
    lo, hi = int(count[0].get("minValue", 1)), int(count[0].get("maxValue", 999))
    if not lo <= frames <= hi:
        raise ControlError(f"Frames must be between {lo} and {hi} on this DWARF.")
    exp_id, gain_id, count_id = int(exp_p["paramId"]), int(gain_p["paramId"]), int(count[0]["paramId"])
    exp_value = int(exp_opt[0]["value"])

    r = _ok(l.request(14, 16402, f_int(1, 2)), "switching to deep-sky mode")
    if sint(one(r["fields"], 2, 0)) != 2:
        raise ControlError("The DWARF didn't switch to deep-sky mode.")
    _ok(l.request(14, 16404, f_bytes(3, f_int(1, 1))), "opening the camera")
    _ok(l.request(14, 16403, f_int(1, 2)), "selecting stacking")
    qs = _ok(l.request(3, 11040, f_int(1, camera_id)), "listing quick settings")
    templates = []
    for raw in qs["fields"].get(2, []):
        q = parse(raw)
        info = text(one(q, 6, b""))
        parts = info.split("|")
        try:
            if len(parts) >= 6 and parts[2].strip() and parts[3].strip() and parts[4].strip():
                templates.append((abs(float(parts[2]) - exposure), abs(int(parts[3]) - gain), parts))
        except ValueError:
            continue
    if not templates:
        raise ControlError("The DWARF returned no deep-sky quick settings.")
    parts = list(min(templates)[2])
    parts[2], parts[3] = f"{exposure:g}", str(gain)
    info_id = "|".join(parts)
    set_exp = f_int(1, exp_id) + f_int(2, 1) + f_int(3, exp_value)
    set_gain = f_int(1, gain_id) + f_int(2, 1) + f_int(3, gain)
    _ok(l.request(15, 16700, set_exp), "the exposure")
    _ok(l.request(15, 16701, set_gain), "the gain")
    _ok(l.request(3, 11041, f_bytes(1, info_id)), "the quick setting")
    # The DWARF 3 reloads stale values after a quick setting, so set them again (as Dwarfium does).
    _ok(l.request(15, 16700, set_exp), "the exposure")
    _ok(l.request(15, 16701, set_gain), "the gain")
    _ok(l.request(15, 16703, f_int(1, count_id) + f_int(2, frames)), "the frame count")
    _ok(l.request(3, 11005, f_int(1, filter_index), timeout=20), "starting the capture")
    return {"started": True, "exposure": exposure, "gain": gain, "frames": frames, "filter": filter_index}


def stop_capture() -> dict:
    l = link()
    l.send(3, 11006)
    return {"stopped": True}
