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


# ---------------------------------------------------------------- the link

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
        self.last_rx = self.last_tx = time.monotonic()
        self.goto = {}            # latest one-click goto progress
        self.events: list[dict] = []
        self.joy_deadline = 0.0
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
                if cmd in (11013, 11014) and typ in (1, 3):
                    d = parse(data)
                    self.goto = {"step": sint(one(d, 1, 0)), "code": sint(one(d, 2, 0)), "done": bool(one(d, 3, 0)),
                                 "at": time.time()}
                if typ == 2:
                    self.events = (self.events + [{"cmd": cmd, "at": time.time()}])[-50:]
                with self.cv:
                    self.replies.setdefault((module, cmd), []).append({"type": typ, "data": data})
                    self.cv.notify_all()
        except (OSError, ConnectionError):
            pass
        finally:
            self.closed = True
            with self.cv:
                self.cv.notify_all()

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

def connect() -> dict:
    l = link(connect=True)
    info = scope.device_info(l.ip) or {}
    try:
        lock = l.request(4, 13004, f_bool(1, True), timeout=6)["code"]
    except ControlError:
        lock = None
    return {"ip": l.ip, "name": info.get("name"), "model": info.get("model"), "control": lock == 0,
            "control_code": lock}


def status() -> dict:
    l = _LINK
    if not l or l.closed:
        return {"connected": False}
    return {"connected": True, "ip": l.ip, "goto": l.goto, "seconds_since_heard": round(time.monotonic() - l.last_rx, 1)}


def disconnect() -> dict:
    global _LINK
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
