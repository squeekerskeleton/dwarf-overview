"""Connection diagnostics: which way to reach the DWARF is fastest and steadiest.

The DWARF 3 can be reached three ways:
  - on your Wi-Fi (STA mode), at whatever address your router gives it;
  - on its own hotspot (a network named DWARF3_..., the DWARF is always 192.168.88.1 there);
  - over Bluetooth, which only carries setup messages (the page handles that part; see
    index.html, "Bluetooth"). Bluetooth can tell you which network the DWARF is on and its address,
    and can ask it to rejoin its saved Wi-Fi, but it can't carry control or pictures.

For every address that answers, this measures round-trip time, jitter and dropped connections on the
control port, the time the DWARF takes to answer a request, and download speed from its file server.
The steadiest route wins: dropped connections count most, then jitter and delay, then speed. On
Windows it also reads the laptop's Wi-Fi signal and link rate, and can list and join DWARF hotspots.
Nothing is sent outside the local network, and Wi-Fi passwords are never logged or shown.
"""
from __future__ import annotations

import ipaddress
import os
import threading
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from xml.sax.saxutils import escape

from . import config, scope

HOTSPOT_SSID = re.compile(r"^DWARF[\w\- ]{0,40}$", re.I)
DEFAULT_HOTSPOT_PASSWORD = "DWARF_12345678"  # DWARFLAB's factory default for the hotspot and Bluetooth


class NetError(Exception):
    pass


def _run(cmd: list[str], timeout: float = 15) -> str:
    try:
        kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace", **kw).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------------------------------------------------------------- the laptop's Wi-Fi

def wifi_status() -> dict:
    """The laptop's Wi-Fi connection(s) and any DWARF hotspots in range."""
    out = {"interfaces": [], "hotspots": [], "supported": sys.platform == "win32"}
    if sys.platform == "win32":
        block = {}
        for line in _run(["netsh", "wlan", "show", "interfaces"]).splitlines() + [""]:
            if ":" not in line:
                if block.get("ssid"):
                    out["interfaces"].append(block)
                block = {} if not line.strip() else block
                continue
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "name":
                if block.get("ssid"):
                    out["interfaces"].append(block)
                block = {"adapter": value}
            elif key == "ssid":
                block["ssid"] = value
            elif key == "signal":
                block["signal_pct"] = int(value.rstrip("%") or 0) if value.rstrip("%").isdigit() else None
            elif key.startswith("receive rate"):
                block["rx_mbps"] = _num(value)
            elif key.startswith("transmit rate"):
                block["tx_mbps"] = _num(value)
            elif key == "channel":
                block["channel"] = value
            elif key == "radio type":
                block["radio"] = value
        ssid, signal = None, None
        for line in _run(["netsh", "wlan", "show", "networks", "mode=bssid"]).splitlines():
            m = re.match(r"^SSID \d+ : (.*)$", line.strip())
            if m:
                ssid = m.group(1).strip()
                continue
            m = re.match(r"^Signal\s*:\s*(\d+)%", line.strip())
            if m and ssid and HOTSPOT_SSID.match(ssid):
                if not any(h["ssid"] == ssid for h in out["hotspots"]):
                    out["hotspots"].append({"ssid": ssid, "signal_pct": int(m.group(1))})
    else:
        for line in _run(["nmcli", "-t", "-f", "ACTIVE,SSID,SIGNAL,RATE", "dev", "wifi"]).splitlines():
            parts = line.split(":")
            if len(parts) >= 4:
                active, ssid, signal, rate = parts[0], parts[1], parts[2], parts[3]
                if active == "yes":
                    out["interfaces"].append({"ssid": ssid, "signal_pct": _num(signal), "rx_mbps": _num(rate)})
                if HOTSPOT_SSID.match(ssid or ""):
                    out["hotspots"].append({"ssid": ssid, "signal_pct": _num(signal)})
    return out


def _num(v) -> float | None:
    m = re.search(r"[\d.]+", str(v or ""))
    return float(m.group()) if m else None


def join_hotspot(ssid: str, password: str = "") -> dict:
    """Join a DWARF's own hotspot (Windows). Saves a Wi-Fi profile for it, like joining it by hand."""
    if sys.platform != "win32":
        raise NetError("Joining the DWARF's hotspot from the app works on Windows; join it from the Wi-Fi menu here.")
    if not HOTSPOT_SSID.match(ssid or ""):
        raise NetError("That isn't a DWARF hotspot name.")
    password = password or DEFAULT_HOTSPOT_PASSWORD
    if not 8 <= len(password) <= 63:
        raise NetError("Wi-Fi passwords are 8 to 63 characters.")
    previous = next((i["ssid"] for i in wifi_status()["interfaces"] if i.get("ssid")), None)
    profile = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>{escape(ssid)}</name>
  <SSIDConfig><SSID><name>{escape(ssid)}</name></SSID></SSIDConfig>
  <connectionType>ESS</connectionType><connectionMode>manual</connectionMode>
  <MSM><security>
    <authEncryption><authentication>WPA2PSK</authentication><encryption>AES</encryption><useOneX>false</useOneX></authEncryption>
    <sharedKey><keyType>passPhrase</keyType><protected>false</protected><keyMaterial>{escape(password)}</keyMaterial></sharedKey>
  </security></MSM>
</WLANProfile>"""
    fd, path = tempfile.mkstemp(suffix=".xml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(profile)
        _run(["netsh", "wlan", "add", "profile", f"filename={path}", "user=current"])
    finally:
        try:
            os.remove(path)  # it holds the password
        except OSError:
            pass
    _run(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}"])
    for _ in range(20):
        time.sleep(1)
        if scope._port_open(scope.HOTSPOT_IP, 8082, 0.8):
            config.save({"dwarf_ip": scope.HOTSPOT_IP, "previous_wifi": previous or ""})
            return {"joined": ssid, "ip": scope.HOTSPOT_IP, "previous": previous}
    config.save({"previous_wifi": previous or ""})
    raise NetError(f"Joined {ssid}, but the DWARF didn't answer at {scope.HOTSPOT_IP}. "
                   "If you changed its hotspot password, enter it and try again.")


def rejoin_previous() -> dict:
    """Go back to the Wi-Fi the laptop was on before joining the DWARF's hotspot (Windows)."""
    name = config.load().get("previous_wifi")
    if sys.platform != "win32" or not name:
        raise NetError("There's no earlier Wi-Fi network to go back to.")
    _run(["netsh", "wlan", "connect", f"name={name}"])
    return {"rejoined": name}


# ---------------------------------------------------------------- measuring a route

def _connect_ms(ip: str, port: int, timeout: float = 1.0) -> float | None:
    t = time.perf_counter()
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return (time.perf_counter() - t) * 1000
    except OSError:
        return None


def _sample_file(ip: str) -> str | None:
    """A reasonably large file on the DWARF to time a download with."""
    link = re.compile(r'href="([^"?]+)"')
    try:
        for root in ("/DWARF3/Astronomy/", "/DWARF_II/Astronomy/", "/Astronomy/"):
            with urllib.request.urlopen(f"http://{ip}{root}", timeout=4) as r:
                html = r.read(512 * 1024).decode("utf-8", "replace")
            folders = sorted({h for h in link.findall(html) if h.endswith("/") and not h.startswith(("..", "/"))},
                             reverse=True)
            for folder in folders[:3]:
                with urllib.request.urlopen(f"http://{ip}{root}{folder}", timeout=4) as r:
                    inner = r.read(512 * 1024).decode("utf-8", "replace")
                files = [h for h in link.findall(inner) if h.lower().endswith((".fits", ".fit", ".jpg", ".png", ".tif"))]
                if files:
                    return f"http://{ip}{root}{folder}{files[0]}"
    except (OSError, ValueError):
        return None
    return None


def measure(ip: str, pings: int = 20) -> dict:
    """Round-trip, jitter, drops, response time and download speed for one address."""
    if not scope._is_local(ip):
        raise NetError("Only addresses on the local network can be tested.")
    times = []
    for _ in range(pings):
        times.append(_connect_ms(ip, 8082))
        time.sleep(0.05)
    ok = [t for t in times if t is not None]
    res = {"ip": ip, "route": "DWARF hotspot" if ip == scope.HOTSPOT_IP else "your Wi-Fi / network",
           "loss_pct": round(100 * (len(times) - len(ok)) / len(times)),
           "rtt_ms": round(statistics.median(ok), 1) if ok else None,
           "rtt_p90_ms": round(sorted(ok)[int(len(ok) * 0.9) - 1 if len(ok) > 1 else 0], 1) if ok else None,
           "jitter_ms": round(statistics.mean(abs(a - b) for a, b in zip(ok, ok[1:])), 1) if len(ok) > 2 else None,
           "control_port": _connect_ms(ip, 9900) is not None}
    if not ok:
        res["grade"] = "unreachable"
        return res
    api = []
    for _ in range(3):
        t = time.perf_counter()
        if scope.device_info(ip, timeout=3):
            api.append((time.perf_counter() - t) * 1000)
    res["api_ms"] = round(statistics.median(api), 1) if api else None
    url = _sample_file(ip)
    if url:
        got, t, deadline = 0, time.perf_counter(), time.perf_counter() + 6
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-16777215"})
            with urllib.request.urlopen(req, timeout=6) as r:
                while got < 16 * 1024 * 1024 and time.perf_counter() < deadline:
                    chunk = r.read(256 * 1024)
                    if not chunk:
                        break
                    got += len(chunk)
            secs = time.perf_counter() - t
            if got > 256 * 1024 and secs > 0:
                res["download_mbps"] = round(got * 8 / secs / 1e6, 1)
                res["download_mb"] = round(got / 1e6, 1)
        except OSError:
            res["download_error"] = True
    loss, jit, rtt = res["loss_pct"], res["jitter_ms"] or 0, res["rtt_ms"] or 0
    res["grade"] = ("excellent" if loss == 0 and jit < 15 and rtt < 30 else
                    "good" if loss <= 5 and jit < 40 and rtt < 80 else "weak")
    # Lower is better: drops dominate, then jitter and delay; speed breaks ties.
    res["score"] = round(loss * 20 + jit * 1.5 + rtt * 0.5 + (res.get("api_ms") or 0) * 0.05
                         - min(res.get("download_mbps") or 0, 100) * 0.3, 1)
    return res


def diagnose(addresses: list[str] | None = None, save: bool = True) -> dict:
    """Test every way the DWARF can be reached right now and pick the steadiest."""
    cands = list(dict.fromkeys([a for a in (addresses or []) if scope._is_local(a)]))
    saved = (config.load().get("dwarf_ip") or "").strip()
    for extra in (saved, scope.HOTSPOT_IP):
        if scope._is_local(extra) and extra not in cands:
            cands.append(extra)
    routes = []
    for ip in cands:
        if scope._port_open(ip, 8082, 1.0) and scope.device_info(ip):
            routes.append(measure(ip))
    usable = [r for r in routes if r.get("grade") != "unreachable"]
    best = min(usable, key=lambda r: r["score"]) if usable else None
    if best and save and best["ip"] != saved:
        config.save({"dwarf_ip": best["ip"]})
    return {"routes": routes, "best": best["ip"] if best else None, "switched": bool(best and best["ip"] != saved),
            "wifi": wifi_status(), "single_radio_note":
            "A laptop with one Wi-Fi adapter can be on your Wi-Fi or the DWARF's hotspot, not both, so only the "
            "routes it's on right now can be measured. Use Join hotspot to try the other one."}


# ---------------------------------------------------------------- live status (Network tab)

_CACHE: dict = {"at": 0.0, "adapters": [], "wifi": {}}
_CACHE_LOCK = threading.Lock()


def adapters() -> list[dict]:
    """The laptop's network connections: wired, Wi-Fi or other, with address and whether it's up."""
    out = []
    if sys.platform == "win32":
        cur = None
        for line in _run(["ipconfig"]).splitlines():
            if line and not line[0].isspace() and line.rstrip().endswith(":"):
                cur = {"adapter": line.strip().rstrip(":"), "ip": None, "up": True}
                low = cur["adapter"].lower()
                cur["kind"] = ("wifi" if "wireless" in low or "wi-fi" in low or "wlan" in low else
                               "wired" if "ethernet" in low and "bluetooth" not in low else
                               "bluetooth" if "bluetooth" in low else "other")
                out.append(cur)
            elif cur is not None:
                if "Media disconnected" in line:
                    cur["up"] = False
                m = re.search(r"IPv4[^:]*:\s*([\d.]+)", line)
                if m:
                    cur["ip"] = m.group(1)
        out = [a for a in out if a["kind"] != "other" or a["ip"]]
        for a in out:
            a["up"] = a["up"] and bool(a["ip"])
    else:
        for name, ip in re.findall(r"\d+: (\S+?)\s+inet ([\d.]+)/", _run(["ip", "-o", "-4", "addr"])):
            if name != "lo":
                kind = "wifi" if name.startswith(("wl", "wlan")) else "wired" if name.startswith(("en", "eth")) else "other"
                out.append({"adapter": name, "ip": ip, "up": True, "kind": kind})
    return out


def _route_for(ip: str, ads: list[dict]) -> str:
    if ip == scope.HOTSPOT_IP:
        return "DWARF hotspot (direct Wi-Fi)"
    try:
        addr = ipaddress.ip_address(ip)
        for a in ads:
            if a.get("ip") and addr in ipaddress.ip_network(f"{a['ip']}/24", strict=False):
                return f"Same network as the laptop's {'Wi-Fi' if a['kind'] == 'wifi' else 'wired' if a['kind'] == 'wired' else a['kind']} connection (DWARF in STA mode)"
    except ValueError:
        pass
    return "Routed network (DWARF in STA mode)"


def live() -> dict:
    """A quick snapshot for the Network tab: cheap enough to call every couple of seconds."""
    from . import control
    now = time.monotonic()
    with _CACHE_LOCK:
        if now - _CACHE["at"] > 10:
            _CACHE.update(at=now, adapters=adapters(), wifi=wifi_status())
        ads, wifi = _CACHE["adapters"], _CACHE["wifi"]
    ip = (config.load().get("dwarf_ip") or "").strip()
    d = {"ip": ip or None, "reachable": False, "rtt_ms": None, "control_port": False}
    if scope._is_local(ip):
        d["rtt_ms"] = _connect_ms(ip, 8082, timeout=1.0)
        d["reachable"] = d["rtt_ms"] is not None
        if d["rtt_ms"] is not None:
            d["rtt_ms"] = round(d["rtt_ms"], 1)
            d["control_port"] = _connect_ms(ip, 9900, timeout=0.8) is not None
        d["route"] = _route_for(ip, ads)
    try:
        d["link"] = control.status()
    except Exception:
        d["link"] = {"connected": False}
    return {"time": time.time(), "laptop": ads, "wifi": wifi, "dwarf": d, "usb": scope.usb_folders()}
