"""Find the DWARF on the local network, and stop everything it's doing.

Finding: the laptop's own addresses are read locally, then only the private
subnets it is on (plus the DWARF's own hotspot address) are probed. A DWARF is
confirmed by asking it for its device info on port 8082, the same check
Dwarfium makes. Nothing is sent outside the local network.

Stopping: the DWARF is controlled over a WebSocket on port 9900 carrying
protobuf packets. "Stop all" sends every stop command the current DWARF
protocol defines (motors, goto, tracking, calibration, captures, focus), using
the command numbers from dwarfii_api, the library Dwarfium uses.

USB-C: the DWARF 3 shows up as a USB drive only; DWARFLAB's firmware offers no
control over the cable. Files can be read straight from that drive, though.
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import re
import socket
import string
import struct
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config

HOTSPOT_IP = "192.168.88.1"  # the DWARF's own Wi-Fi hotspot
MODELS = {1: "DWARF II", 2: "DWARF 3", 4: "DWARF mini"}
CLIENT_IDS = {1: "0000DAF2", 2: "0000DAF3", 4: "0000DAF4"}
CLIENT_SUFFIX = "-0000-1000-8000-00805F9B34FB"


class ScopeError(Exception):
    pass


def _is_local(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return a.version == 4 and (a.is_private or a.is_link_local) and not a.is_loopback


# ---------------------------------------------------------------- finding it on the network

def local_addresses() -> list[str]:
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    # Connecting a UDP socket sends nothing; it just reveals which local address would be used.
    for probe in (HOTSPOT_IP, "10.255.255.255", "172.31.255.255"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe, 9))
                found.add(s.getsockname()[0])
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            out = subprocess.run(["ipconfig"], capture_output=True, text=True, timeout=10,
                                 creationflags=subprocess.CREATE_NO_WINDOW).stdout
            found.update(re.findall(r"IPv4[^:]*:\s*([\d.]+)", out))
        except (OSError, subprocess.SubprocessError):
            pass
    return sorted(ip for ip in found if _is_local(ip))


def _port_open(ip: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def device_info(ip: str, timeout: float = 3) -> dict | None:
    """Ask a host whether it is a DWARF. Returns its model and name, or None."""
    if not _is_local(ip):
        return None
    req = urllib.request.Request(f"http://{ip}:8082/deviceInfo", data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read(64 * 1024))
    except (OSError, ValueError):
        return None
    data = body.get("data") if isinstance(body, dict) else None
    if body.get("code") != 0 or not isinstance(data, dict) or data.get("deviceId") not in MODELS:
        return None
    return {"ip": ip, "model": MODELS[data["deviceId"]], "hardware_id": data["deviceId"],
            "name": str(data.get("deviceName") or MODELS[data["deviceId"]])}


def scan() -> dict:
    mine = local_addresses()
    candidates = [HOTSPOT_IP]
    saved = (config.load().get("dwarf_ip") or "").strip()
    if _is_local(saved):
        candidates.insert(0, saved)
    subnets = []
    for ip in mine:
        net = ipaddress.ip_network(f"{ip}/24", strict=False)
        if net not in subnets:
            subnets.append(net)
    for net in subnets[:4]:
        candidates += [str(h) for h in net.hosts() if str(h) not in mine]
    candidates = list(dict.fromkeys(candidates))
    with ThreadPoolExecutor(max_workers=128) as pool:
        open_hosts = [ip for ip, ok in zip(candidates, pool.map(lambda h: _port_open(h, 8082, 0.6), candidates)) if ok]
        devices = [d for d in pool.map(device_info, open_hosts) if d]
    return {"devices": devices, "searched": [str(n) for n in subnets[:4]] + [HOTSPOT_IP], "laptop": mine}


# ---------------------------------------------------------------- USB-C (the DWARF as a drive)

def _drive_roots() -> list[Path]:
    if sys.platform == "win32":
        return [Path(f"{c}:\\") for c in string.ascii_uppercase[3:] if os.path.exists(f"{c}:\\")]
    roots = []
    for base in ("/Volumes", "/media", "/run/media"):
        p = Path(base)
        if p.is_dir():
            for child in p.iterdir():
                roots += [child] + ([g for g in child.iterdir() if g.is_dir()] if base != "/Volumes" and child.is_dir() else [])
    return roots


def usb_folders() -> list[str]:
    """Astronomy folders on any DWARF connected over USB-C."""
    found = []
    for root in _drive_roots():
        for rel in ("DWARF3/Astronomy", "Astronomy", "DWARF_II/Astronomy"):
            try:
                folder = root / rel
                if folder.is_dir() and (any(folder.glob("DWARF_*")) or (root / "DWARF3").is_dir()):
                    found.append(str(folder))
                    break
            except OSError:
                continue
    return found


# ---------------------------------------------------------------- stop all (WebSocket + protobuf)

# (label, module, command, request body); every request here has an empty body
# except the per-motor stop, which carries the motor id.
STOP_COMMANDS = [
    ("joystick", 6, 14008, b""),
    ("motor 1", 6, 14002, b"\x08\x01"),
    ("motor 2", 6, 14002, b"\x08\x02"),
    ("goto", 3, 11004, b""),
    ("one-click goto", 3, 11015, b""),
    ("calibration", 3, 11001, b""),
    ("EQ solving", 3, 11019, b""),
    ("sun/moon tracking", 3, 11012, b""),
    ("tracking", 7, 14801, b""),
    ("sentry mode", 7, 14803, b""),
    ("tele live stacking", 3, 11006, b""),
    ("wide live stacking", 3, 11017, b""),
    ("tele burst", 1, 10004, b""),
    ("wide burst", 2, 12024, b""),
    ("tele video", 1, 10006, b""),
    ("wide video", 2, 12031, b""),
    ("tele timelapse", 1, 10034, b""),
    ("wide timelapse", 2, 12026, b""),
    ("focus", 8, 15003, b""),
    ("astro autofocus", 8, 15005, b""),
]


def _varint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(no: int, value) -> bytes:
    if isinstance(value, int):
        return _varint(no << 3) + _varint(value)
    return _varint(no << 3 | 2) + _varint(len(value)) + value


def packet(module: int, cmd: int, data: bytes, client_id: str) -> bytes:
    """A WsPacket: major 1, minor 20, wire device id 4, request type 0 (omitted)."""
    return (_field(1, 1) + _field(2, 20) + _field(3, 4) + _field(4, module) + _field(5, cmd)
            + _field(7, data) + _field(8, client_id.encode()))


def _parse(buf: bytes) -> dict:
    fields, i = {}, 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        no, wt = key >> 3, key & 7
        if wt == 0:
            fields[no], i = _read_varint(buf, i)
        elif wt == 2:
            ln, i = _read_varint(buf, i)
            fields[no], i = buf[i:i + ln], i + ln
        elif wt == 5:
            i += 4
        elif wt == 1:
            i += 8
        else:
            break
    return fields


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    n = shift = 0
    while i < len(buf):
        b = buf[i]
        i += 1
        n |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            break
    return n, i


class _WebSocket:
    """Just enough of RFC 6455 to send and receive binary frames."""

    def __init__(self, ip: str, port: int, timeout: float):
        self.sock = socket.create_connection((ip, port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET / HTTP/1.1\r\nHost: {ip}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ScopeError("The DWARF closed the connection.")
            head += chunk
        head, self.pending = head.split(b"\r\n\r\n", 1)
        expect = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
        if b" 101 " not in head.split(b"\r\n")[0] or expect not in head:
            raise ScopeError("The DWARF didn't accept the control connection.")

    def send(self, payload: bytes, opcode: int = 2) -> None:
        mask = os.urandom(4)
        n = len(payload)
        header = bytes([0x80 | opcode]) + (bytes([0x80 | n]) if n < 126 else
                                           bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536 else
                                           bytes([0x80 | 127]) + struct.pack(">Q", n))
        self.sock.sendall(header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def _read(self, n: int) -> bytes:
        while len(self.pending) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("closed")
            self.pending += chunk
        out, self.pending = self.pending[:n], self.pending[n:]
        return out

    def recv(self) -> bytes | None:
        b0, b1 = self._read(2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", self._read(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", self._read(8))[0]
        mask = self._read(4) if b1 & 0x80 else b""
        data = self._read(n)
        if mask:
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        op = b0 & 0x0F
        if op == 9:
            self.send(data, 10)
            return None
        if op == 8:
            raise ConnectionError("closed")
        return data if op in (0, 2) else None

    def close(self) -> None:
        try:
            self.send(b"", 8)
        except OSError:
            pass
        self.sock.close()


def stop_all(ip: str, hardware_id: int = 2) -> dict:
    if not _is_local(ip):
        raise ScopeError("Set the DWARF's address first (Find my DWARF, or Settings).")
    client_id = CLIENT_IDS.get(hardware_id, "0000DAF3") + CLIENT_SUFFIX
    try:
        ws = _WebSocket(ip, 9900, timeout=4)
    except OSError as e:
        raise ScopeError(f"Couldn't reach the DWARF at {ip} ({e}). Is the laptop on the same Wi-Fi?")
    replies = {}
    try:
        for _, module, cmd, data in STOP_COMMANDS:
            ws.send(packet(module, cmd, data, client_id))
        wanted = {(m, c) for _, m, c, _ in STOP_COMMANDS}
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and wanted - replies.keys():
            ws.sock.settimeout(max(0.1, deadline - time.monotonic()))
            try:
                msg = ws.recv()
            except (socket.timeout, ConnectionError):
                break
            if not msg:
                continue
            f = _parse(msg)
            key = (f.get(4), f.get(5))
            if key in wanted and f.get(6, 0) in (1, 3):
                code = _parse(f.get(7, b"")).get(1, 0)
                replies[key] = code - (1 << 64) if code >= 1 << 63 else code
    finally:
        ws.close()
    results = []
    for label, module, cmd, _ in STOP_COMMANDS:
        code = replies.get((module, cmd))
        results.append({"what": label, "code": code,
                        "result": "stopped" if code == 0 else "no reply" if code is None else f"declined (code {code})"})
    return {"ip": ip, "results": results, "sent": len(STOP_COMMANDS),
            "acknowledged": sum(1 for r in results if r["code"] is not None)}
