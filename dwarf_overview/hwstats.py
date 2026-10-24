"""Hardware usage for the Network tab: CPU, memory, GPU, drives, network throughput and the busiest programs.

Uses psutil (installed with the app). The GPU comes from nvidia-smi when an NVIDIA card is present.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time

from . import config

# Programs worth naming in the list: the app itself and the tools it drives.
KNOWN = {"siril": "Siril", "siril-cli": "Siril (script)", "graxpert": "GraXpert", "pipp": "PIPP",
         "autostakkert": "AutoStakkert", "registax6": "RegiStax", "imppg": "ImPPG", "dwarfium": "Dwarfium",
         "dwarfiumproxy": "Dwarfium helper", "mediamtx": "Dwarfium video relay", "astap": "ASTAP"}

_LOCK = threading.Lock()
_STATE: dict = {"net": None, "gpu": None, "gpu_at": 0.0, "procs": {}}


def _gpu() -> list[dict] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                             creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            num = lambda v: float(v) if v.replace(".", "", 1).isdigit() else None
            gpus.append({"name": p[0], "percent": num(p[1]), "mem_used_mb": num(p[2]), "mem_total_mb": num(p[3]),
                         "temp_c": num(p[4])})
    return gpus or None


def _app_label(p) -> str | None:
    name = (p.info.get("name") or "").lower().removesuffix(".exe")
    if name in KNOWN:
        return KNOWN[name]
    if name.startswith("python"):
        cmd = " ".join(p.info.get("cmdline") or [])
        if "dwarf_overview" in cmd:
            return "DWARF Overview"
    return None


def snapshot() -> dict:
    try:
        import psutil
    except ImportError:
        return {"available": False, "note": "Restart the app once to finish installing the hardware monitor (psutil)."}
    now = time.monotonic()
    out = {"available": True, "cpu_percent": psutil.cpu_percent(None), "cores": psutil.cpu_count(logical=True),
           "per_core": psutil.cpu_percent(None, percpu=True)}
    try:
        f = psutil.cpu_freq()
        out["cpu_mhz"] = round(f.current) if f else None
    except Exception:
        out["cpu_mhz"] = None
    m = psutil.virtual_memory()
    out["memory"] = {"used_gb": round((m.total - m.available) / 1e9, 1), "total_gb": round(m.total / 1e9, 1),
                     "percent": m.percent}

    drives = []
    watch = {}
    for key, label in (("work_folder", "processing"), ("library_folder", "library")):
        v = (config.load().get(key) or "").strip()
        if v:
            watch[v[:3].upper() if sys.platform == "win32" else v] = label
    for part in psutil.disk_partitions(all=False):
        if "cdrom" in part.opts or not part.fstype:
            continue
        try:
            u = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        if u.total < 2e9:  # skip tiny system mounts
            continue
        drives.append({"mount": part.mountpoint, "free_gb": round(u.free / 1e9), "total_gb": round(u.total / 1e9),
                       "percent": u.percent, "used_for": watch.get(part.mountpoint.upper(), "")})
    out["drives"] = drives[:8]

    with _LOCK:
        io = psutil.net_io_counters(pernic=True)
        prev = _STATE["net"]
        _STATE["net"] = (now, io)
        nics = []
        if prev:
            dt = max(0.2, now - prev[0])
            for nic, c in io.items():
                p = prev[1].get(nic)
                if not p:
                    continue
                rx, tx = (c.bytes_recv - p.bytes_recv) / dt, (c.bytes_sent - p.bytes_sent) / dt
                if rx > 1000 or tx > 1000:
                    nics.append({"name": nic, "down_mbps": round(rx * 8 / 1e6, 2), "up_mbps": round(tx * 8 / 1e6, 2)})
        out["network"] = sorted(nics, key=lambda n: -(n["down_mbps"] + n["up_mbps"]))[:4]

        if now - _STATE["gpu_at"] > 2:
            _STATE["gpu"], _STATE["gpu_at"] = _gpu(), now
        out["gpu"] = _STATE["gpu"]

        procs, seen = [], {}
        for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            try:
                cached = _STATE["procs"].get(p.pid)
                if cached is None:
                    cached = p
                    p.cpu_percent(None)  # first call starts the measurement
                seen[p.pid] = cached
                cpu = cached.cpu_percent(None)
                label = _app_label(p)
                if label or cpu > 5:
                    procs.append({"name": label or (p.info.get("name") or "?"), "ours": bool(label),
                                  "cpu_percent": round(cpu, 1),
                                  "cores": round(cpu / 100, 1),
                                  "memory_mb": round((p.info.get("memory_info").rss if p.info.get("memory_info") else 0) / 1e6)})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        _STATE["procs"] = seen
    out["processes"] = sorted(procs, key=lambda x: (not x["ours"], -x["cpu_percent"]))[:12]
    return out
