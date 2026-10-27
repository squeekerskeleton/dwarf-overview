"""Measure this computer once at startup and tune the processing tools to it.

The app runs on different laptops and desktops, so nothing is fixed to one machine: at startup it measures the
processor (threads, and a short speed test), memory, the graphics card, and how fast the processing drive
writes, then works out:

- how many threads Siril may use (all but one, so the app and the browser stay responsive) and what share of
  free memory it may take;
- whether GraXpert runs its AI on the graphics card, and its batch size from the card's memory;
- how many frames to copy at once from the DWARF;
- whether Siril and GraXpert get a raised priority (only when there are cores to spare).

The result is kept in memory and saved as hw_profile.json in the data folder (shown in the Network panel).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import config

_PROFILE: dict | None = None
_LOCK = threading.Lock()


def _gpu_info() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5,
                                 creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0).stdout
            name, mem = [x.strip() for x in out.strip().splitlines()[0].split(",")][:2]
            return {"name": name, "vram_gb": round(float(mem) / 1024, 1), "vendor": "NVIDIA"}
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
    if sys.platform == "win32":
        # Other cards (AMD, Intel): GraXpert can still use them through DirectML.
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json"],
                                 capture_output=True, text=True, timeout=15,
                                 creationflags=subprocess.CREATE_NO_WINDOW).stdout
            cards = json.loads(out or "[]")
            cards = cards if isinstance(cards, list) else [cards]
            real = [c for c in cards if c.get("Name") and not any(w in c["Name"].lower() for w in ("virtual", "basic", "remote"))]
            if real:
                c = max(real, key=lambda c: c.get("AdapterRAM") or 0)
                return {"name": c["Name"], "vram_gb": round((c.get("AdapterRAM") or 0) / 2**30, 1),
                        "vendor": "AMD" if "amd" in c["Name"].lower() or "radeon" in c["Name"].lower() else "Intel"
                        if "intel" in c["Name"].lower() else "other"}
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return None


def _cpu_score() -> float:
    """A short single-thread speed test (about 0.3 s): millions of simple float operations per second."""
    import numpy as np
    a = np.random.rand(1_000_000).astype("float32")
    t, n = time.perf_counter(), 0
    while time.perf_counter() - t < 0.3:
        a = np.sqrt(a * 1.0001 + 0.5)
        n += 1
    return round(n * 2 / (time.perf_counter() - t), 1)


def _disk_mb_s(folder: Path) -> float | None:
    """Write and read back 64 MB in the processing folder to see how fast that drive is."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / ".overview_speedtest"
        block = os.urandom(1 << 20)
        t = time.perf_counter()
        with open(f, "wb") as out:
            for _ in range(64):
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        speed = 64 / max(1e-3, time.perf_counter() - t)
        f.unlink(missing_ok=True)
        return round(speed)
    except OSError:
        return None


def _drive_tip(work: Path) -> str:
    """When the processing folder is on the system drive, suggest a roomier other drive (an SSD is best)."""
    import psutil
    try:
        sysdrive = Path(os.environ.get("SystemDrive", "/") + os.sep).anchor.lower()
        if Path(work).resolve().anchor.lower() != sysdrive:
            return ""
        others = []
        for part in psutil.disk_partitions(all=False):
            root = part.mountpoint
            if Path(root).anchor.lower() == sysdrive or "cdrom" in part.opts or not part.fstype:
                continue
            try:
                others.append((psutil.disk_usage(root).free, root))
            except OSError:
                continue
        if not others:
            return ""
        free, root = max(others)
        if free < 100e9:
            return ""
        return (f" The processing folder is on the system drive; {root} has {free / 1e9:.0f} GB free. Setting "
                f"Processing folder to e.g. {Path(root) / 'IN-PROCESS'} (fastest if it's an SSD) and Finished runs "
                "folder to a big hard drive keeps the system drive free.")
    except Exception:
        return ""


def measure() -> dict:
    import psutil
    from .autoprocess import work_root
    threads = psutil.cpu_count(logical=True) or os.cpu_count() or 2
    cores = psutil.cpu_count(logical=False) or max(1, threads // 2)
    mem = psutil.virtual_memory()
    gpu = _gpu_info()
    work = work_root()
    disk = _disk_mb_s(work)
    ram_gb = mem.total / 1e9

    siril_threads = max(1, threads - 1) if threads > 2 else threads
    # Siril's memory setting is a share of the memory that's free when it runs.
    siril_mem = 0.9 if ram_gb >= 30 else 0.8 if ram_gb >= 15 else 0.65
    gx_gpu = bool(gpu) and (gpu["vendor"] in ("NVIDIA", "AMD") or (gpu.get("vram_gb") or 0) >= 2)
    vram = (gpu or {}).get("vram_gb") or 0
    gx_batch = 16 if vram >= 12 else 8 if vram >= 7 else 4 if vram >= 3 else 2
    copies = 4 if (disk or 0) >= 300 else 3 if (disk or 0) >= 80 else 2
    raised = threads >= 8
    profile = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": {"threads": threads, "cores": cores, "score": _cpu_score(), "name": _cpu_name()},
        "memory_gb": round(ram_gb, 1), "gpu": gpu, "work_folder": str(work), "work_disk_mb_s": disk,
        "tuning": {"siril_threads": siril_threads, "siril_mem_ratio": siril_mem, "graxpert_gpu": gx_gpu,
                   "graxpert_batch": gx_batch, "parallel_copies": copies, "raised_priority": raised,
                   "opencv_threads": max(2, min(8, threads // 2))},
    }
    profile["summary"] = (f"Tuned for this computer: Siril uses {siril_threads} of {threads} threads and up to "
                          f"{int(siril_mem * 100)}% of free memory; GraXpert AI on "
                          f"{'the graphics card (' + gpu['name'] + ')' if gx_gpu else 'the processor'}; "
                          f"{copies} frames copied at a time" + ("; processing runs at raised priority" if raised else "") + ".") + _drive_tip(work)
    try:
        (config.data_dir() / "hw_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    except OSError:
        pass
    return profile


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or "processor"


DEFAULT_TUNING = {"siril_threads": max(1, (os.cpu_count() or 2) - 1), "siril_mem_ratio": 0.7, "graxpert_gpu": False,
                  "graxpert_batch": 4, "parallel_copies": 2, "raised_priority": False,
                  "opencv_threads": max(2, min(8, (os.cpu_count() or 4) // 2))}


def start() -> None:
    """Measure in the background at startup."""
    def run():
        global _PROFILE
        try:
            p = measure()
        except Exception as e:  # never stop the app over this
            p = {"error": str(e), "tuning": DEFAULT_TUNING, "summary": "Couldn't measure this computer; using safe defaults."}
        with _LOCK:
            _PROFILE = p
    threading.Thread(target=run, daemon=True).start()


def profile() -> dict:
    with _LOCK:
        return _PROFILE or {"pending": True, "tuning": DEFAULT_TUNING, "summary": "Measuring this computer…"}


def tuning() -> dict:
    return profile()["tuning"]


def priority_flags() -> int:
    """Windows creation flags for a processing program: no console window, raised priority when there's room."""
    if sys.platform != "win32":
        return 0
    flags = subprocess.CREATE_NO_WINDOW
    if tuning().get("raised_priority"):
        flags |= subprocess.ABOVE_NORMAL_PRIORITY_CLASS
    return flags
