"""Find, open and drive the processing programs: Siril and GraXpert for deep sky, PIPP, AutoStakkert,
RegiStax and ImPPG for the Moon."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from . import config

IS_WIN, IS_MAC = sys.platform == "win32", sys.platform == "darwin"


def _first_existing(paths) -> str | None:
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    return None


def find_graxpert() -> str | None:
    custom = config.load().get("graxpert_path")
    if custom:
        return custom if Path(custom).exists() else None
    if IS_WIN:
        local = os.environ.get("LOCALAPPDATA", "")
        return _first_existing([
            Path(local) / "Programs" / "GraXpert" / "GraXpert.exe",
            Path(r"C:\Program Files\GraXpert\GraXpert.exe"),
            Path(r"C:\Program Files (x86)\GraXpert\GraXpert.exe"),
        ])
    if IS_MAC:
        return _first_existing([Path("/Applications/GraXpert.app/Contents/MacOS/GraXpert")])
    return shutil.which("graxpert") or shutil.which("GraXpert") or _first_existing(
        sorted((Path.home() / "Applications").glob("GraXpert*.AppImage")))


def find_siril() -> dict:
    """Paths to the Siril GUI and to siril-cli (used for scripts)."""
    custom = config.load().get("siril_path")
    if custom:
        gui = custom if Path(custom).exists() else None
        cli = None
        if gui:
            p = Path(gui)
            cand = p.with_name("siril-cli.exe" if IS_WIN else "siril-cli")
            cli = str(cand) if cand.exists() else None
        return {"gui": gui, "cli": cli}
    if IS_WIN:
        base = Path(r"C:\Program Files\Siril\bin")
        return {"gui": _first_existing([base / "siril.exe"]), "cli": _first_existing([base / "siril-cli.exe"])}
    if IS_MAC:
        base = Path("/Applications/Siril.app/Contents/MacOS")
        return {"gui": _first_existing([base / "siril", base / "Siril"]), "cli": _first_existing([base / "siril-cli"])}
    if shutil.which("siril"):
        return {"gui": shutil.which("siril"), "cli": shutil.which("siril-cli")}
    if shutil.which("flatpak"):
        try:
            out = subprocess.run(["flatpak", "info", "org.siril.Siril"], capture_output=True, timeout=10)
            if out.returncode == 0:
                return {"gui": "flatpak:org.siril.Siril", "cli": "flatpak:org.siril.Siril"}
        except (OSError, subprocess.SubprocessError):
            pass
    return {"gui": None, "cli": None}


# Moon programs and Dwarfium are Windows apps, often unzipped wherever the download landed,
# so look in the usual install folders and a few likely download spots.
PROGRAMS = {
    "dwarfium": {"label": "Dwarfium", "globs": ["Dwarfium/Dwarfium.exe", "Dwarfium/dwarfium.exe"]},
    "pipp": {"label": "PIPP", "globs": ["PIPP*/pipp*.exe", "PIPP*/*/pipp*.exe"]},
    "autostakkert": {"label": "AutoStakkert", "globs": ["AutoStakkert*/AutoStakkert*.exe", "AS*/AutoStakkert*.exe"]},
    "registax": {"label": "RegiStax", "globs": ["RegiStax*/RegiStax*.exe"]},
    "imppg": {"label": "ImPPG", "globs": ["ImPPG*/bin/imppg*.exe", "ImPPG*/imppg*.exe", "ImPPG*/bin/impp*.exe"]},
}


def _search_roots() -> list[Path]:
    """Install folders first, then the places a download usually lands."""
    home = Path.home()
    roots = []
    if IS_WIN:
        roots += [Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
                  Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
                  Path(os.environ.get("LOCALAPPDATA", "")) / "Programs", Path(os.environ.get("LOCALAPPDATA", ""))]
    roots += [home / "Desktop", home / "OneDrive" / "Desktop", home / "Documents", home / "Downloads"]
    if IS_WIN:
        roots.append(Path("C:/"))
    return [r for r in roots if str(r) and r.is_dir()]


# Installers, uninstallers and helper programs that sit next to the real one (PIPP ships pipp_core.exe).
_NOT_THE_APP = ("install", "setup", "uninst", "unins", "update", "_core", "-core", "crash", "helper")


def _best_exe(hits: list[Path], label: str) -> Path | None:
    hits = [h for h in hits if not any(w in h.name.lower() for w in _NOT_THE_APP)]
    if not hits:
        return None
    want = label.lower().replace(" ", "")
    # The exact name (PIPP.exe) first, then the shortest name, then the newest version folder.
    return sorted(hits, key=lambda h: (h.stem.lower() != want, len(h.stem), [-ord(c) for c in str(h.parent).lower()]))[0]


_FOUND: dict[str, str | None] = {}


def find_tool(key: str) -> str | None:
    """Locate Dwarfium, a Moon program, or the user's own extra program, from Settings or by searching."""
    custom = config.load().get(f"{key}_path")
    if custom:
        return custom if Path(custom).exists() else None
    if key not in PROGRAMS:
        return None
    if key not in _FOUND or (_FOUND[key] and not Path(_FOUND[key]).exists()):
        _FOUND[key] = None
        for root in _search_roots():
            for pattern in PROGRAMS[key]["globs"]:
                try:
                    hit = _best_exe(list(root.glob(pattern)), PROGRAMS[key]["label"])
                except OSError:
                    hit = None
                if hit:
                    _FOUND[key] = str(hit)
                    break
            if _FOUND[key]:
                break
    return _FOUND[key]


def status() -> dict:
    siril = find_siril()
    out = {"graxpert": find_graxpert(), "siril": siril["gui"], "siril_cli": siril["cli"]}
    for key in PROGRAMS:
        out[key] = find_tool(key)
    from .platesolve import find_astap
    out["astap"] = find_astap()
    out["other_app"] = find_tool("other_app")
    out["other_app_name"] = config.load().get("other_app_name") or "Other app"
    return out


def _popen_detached(args: list[str], cwd: str | None = None) -> None:
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "cwd": cwd}
    if IS_WIN:
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(args, **kwargs)


def _app_bundle(exe: str) -> str | None:
    p = Path(exe)
    for parent in p.parents:
        if parent.suffix == ".app":
            return str(parent)
    return None


def open_in(tool: str, file_path: str) -> None:
    """Open the image in the tool's window, so the suggested settings can be applied by hand."""
    if tool == "graxpert":
        exe = find_graxpert()
    elif tool == "siril":
        exe = find_siril()["gui"]
    elif tool in PROGRAMS or tool == "other_app":
        exe = find_tool(tool)
    elif tool == "folder":
        folder = str(Path(file_path).parent)
        _popen_detached(["explorer", folder] if IS_WIN else ["open", folder] if IS_MAC else ["xdg-open", folder])
        return
    else:
        raise ValueError("Unknown tool")
    if not exe:
        name = PROGRAMS.get(tool, {}).get("label") or tool.capitalize()
        raise FileNotFoundError(f"{name} wasn't found. Set its location in Settings.")
    if tool == "dwarfium":
        _popen_detached([exe])  # Dwarfium talks to the telescope itself; it doesn't open files
    elif exe.startswith("flatpak:"):
        _popen_detached(["flatpak", "run", exe.split(":", 1)[1], file_path])
    elif IS_MAC and _app_bundle(exe):
        _popen_detached(["open", "-a", _app_bundle(exe), file_path])
    else:
        # Started from its own folder: some Windows programs (PIPP among them) look for their files there.
        try:
            _popen_detached([exe, file_path], cwd=str(Path(exe).parent))
        except OSError as e:
            name = PROGRAMS.get(tool, {}).get("label") or tool.capitalize()
            raise FileNotFoundError(f"Couldn't start {name} ({exe}): {e}. Check its location in Settings.")


# ---------------------------------------------------------------- background jobs

JOBS: dict[str, dict] = {}


def _run_job(job_id: str, steps: list[tuple[str, callable, str | None]], result_path: callable) -> None:
    job = JOBS[job_id]
    try:
        for label, make_args, cwd in steps:
            job["log"].append(f"$ {label}")
            args = make_args()
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd,
                                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            job["proc"] = proc
            try:
                stdout, stderr = proc.communicate(timeout=1800)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            job.pop("proc", None)
            tail = (stdout or "")[-3000:] + (stderr or "")[-2000:]
            job["log"].append(tail.strip())
            if job["state"] == "stopped":
                return
            if proc.returncode != 0:
                raise RuntimeError(f"{label} exited with code {proc.returncode}")
        job["output"] = result_path()
        job["state"] = "done"
    except Exception as e:  # report every failure back to the page
        if job["state"] != "stopped":
            job["state"] = "failed"
            job["error"] = str(e)


def stop_jobs() -> int:
    """Kill every GraXpert/Siril run still going. Returns how many were stopped."""
    stopped = 0
    for job in list(JOBS.values()):
        proc = job.get("proc")
        if job["state"] == "running" and proc and proc.poll() is None:
            job["state"], job["error"] = "stopped", "Stopped by Stop all."
            proc.kill()
            stopped += 1
    return stopped


def _start(steps, result_path) -> str:
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"state": "running", "log": [], "output": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, steps, result_path), daemon=True).start()
    return job_id


def start_task(fn, label: str) -> str:
    """Run a Python function in the background and report it like the program runs."""
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"state": "running", "log": [label], "output": None, "error": None}

    def run():
        job = JOBS[job_id]
        try:
            job["output"] = fn()
            job["state"] = "done"
        except Exception as e:  # report every failure back to the page
            job["state"], job["error"] = "failed", str(e)

    threading.Thread(target=run, daemon=True).start()
    return job_id


def _newest_with_stem(folder: Path, stem: str) -> str | None:
    hits = sorted(folder.glob(stem + ".*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(hits[0]) if hits else None


def run_graxpert(file_path: str, gx: dict) -> str:
    """Run GraXpert's command line with the recommended settings. Writes new files and never overwrites the original."""
    exe = find_graxpert()
    if not exe:
        raise FileNotFoundError("GraXpert wasn't found. Set its location in Settings.")
    src = Path(file_path)
    folder = src.parent

    def from_input(path_fn, out_stem, cmd_args):
        return lambda: [exe, path_fn(), "-cli", "-output", str(folder / out_stem)] + cmd_args

    steps, stem = [], None
    if gx.get("run_background"):
        stem = src.stem + "_gx_bg"
        steps.append(("GraXpert background extraction", from_input(lambda: str(src), stem, [
            "-cmd", "background-extraction", "-correction", gx.get("correction", "Subtraction"),
            "-smoothing", f"{gx['smoothing']:.2f}"]), None))
    if gx.get("run_denoise"):
        prev = stem
        stem = (prev or src.stem + "_gx") + "_dn"
        # GraXpert picks the output extension itself, so find the previous step's file when this step starts.
        steps.append(("GraXpert AI denoise", from_input(
            (lambda: _newest_with_stem(folder, prev) or str(src)) if prev else (lambda: str(src)), stem,
            ["-cmd", "denoising", "-strength", f"{gx['denoise_strength']:.2f}"]), None))
    if not steps:
        raise ValueError("The assessment didn't recommend a GraXpert step for this image.")
    final = stem
    return _start(steps, lambda: _newest_with_stem(folder, final))


_ARG_OK = re.compile(r"^[A-Za-z0-9 .,=_+\-:]{0,120}$")


def build_siril_script(file_path: str, commands: list[dict]) -> str:
    from .assess import SIRIL_COMMANDS

    src = Path(file_path)
    lines = ["# Written by DWARF Overview. Review before running.", "requires 1.2.0", f'load "{src.name}"']
    saved = False
    for c in commands:
        name, args = c.get("command"), (c.get("args") or "").strip()
        if name not in SIRIL_COMMANDS or not _ARG_OK.match(args):
            lines.append(f"# skipped unsafe or unknown command: {name}")
            continue
        if name in ("save", "savetif"):
            args = re.sub(r"[^A-Za-z0-9_\-]", "", args.split(" ")[0]) or f"{src.stem}_siril"
            saved = True
        if c.get("comment"):
            lines.append("# " + c["comment"].replace("\n", " ")[:200])
        lines.append(f"{name} {args}".rstrip())
    if not saved:
        lines += [f"save {src.stem}_siril", f"savetif {src.stem}_siril"]
    return "\n".join(lines) + "\n"


def run_siril_script(file_path: str, script: str) -> str:
    cli = find_siril()["cli"]
    if not cli:
        raise FileNotFoundError("siril-cli wasn't found. Set Siril's location in Settings.")
    src = Path(file_path)
    script_path = src.with_name(src.stem + "_overview.ssf")
    script_path.write_text(script, encoding="utf-8")
    if cli.startswith("flatpak:"):
        args = ["flatpak", "run", "--command=siril-cli", cli.split(":", 1)[1], "-d", str(src.parent), "-s", str(script_path)]
    else:
        args = [cli, "-d", str(src.parent), "-s", str(script_path)]
    return _start([("siril-cli " + script_path.name, lambda: args, str(src.parent))],
                  lambda: _newest_with_stem(src.parent, src.stem + "_siril"))
