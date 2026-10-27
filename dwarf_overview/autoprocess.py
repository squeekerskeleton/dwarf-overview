"""Auto-apply a processing plan: restack every raw frame of the session behind the photo, then run the plan.

The DWARF's own stack (stacked-*.fits) is what the Assess tab measures. Its raw frames sit in the same session
folder on the telescope (and in the laptop library only if they were copied). Auto-apply:

1. finds every raw frame of that session: the library folder, the work folder, the DWARF over USB-C, or the DWARF
   over Wi-Fi. Frames the DWARF rejected (failed_*) are left out unless asked for;
2. copies the missing ones into the processing folder (Settings), after checking there's room;
   It also looks for the DWARF's own master dark (CALI_FRAME) for that exposure and gain (darks.py);
3. Siril: dark subtraction and cosmetic correction (when a dark fits), debayer, register (2-pass), keep the best
   frames (FWHM and roundness), stack with Winsorized rejection (weighted by sharpness on Siril 1.4), 32-bit;
4. crops the stacking edges;
5. GraXpert background extraction;
6. plate solve (Siril, offline with its local catalogues; ASTAP as the fallback) and colour calibration: SPCC
   (narrowband mode for the Duo-Band filter), else PCC, else balanced from the image's own stars;
7. deconvolution (GraXpert AI, else Siril Richardson-Lucy with a PSF from the stars);
8. noise reduction (GraXpert AI, else Siril's NL-Bayes), strength from how much integration there is;
9. Siril: the plan's own commands, a linked auto-stretch and saturation;
10. star reduction (StarNet when installed, else a star mask and erosion), an HOO version for Duo-Band, and
   TIFF, PNG, JPG and FITS (finishing.py).
Settings the plan doesn't give are chosen from the data (frame count, integration, filter, object type).

Nothing on the DWARF or in the library is changed or deleted; everything is written to the processing folder.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from . import config, dwarf, finishing, history, hwprofile, scope, tools

FRAME = re.compile(r"\.(fits?|fts)$", re.I)
NOT_A_FRAME = re.compile(r"^(stacked|thumbnail|result_|overview_)|_(gx|siril|crop|bg|dn)", re.I)
FAILED_DIR = "failed_frames"
# Debayered 16-bit RGB is three times a raw frame; Siril keeps the converted and the registered copies.
SPACE_FACTOR = 7.0


class ProcessError(Exception):
    pass


def work_root() -> Path:
    custom = (config.load().get("work_folder") or "").strip()
    return Path(custom).expanduser() if custom else dwarf.library_dir() / "_processing"


def finished_root() -> Path | None:
    """Where finished runs are moved (Settings: Finished runs folder), so the fast processing drive stays free.
    Empty = finished runs stay in the processing folder, as before."""
    custom = (config.load().get("finished_folder") or "").strip()
    if not custom:
        return None
    root = Path(custom).expanduser()
    try:
        if root.resolve() == work_root().resolve():
            return None
    except OSError:
        return None
    return root


def _finished_dir(name: str) -> Path | None:
    root = finished_root()
    return root / _safe(name) if root else None


def _repath(obj, old: str, new: str):
    if isinstance(obj, str):
        return new + obj[len(old):] if obj == old or obj.startswith((old + "/", old + "\\")) else obj
    if isinstance(obj, list):
        return [_repath(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: _repath(v, old, new) for k, v in obj.items()}
    return obj


def _merge_move(src: Path, dst: Path) -> None:
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            _merge_move(child, dst / child.name)
        try:
            src.rmdir()
        except OSError:
            pass
    else:
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))


def archive(j, work: Path, out: dict, drop: tuple = ("process",)) -> dict:
    """A finished run: delete Siril's working copies, then move everything else (frames, stack, results, run
    record) from the processing folder to the finished-runs folder, and point the results at their new place."""
    dest_root = finished_root()
    if not dest_root or not work.is_dir():
        return out
    dest = dest_root / work.name
    try:
        if dest.resolve() == work.resolve():  # it already ran there (the processing drive was short of room)
            return out
    except OSError:
        return out
    for d in drop:
        shutil.rmtree(work / d, ignore_errors=True)
    need = _folder_bytes(work)
    try:
        dest_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(dest_root).free
    except OSError as e:
        j.log(f"Couldn't reach the finished-runs folder ({e}); the results stay in {work}.")
        return out
    if free < need + 1e9:
        j.log(f"The finished-runs folder has only {free / 1e9:.0f} GB free; the results stay in {work}.")
        return out
    j.log(f"Moving the finished run ({need / 1e9:.1f} GB) to {dest}…")
    try:
        _merge_move(work, dest)
    except OSError as e:
        j.log(f"Moving stopped part-way ({e}); some files are still in {work}.")
        return out
    if j.folder:
        j.folder = dest
    j.log(f"Finished run moved to {dest}.")
    out = _repath(out, str(work), str(dest))
    if isinstance(out, dict) and "cleanup_bytes" in out:
        out["cleanup_bytes"] = 0  # Siril's working copies are already gone
    return out


def _session_of(preview: Path) -> tuple[str, Path]:
    """The session folder name (DWARF_RAW_..._EXP_.._GAIN_.._<time>) and its local folder."""
    for folder in [preview.parent] + list(preview.parents)[:2]:
        if (folder / "shotsInfo.json").exists() or folder.name.upper().startswith("DWARF_RAW"):
            return folder.name, folder
    return preview.parent.name, preview.parent


def _is_frame(name: str, include_failed: bool) -> bool:
    if not FRAME.search(name) or NOT_A_FRAME.search(name):
        return False
    return include_failed or not name.lower().startswith("failed")


def _size(text: str) -> int:
    m = re.match(r"([\d.]+)\s*([KMG]?)", text or "", re.I)
    if not m:
        return 0
    return int(float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2).upper()])


def find_frames(preview_path: str, include_failed: bool = False) -> dict:
    """Every raw frame of the session behind this photo, and where each one can be read from."""
    preview = Path(preview_path)
    name, local = _session_of(preview)
    lights = work_root() / _safe(name) / "lights"
    frames: dict[str, dict] = {}

    def add(fname, source, where, size):
        if _is_frame(fname, include_failed) and fname not in frames:
            frames[fname] = {"name": fname, "source": source, "where": where, "size": size}

    held = lights.parent / FAILED_DIR  # rejected frames kept out of the stack by the last screening
    done = _finished_dir(name)
    places = [(lights, "processing folder"), (held, "processing folder")]
    if done:  # an earlier finished run's frames, moved off the processing drive
        places += [(done / "lights", "finished runs folder"), (done / FAILED_DIR, "finished runs folder")]
    for folder, source in places + [(local, "laptop library")]:
        if folder.is_dir():
            for f in folder.iterdir():
                if f.is_file():
                    add(f.name, source, str(f), f.stat().st_size)
    remote = None
    for usb in scope.usb_folders():
        for cand in [Path(usb) / name] + [d / name for d in Path(usb).iterdir() if d.is_dir()]:
            if cand.is_dir():
                remote = "USB-C"
                for f in cand.iterdir():
                    if f.is_file():
                        add(f.name, "DWARF (USB-C)", str(f), f.stat().st_size)
                break
    if remote is None:
        for root in dwarf.ROOTS.values():
            try:
                listing = dwarf.session_files(root, name)
            except (dwarf.DwarfError, OSError):
                continue
            remote = "Wi-Fi"
            for e in listing["files"]:
                add(e["name"], "DWARF (Wi-Fi)", f"{root}{name}/{e['name']}", _size(e["size"]))
            break
    shown = sorted(frames.values(), key=lambda f: f["name"])
    failed = sum(1 for f in shown if f["name"].lower().startswith("failed"))
    total = sum(f["size"] for f in shown)
    to_copy = sum(f["size"] for f in shown if f["source"] != "processing folder")
    free = shutil.disk_usage(_existing_parent(work_root())).free
    need = to_copy + total * SPACE_FACTOR
    work, fallback = work_root() / _safe(name), ""
    if free <= need and done:  # the fast drive is short: work on the finished-runs drive instead (slower)
        try:
            done_free = shutil.disk_usage(_existing_parent(done)).free
        except OSError:
            done_free = 0
        if done_free > need:
            work, free = done, done_free
            fallback = (f"Not enough room in the processing folder, so this run works in {done} instead "
                        "(slower, but it fits).")
    return {"session": name, "frames": shown, "count": len(shown), "failed": failed, "dwarf_reached": remote,
            "has_stack": (work_root() / _safe(name) / "result_stack.fits").exists()
                         or bool(done and (done / "result_stack.fits").exists()),
            "bytes": total, "bytes_to_copy": to_copy, "bytes_needed": int(need), "bytes_free": free,
            "fits": free > need, "work_folder": str(work), "fallback": fallback}


def _existing_parent(p: Path) -> Path:
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def _safe(name: str) -> str:
    return re.sub(r'[<>:"|?*\\/]', "_", name)


# ---------------------------------------------------------------- plan settings

def _plan_settings(a: dict) -> dict:
    """Read the numbers the plan gives (frame filtering, rejection, crop, green) with safe defaults."""
    text = json.dumps(a.get("steps", []))
    out = {"keep_percent": 75, "roundness": 0.75, "sigma_low": 3.0, "sigma_high": 3.0, "crop_px": 50,
           "rmgreen": None, "focal": 150, "pixel": 2.0, "_from_plan": set()}
    for s in a.get("steps", []):
        blob = " ".join([s.get("action", ""), s.get("menu_path", "")] + [f"{x['name']} {x['value']}" for x in s.get("settings", [])])
        low = blob.lower()
        if "fwhm" in low and (m := re.search(r"best\s*(\d{2})\s*%", low)):
            out["keep_percent"] = min(95, max(40, int(m.group(1))))
            out["_from_plan"].add("keep_percent")
        if m := re.search(r"roundness\D{0,12}(0\.\d+)", low):
            out["roundness"] = min(0.95, max(0.3, float(m.group(1))))
        if "rejection" in low and (m := re.search(r"sigma\s*([\d.]+)\s*/\s*([\d.]+)", low)):
            out["sigma_low"], out["sigma_high"] = float(m.group(1)), float(m.group(2))
            out["_from_plan"].add("sigma")
        if "crop" in low and (m := re.findall(r"(\d{2,3})\s*(?:to|-)?\s*(\d{2,3})?\s*px", low)):
            nums = [int(x) for x in m[0] if x]
            out["crop_px"] = min(300, max(0, sum(nums) // len(nums)))
        if "remove green" in low or "rmgreen" in low:
            out["rmgreen"] = not s.get("action", "").lower().startswith("skip")
        if m := re.search(r"focal length\D{0,6}(\d{2,4})\s*mm", low):
            out["focal"] = int(m.group(1))
        if m := re.search(r"pixel size\D{0,6}([\d.]+)\s*(?:µ|u)m", low):
            out["pixel"] = float(m.group(1))
    if out["rmgreen"] is None:
        out["rmgreen"] = '"rmgreen"' in json.dumps(a.get("siril_script", []))
    return out


# ---------------------------------------------------------------- the job

def _siril_args(script: Path, workdir: Path) -> list[str]:
    cli = tools.find_siril()["cli"]
    if not cli:
        raise ProcessError("siril-cli wasn't found. Set Siril's location in Settings.")
    if cli.startswith("flatpak:"):
        return ["flatpak", "run", "--command=siril-cli", cli.split(":", 1)[1], "-d", str(workdir), "-s", str(script)]
    return [cli, "-d", str(workdir), "-s", str(script)]


RUN_FILE = "run.json"


class _Job:
    def __init__(self, steps: list[str], folder: Path | None = None, meta: dict | None = None):
        self.id = uuid.uuid4().hex[:10]
        self.folder, self.meta = folder, dict(meta or {})
        self.outputs: dict[str, str] = {}
        self.job = tools.JOBS[self.id] = {"state": "running", "log": [], "output": None, "error": None,
                                          "steps": [{"label": s, "state": "waiting", "note": "", "fraction": 0.0}
                                                    for s in steps],
                                          "progress": 0.0, "started": time.time(), "eta_seconds": None}
        self.weights = list(WEIGHTS)
        self.paused_for = 0.0
        self.save()

    def save(self, status: str | None = None) -> None:
        """Keep run.json in the session's processing folder up to date, so LAST SESSION can pick the run up
        again after the app is closed, updated or the computer restarts."""
        if not self.folder:
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            data = {**self.meta, "job_id": self.id, "status": status or self.job["state"],
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "progress": self.job["progress"],
                    "steps": [{k: s[k] for k in ("label", "state", "note")} for s in self.job["steps"]],
                    "outputs": self.outputs, "error": self.job.get("error")}
            tmp = self.folder / (RUN_FILE + ".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tmp.replace(self.folder / RUN_FILE)
        except OSError:
            pass

    def step(self, n: int, state: str, note: str = "") -> None:
        s = self.job["steps"][n]
        s.update(state=state, note=note or s["note"] if state == "running" else note)
        if state in ("done", "skipped"):
            s["fraction"] = 1.0
        if note:
            self.log(note)
        self._progress()
        self.save()

    def part(self, n: int, fraction: float) -> None:
        """How far through step n (0 to 1)."""
        s = self.job["steps"][n]
        s["fraction"] = max(s["fraction"], min(1.0, fraction))
        self._progress()

    def _progress(self) -> None:
        total = sum(self.weights) or 1
        done = sum(w * s["fraction"] for w, s in zip(self.weights, self.job["steps"]))
        p = round(100 * done / total, 1)
        self.job["progress"] = p
        spent = time.time() - self.job["started"] - self.paused_for
        self.job["eta_seconds"] = round(spent * (100 - p) / p) if p >= 3 else None

    def wait_if_paused(self) -> None:
        if tools.PAUSED.is_set():
            t = time.time()
            tools.wait_if_paused()
            self.paused_for += time.time() - t

    def log(self, line: str) -> None:
        self.job["log"] = (self.job["log"] + [line])[-400:]

    def stopped(self) -> bool:
        return self.job["state"] == "stopped"

    def run(self, args: list[str], cwd: Path, timeout: float = 6 * 3600, step: int | None = None,
            phases: list[str] | None = None) -> tuple[int, str]:
        """Run a program, logging its output. With phases (the script's commands in order), progress within
        `step` follows which command Siril is on and the percentage it prints."""
        self.wait_if_paused()
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(cwd),
                                errors="replace", creationflags=hwprofile.priority_flags())
        self.job["proc"] = proc
        out, end = [], time.monotonic() + timeout
        phase = 0
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                out.append(line)
                pct = re.search(r"(\d{1,3}(?:\.\d+)?)\s?%", line)
                if not re.search(r"^\s*(\d+(\.\d+)?%|progress)", line, re.I):
                    self.log(line[:300])
                if step is not None and phases:
                    low = line.lower()
                    for k in range(phase + 1, len(phases)):
                        if re.search(r"\b" + re.escape(phases[k]) + r"\b", low):
                            phase = k
                            break
                    within = min(1.0, float(pct.group(1)) / 100) if pct and "%" in line else 0.0
                    self.part(step, (phase + within) / len(phases))
            if time.monotonic() > end + self.paused_for:
                proc.kill()
        proc.wait()
        self.job.pop("proc", None)
        return proc.returncode, "\n".join(out[-60:])


# Share of the whole run each step usually takes (copying is re-weighted by how many frames need copying).
WEIGHTS = [20, 50, 1, 6, 5, 5, 6, 4, 3]
STEPS = ["Find and copy the raw frames and the DWARF's matching dark",
         "Siril: calibrate, debayer, register, select frames, stack",
         "Stacking edges (full frame, or cropped)", "GraXpert: background extraction",
         "Plate solve and colour calibration", "Deconvolution (sharpen fine detail)", "Noise reduction",
         "Siril: stretch, saturation and the plan's own steps", "Star reduction and final save"]
# Steps of runs saved by older versions, so LAST SESSION still recognises them.
OLD_LABELS = {"Find and copy the raw frames": 0, "Siril: debayer, register, select frames, stack": 1,
              "Crop the stacking edges": 2}
# DWARF 3 Duo-Band passbands: Hα about 18 nm and OIII about 30 nm wide (the values the Siril community's smart-telescope
# preprocessing script uses for the DWARF 3, and what Siril's measured curve of the sister DWARF Mini filter shows).
DUO_BAND = {"rwl": 656.28, "rbw": 18, "gwl": 500.7, "gbw": 30, "bwl": 500.7, "bbw": 30}
# The DWARF 3's "Astro" filter is a UV/IR cut; Siril's SPCC database names it like this.
ASTRO_FILTER = "UV/IR Block"
DWARF3_SENSOR = "Sony IMX678"


def start(assessment_id: int, include_failed: bool = False, resume: dict | None = None, full_frame: bool = True,
          reuse_stack: bool = False) -> str:
    a = history.get_assessment(int(assessment_id))
    if not a:
        raise ProcessError("Assess the photo first, so there's a plan to apply.")
    found = find_frames(a["file_path"], include_failed)
    if reuse_stack and not resume:
        # Redo only the finishing: carry the last run's stack over as if resuming after the stacking step.
        stack = Path(found["work_folder"]) / "result_stack.fits"
        done = _finished_dir(found["session"])
        if not stack.exists() and done and (done / "result_stack.fits").exists():
            # The last run was moved to the finished-runs folder: bring its stack back to the fast drive.
            stack.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(done / "result_stack.fits", stack)
            if (done / RUN_FILE).exists() and not (stack.parent / RUN_FILE).exists():
                shutil.copy2(done / RUN_FILE, stack.parent / RUN_FILE)
        if not stack.exists():
            raise ProcessError("There's no earlier stack for this session to reuse; untick that option.")
        try:
            prev = json.loads((Path(found["work_folder"]) / RUN_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev = {}
        resume = {"steps": [{"label": STEPS[1], "state": "done", "note": "Reused the last stack"}],
                  "outputs": {"stacked": str(stack)}, "started": None,
                  "full_frame": prev.get("full_frame", full_frame)}
    if resume:  # the earlier run's working files will be reused or replaced, so count their space as free
        found["bytes_free"] += _folder_bytes(Path(found["work_folder"]) / "process")
        found["fits"] = found["bytes_free"] > found["bytes_needed"] or bool(resume.get("outputs", {}).get("stacked"))
    if found["count"] < 3:
        where = "" if found["dwarf_reached"] else " The DWARF wasn't reachable over Wi-Fi or USB-C, so its frames couldn't be listed."
        raise ProcessError(f"Only {found['count']} raw frames were found for {found['session']}.{where}")
    if not found["fits"]:
        raise ProcessError(f"Not enough room: this needs about {found['bytes_needed'] / 1e9:.0f} GB and "
                           f"{found['work_folder'][:3]} has {found['bytes_free'] / 1e9:.0f} GB free. Pick a bigger "
                           "drive for the processing folder in Settings.")
    meta = {"assessment_id": int(assessment_id), "include_failed": include_failed, "full_frame": full_frame,
            "session": found["session"],
            "photo": a["file_path"], "frames": found["count"],
            "started": (resume or {}).get("started") or time.strftime("%Y-%m-%d %H:%M:%S")}
    j = _Job(STEPS, Path(found["work_folder"]), meta)
    threading.Thread(target=_run, args=(j, a, found, resume), daemon=True).start()
    return j.id


def _run(j: _Job, a: dict, found: dict, resume: dict | None = None) -> None:
    try:
        out = _pipeline(j, a, found, resume)
        if not j.stopped():
            out = archive(j, Path(found["work_folder"]), out)
            j.job.update(state="done", output=out)
    except Exception as e:  # every failure goes back to the page
        if not j.stopped():
            for s in j.job["steps"]:
                if s["state"] == "running":
                    s["state"] = "failed"
            j.job.update(state="failed", error=str(e))
    j.save()


def _copy_frames(j: _Job, found: dict, lights: Path) -> int:
    """Copy the missing frames, a few at a time (how many depends on this computer's drive)."""
    from concurrent.futures import ThreadPoolExecutor
    lights.mkdir(parents=True, exist_ok=True)
    todo = [f for f in found["frames"] if f["source"] != "processing folder"]
    if not todo:
        return 0
    base = dwarf._base() if any(f["source"] == "DWARF (Wi-Fi)" for f in todo) else ""
    lock, count = threading.Lock(), [0]

    def one(f):
        j.wait_if_paused()
        if j.stopped():
            raise ProcessError("Stopped.")
        target, tmp = lights / f["name"], lights / (f["name"] + ".part")
        try:
            if f["source"] == "DWARF (Wi-Fi)":
                with urllib.request.urlopen(base + urllib.parse.quote(f["where"]), timeout=60) as r, open(tmp, "wb") as o:
                    while chunk := r.read(1 << 20):
                        o.write(chunk)
                        if tools.PAUSED.is_set():
                            j.wait_if_paused()
            else:
                shutil.copyfile(f["where"], tmp)
            tmp.replace(target)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            raise ProcessError(f"Copying {f['name']} failed: {e}")
        with lock:
            count[0] += 1
            n = count[0]
        j.part(0, n / len(todo))
        if n % 25 == 0 or n == len(todo):
            j.log(f"Copied {n} of {len(todo)} frames from the {f['source']}")

    with ThreadPoolExecutor(max_workers=hwprofile.tuning()["parallel_copies"]) as pool:
        for _ in pool.map(one, todo):
            pass
    return count[0]


_SIRIL_VERSION: dict[str, tuple] = {}


def _siril_version() -> tuple:
    """(major, minor) of siril-cli; 1.4 adds SPCC, offline catalogues and frame weighting."""
    cli = tools.find_siril()["cli"] or ""
    if cli not in _SIRIL_VERSION:
        v = (1, 2)
        if cli and not cli.startswith("flatpak:"):
            try:
                out = subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=20,
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                if m := re.search(r"(\d+)\.(\d+)", out.stdout + out.stderr):
                    v = (int(m.group(1)), int(m.group(2)))
            except (OSError, subprocess.SubprocessError):
                pass
        _SIRIL_VERSION[cli] = v
    return _SIRIL_VERSION[cli]


def _duo_band(shot: dict, found: dict) -> bool:
    text = (json.dumps(shot) + " " + " ".join(f["name"] for f in found["frames"][:5])).lower()
    return any(w in text for w in ("duo", "dual", "dualband", "duo-band", "narrowband"))


def _shot_temp(shot: dict, found: dict) -> float | None:
    for k, v in shot.items():
        if "temp" in k.lower():
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    for f in found["frames"][:3]:
        if m := re.search(r"_(-?\d{1,2})C[_.]", f["name"]):
            return float(m.group(1))
    return None


def _recommend(plan: dict, a: dict, found: dict, shot: dict) -> dict:
    """Settings for the steps the plan doesn't pin down, chosen from the data: how many frames, total
    integration, the filter, and what kind of object it is."""
    assess = a["assessment"]
    gx = assess.get("graxpert") or {}
    m = re.search(r"EXP_([\d.]+)", found["session"], re.I)
    exp = float(m.group(1)) if m else 15.0
    minutes = found["count"] * exp / 60
    kind = ((assess.get("identification") or {}).get("object_type") or "").lower()
    r = {"minutes": minutes, "duo": _duo_band(shot, found), "kind": kind}
    if "keep_percent" not in plan["_from_plan"]:
        # The roundness cut (best 90%) comes on top of this one, so together they keep about 80% of a big session
        # (0.10 kept 75% x 80% = 58% of M 45's frames).
        r["keep_percent"] = 95 if found["count"] < 50 else 90 if found["count"] < 300 else 88
    # Long sessions are cleaner, and GraXpert's denoise softens stars, so it eases off as the data grows.
    r["denoise"] = 0.7 if minutes < 60 else 0.6 if minutes < 180 else 0.5
    if gx.get("run_denoise") and gx.get("denoise_strength") is not None:
        # A plan may ask for more, never much less: plans asked for 0.30, which measurably did nothing on C 33.
        r["denoise"] = max(r["denoise"] - 0.05, min(1.0, float(gx["denoise_strength"])))
    if "cluster" in kind:
        r["stars"] = 0.0  # the stars are the subject
    elif "galaxy" in kind:
        r["stars"] = 0.4
    else:
        r["stars"] = 0.7  # nebulae: let the gas show through the star field
    # Object deconvolution at half strength: the DWARF's 2.75"/px is undersampled, so there's detail to gain in
    # the nebula but not much in the stars, and too much brings out noise.
    r["deconv"] = 0.5
    return r


def _pipeline(j: _Job, a: dict, found: dict, resume: dict | None = None) -> dict:
    plan = _plan_settings(a["assessment"])
    work = Path(found["work_folder"])
    prev_out = (resume or {}).get("outputs") or {}
    prev = {}
    for s in (resume or {}).get("steps") or []:
        n = STEPS.index(s["label"]) if s.get("label") in STEPS else OLD_LABELS.get(s.get("label"))
        if n is not None:
            prev[n] = s

    def reuse(n: int, key: str | None) -> str | bool | None:
        """For LAST SESSION: a step that finished last time and whose file is still there isn't redone."""
        s = prev.get(n)
        if not s or s.get("state") not in ("done", "skipped"):
            return None
        path = prev_out.get(key) if key else None
        if key and s["state"] == "done" and not (path and Path(path).exists()):
            return None  # its file is gone: redo the step
        j.step(n, s["state"], (s.get("note") or "").replace(" (kept from the last session)", "") + " (kept from the last session)")
        if path:
            j.outputs[key] = path
        return path or True

    lights, proc = work / "lights", work / "process"
    try:
        shot = json.loads((_session_of(Path(a["file_path"]))[1] / "shotsInfo.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        shot = {}
    rec = _recommend(plan, a, found, shot)
    full = j.meta.get("full_frame", True)
    plan.update({k: v for k, v in rec.items() if k == "keep_percent"})
    v14 = _siril_version() >= (1, 4)

    # 1. frames, and the DWARF's master dark
    j.weights[0] = max(1, round(WEIGHTS[0] * found["bytes_to_copy"] / max(1, found["bytes"])))
    t = hwprofile.tuning()
    tune = [f"setcpu {t['siril_threads']}", f"setmem {t['siril_mem_ratio']}"]
    j.log(hwprofile.profile().get("summary", ""))
    j.log(f"Siril {'.'.join(map(str, _siril_version()))}; {found['count']} frames, {rec['minutes']:.0f} min of "
          f"integration{', Duo-Band filter (Hα + OIII)' if rec['duo'] else ''}"
          f"{', ' + rec['kind'] if rec['kind'] else ''}.")
    j.step(0, "running", f"{found['count']} frames for {found['session']}" +
           (f" ({found['failed']} the DWARF rejected are included)" if found["failed"] else "") + ".")
    copied = _copy_frames(j, found, lights)
    failed_note = _screen_failed(j, work, lights, j.meta.get("include_failed"))
    dark_note = _master_dark(j, found, shot, work, lights)
    ready = sum(1 for f in lights.iterdir() if f.is_file() and FRAME.search(f.name))
    j.step(0, "done", f"{ready} frames ready to stack in {lights} ({copied} copied). {failed_note} {dark_note}".replace("  ", " "))

    # 2. restack
    stacked = work / "result_stack.fits"
    if not reuse(1, "stacked"):
        _restack(j, plan, work, proc, tune, (work / "master_dark.fits").exists(), v14, full)
        j.outputs["stacked"] = str(stacked)
        j.step(1, "done", f"Stacked: {stacked}")

    # 3. crop
    cropped = work / "result_crop.fits"
    if not reuse(2, "cropped"):
        j.step(2, "running")
        if full:
            j.outputs["cropped"] = str(cropped)
            j.step(2, "done", _fill_empty(stacked, cropped))
        else:
            _crop(stacked, cropped, plan["crop_px"])
            j.outputs["cropped"] = str(cropped)
            j.step(2, "done", f"Cropped {plan['crop_px']} px off each edge.")

    gx = a["assessment"].get("graxpert") or {}
    exe = tools.find_graxpert()

    def carry(n, key, current, run):
        kept = reuse(n, key)
        if kept:
            return Path(kept) if isinstance(kept, str) else current
        return run(current)

    current = cropped
    # 4. GraXpert background extraction
    current = carry(3, "graxpert", current, lambda c: _graxpert(j, gx, exe, c, work, t, rec) if exe else
                    (j.step(3, "skipped", "GraXpert wasn't found (set it in Settings), so background extraction was skipped.") or c))
    # 5. plate solve + colour
    current = carry(4, "calibrated", current, lambda c: _calibrate(j, plan, shot, c, work, tune, rec, v14))
    # 6. deconvolution
    current = carry(5, "deconvolved", current, lambda c: _deconvolve(j, exe, c, work, tune, t, rec))
    # 7. noise reduction
    current = carry(6, "denoised", current, lambda c: _denoise(j, exe, c, work, tune, t, rec))
    # 8. stretch and the plan's own commands
    calibrated = (j.job["steps"][4]["state"] == "done")
    stretched = carry(7, "stretched", current, lambda c: _stretch(j, a, plan, c, work, tune, rec, calibrated))
    # 9. star reduction and saving
    return _finish(j, stretched, work, proc, found, rec)


def _screen_failed(j: _Job, work: Path, lights: Path, include: bool) -> str:
    """The DWARF's rejected frames (failed_*) rank far below the ones it passed: they wait in failed_frames/
    and only the few that beat the passed frames' median quality join the stack (framescore.py)."""
    from . import framescore
    held = work / FAILED_DIR
    for f in list(lights.glob("*")) if lights.is_dir() else []:
        if f.is_file() and f.name.lower().startswith("failed") and FRAME.search(f.name):
            held.mkdir(exist_ok=True)
            f.replace(held / f.name)
    failed = sorted(f for f in held.glob("*") if FRAME.search(f.name)) if held.is_dir() else []
    if not include or not failed:
        return ""
    j.log(f"Checking the {len(failed)} frames the DWARF rejected against a sample of the passed ones…")
    passed = [f for f in lights.iterdir() if f.is_file() and FRAME.search(f.name)]
    accepted, note = framescore.screen_failed(passed, failed)
    for f in accepted:
        f.replace(lights / f.name)
    return note


def _master_dark(j: _Job, found: dict, shot: dict, work: Path, lights: Path) -> str:
    from . import darks
    target = work / "master_dark.fits"
    sample = next((f for f in sorted(lights.iterdir()) if FRAME.search(f.name)), None) if lights.is_dir() else None
    if not sample:
        return ""
    try:
        c = darks.pick(found["session"], _shot_temp(shot, found))
    except Exception as e:  # never stop the run over the dark
        return f"Couldn't look for the DWARF's darks ({e}); stacking without one."
    if not c:
        target.unlink(missing_ok=True)
        return ("No DWARF master dark matches this exposure and gain (CALI_FRAME on the DWARF), so the stack "
                "relies on rejection for hot pixels.")
    try:
        f = darks.fetch(c, work_root() / "_darks")
        why = darks.prepare(f, sample, target)
        return f"Using the DWARF's master dark {c['name']} ({c['stack']} darks at {c['temp']}°C; {why})."
    except Exception as e:
        target.unlink(missing_ok=True)
        return f"Found the DWARF's dark {c['name']} but didn't use it: {e}."


def _restack(j: _Job, plan: dict, work: Path, proc: Path, tune: list[str], dark: bool, v14: bool,
             full: bool = True) -> None:
    # The DWARF is on an alt-az mount, so the sky turns in the frame during a session: the area every frame
    # shares can shrink to a strip over a few hours. Full-frame mode keeps the reference frame's whole field.
    framing = "current" if full else "min"
    found_n = sum(1 for f in (work / "lights").iterdir() if FRAME.search(f.name))
    j.step(1, "running", f"{'Dark-subtracting with cosmetic correction, then k' if dark else 'K'}eeping the sharpest "
                         f"{plan['keep_percent']}% of frames that are also among the roundest 90%; middle frame as reference; {'full original frame size' if full else 'cropped to the area all frames share'}; "
                         f"{'Generalized ESD rejection (catches satellite trails)' if found_n >= 50 and 'sigma' not in plan['_from_plan'] else 'Winsorized rejection ' + str(plan['sigma_low']) + '/' + str(plan['sigma_high'])}"
                         f"{', frames weighted by their sharpness' if v14 else ''}.")
    for use_dark in ([True, False] if dark else [False]):
        if proc.exists():
            shutil.rmtree(proc, ignore_errors=True)
        seq = "pp_light" if use_dark else "light"
        prep = (["convert light -out=../process", "cd ../process",
                 "calibrate light -dark=../master_dark.fits -cc=dark -cfa -debayer"] if use_dark else
                ["convert light -debayer -out=../process", "cd ../process"])
        n = sum(1 for f in (work / "lights").iterdir() if FRAME.search(f.name))
        # Big sessions: the Generalized ESD test (up to 30% outliers per pixel, 5% significance) is what Siril
        # recommends past about 50 frames; it catches satellite and plane trails that Winsorized 3/3 lets through.
        rejection = (f"rej g 0.3 0.05" if n >= 50 and "sigma" not in plan["_from_plan"]
                     else f"rej w {plan['sigma_low']} {plan['sigma_high']}")
        script = work / "overview_stack.ssf"
        script.write_text("\n".join([
            "# Written by DWARF Post-Processing Hub: restack with frame selection",
            "requires 1.2.0", *tune, "setext fits", "set16bits", "cd lights", *prep,
            # The middle frame as reference spreads the field rotation evenly both ways.
            f"setref {seq} {max(1, (n + 1) // 2)}",
            f"register {seq} -2pass",
            f"seqapplyreg {seq} -filter-wfwhm={plan['keep_percent']}% -filter-round=90% -framing={framing}",
            "set32bits",
            f"stack r_{seq} {rejection} -norm=addscale -output_norm"
            # -feather blends each frame's edge in over 40 px, so the turning alt-az field leaves no seams.
            f"{' -weight=wfwhm -feather=40' if v14 else ''} -32b -out=../result_stack",
            ""]), encoding="utf-8")
        (work / "result_stack.fits").unlink(missing_ok=True)
        code, tail = j.run(_siril_args(script, work), work, step=1,
                           phases=["convert", "calibrate", "register", "seqapplyreg", "stack"] if use_dark else
                           ["convert", "register", "seqapplyreg", "stack"])
        if code == 0 and (work / "result_stack.fits").exists():
            return
        if use_dark and not j.stopped():
            j.log("Calibrating with the dark didn't work in Siril; stacking again without it.")
    raise ProcessError("Siril's restack didn't finish:\n" + tail[-1500:])


def _gx_run(j: _Job, exe: str, current: Path, work: Path, t: dict, cmd: list[str], stem: str, step: int,
            batch: bool = False, timeout: float = 3600) -> tuple[Path | None, str]:
    """Run one GraXpert command on the graphics card when this computer has a suitable one, falling back to
    the call an older GraXpert understands."""
    base = [exe, str(current), "-cli", "-cmd", *cmd, "-output", str(work / stem)]
    extra = ["-gpu", "true" if t["graxpert_gpu"] else "false"] + (["-batch_size", str(t["graxpert_batch"])] if batch else [])
    for f in work.glob(stem + ".*"):
        f.unlink(missing_ok=True)
    code, tail = j.run(base + extra, work, timeout, step=step, phases=[cmd[0]])
    if code != 0 and not j.stopped():
        code, tail = j.run(base, work, timeout)
    out = tools._newest_with_stem(work, stem)
    return (Path(out) if code == 0 and out else None), tail


def _graxpert(j: _Job, gx: dict, exe: str, current: Path, work: Path, t: dict, rec: dict | None = None) -> Path:
    j.step(3, "running")
    corr = gx.get("correction") or "Subtraction"
    # Galaxies like M 31 fill much of the DWARF's 2.9°×1.6° field, and a rough background model eats their faint
    # outer halo; a smoother model leaves it. Nebulae and clusters take the usual light smoothing.
    kind = (rec or {}).get("kind") or ""
    default = 0.6 if "galaxy" in kind else 0.5 if ("reflection" in kind or "cluster" in kind) else 0.2
    smooth = float(gx.get("smoothing") if gx.get("smoothing") is not None else default)
    out, tail = _gx_run(j, exe, current, work, t, ["background-extraction", "-correction", corr,
                                                   "-smoothing", f"{smooth:.2f}"], "result_bg", 3)
    if not out:
        raise ProcessError("GraXpert's background extraction failed:\n" + tail[-1200:])
    j.outputs["graxpert"] = str(out)
    j.step(3, "done", f"{corr}, smoothing {smooth:.2f}{', on the graphics card' if t['graxpert_gpu'] else ''}.")
    return out


def _siril_try(j: _Job, work: Path, tune: list[str], body: list[str], out_name: str, name: str,
               timeout: float = 1800) -> tuple[bool, str]:
    script = work / f"overview_{name}.ssf"
    script.write_text("\n".join(["requires 1.2.0", *tune, *body, f"save {out_name}", ""]), encoding="utf-8")
    (work / f"{out_name}.fits").unlink(missing_ok=True)
    code, tail = j.run(_siril_args(script, work), work, timeout)
    return code == 0 and (work / f"{out_name}.fits").exists(), tail


def _has_wcs(path: Path) -> bool:
    try:
        from astropy.io import fits
        h = fits.getheader(path)
        return "CRVAL1" in h and ("CD1_1" in h or "CDELT1" in h or "PC1_1" in h)
    except OSError:
        return False


def _astap_solve(j: _Job, src: Path, dst: Path, plan: dict, hint: dict) -> bool:
    from .platesolve import find_astap
    exe = find_astap()
    if not exe:
        return False
    shutil.copyfile(src, dst)
    try:
        from astropy.io import fits
        height = fits.getheader(dst).get("NAXIS2") or 2000
    except OSError:
        height = 2000
    fov = height * 206.265 * plan["pixel"] / plan["focal"] / 3600
    args = [exe, "-f", str(dst), "-update", "-fov", f"{fov:.3f}", "-z", "0"]
    args += (["-ra", f"{hint['ra'] / 15:.5f}", "-spd", f"{hint['dec'] + 90:.5f}", "-r", "15"] if "ra" in hint else ["-r", "180"])
    j.run(args, dst.parent, 600)
    return _has_wcs(dst)


def _calibrate(j: _Job, plan: dict, shot: dict, current: Path, work: Path, tune: list[str], rec: dict, v14: bool) -> Path:
    j.step(4, "running")
    from .platesolve import hint_from_shot_info
    hint = hint_from_shot_info(shot)
    coords = f"{hint['ra']:.5f},{hint['dec']:.5f} " if "ra" in hint else ""
    solved = work / "result_solved.fits"
    # Plate solve: Siril first (offline with its local Gaia catalogues on 1.4), then ASTAP.
    ok, tail = _siril_try(j, work, tune, [f'load "{current.name}"',
                                          f"platesolve {coords}-focal={plan['focal']} -pixelsize={plan['pixel']}"],
                          "result_solved", "solve")
    how = "Siril"
    if not ok or not _has_wcs(solved):
        how = "ASTAP" if _astap_solve(j, current, solved, plan, hint) else ""
    if not how:
        note = finishing.colour_balance(current, work / "result_calibrated.fits")
        j.outputs["calibrated"] = str(work / "result_calibrated.fits")
        j.step(4, "done", "Couldn't plate solve (Siril and ASTAP), so the colour was calibrated from the image's "
                          f"own stars instead: {note}.")
        return work / "result_calibrated.fits"
    # Colour: SPCC (Siril 1.4; narrowband mode for the Duo-Band filter), then PCC, then from the stars.
    nb = DUO_BAND
    tries = []
    if v14:
        sensor = f'"-oscsensor={DWARF3_SENSOR}"'
        if rec["duo"]:
            narrow = (f"spcc -narrowband -rwl={nb['rwl']} -rbw={nb['rbw']} -gwl={nb['gwl']} -gbw={nb['gbw']} "
                      f"-bwl={nb['bwl']} -bbw={nb['bbw']}")
            tries += [(f"{narrow} {sensor}", "SPCC in narrowband mode (Hα 656 nm, OIII 501 nm)"),
                      (narrow, "SPCC in narrowband mode (Hα 656 nm, OIII 501 nm)")]
        else:
            # Without -oscfilter Siril reuses whatever filter was last picked in its SPCC window.
            full = f'spcc {sensor} "-oscfilter={ASTRO_FILTER}" "-whiteref=Average Spiral Galaxy"'
            tries += [(full, "SPCC with the DWARF 3's sensor and UV/IR-cut filter"),
                      (f"spcc {sensor}", "SPCC with the DWARF 3's sensor"), ("spcc", "SPCC")]
    tries.append(("pcc", "photometric colour calibration (PCC)"))
    for cmd, label in tries:
        ok, tail = _siril_try(j, work, tune, ['load "result_solved"', cmd], "result_calibrated", "colour")
        if ok:
            neutral = finishing.neutralise(work / "result_calibrated.fits", work / "result_calibrated.fits")
            j.outputs["calibrated"] = str(work / "result_calibrated.fits")
            j.step(4, "done", f"Plate solved with {how}; colour by {label}; {neutral}.")
            return work / "result_calibrated.fits"
    note = finishing.colour_balance(solved, work / "result_calibrated.fits")
    j.outputs["calibrated"] = str(work / "result_calibrated.fits")
    j.step(4, "done", f"Plate solved with {how}, but Siril's colour calibration didn't run "
                      f"({(tail.strip().splitlines() or [''])[-1][:160]}), so the colour came from the image's own stars: {note}.")
    return work / "result_calibrated.fits"


def _deconvolve(j: _Job, exe: str | None, current: Path, work: Path, tune: list[str], t: dict, rec: dict) -> Path:
    """GraXpert's AI deconvolution, given the stack's measured star size. Siril's Richardson-Lucy was tried in
    0.8.0: on the DWARF's undersampled data it barely tightened stars (2.53 to 2.47 px) and added noise.
    GraXpert's -psfsize is the FWHM in pixels; left out, it falls back to 0.3 px, which asks the model for almost
    nothing, so it's always measured and passed. Nebulae and galaxies get the object model (sharpens structure,
    leaves stars), clusters the stellar one."""
    j.step(5, "running")
    if exe:
        fwhm = None
        try:
            import numpy as np
            from . import imaging
            data, _ = finishing.read_fits_rgb(current)
            data = np.clip(data / max(1e-9, float(np.percentile(data, 99.99))), 0, 1)
            st = imaging.measure(imaging.LoadedImage(current, data, "fits", "32-bit")).get("stars") or {}
            if st.get("fwhm_px_median") and st["fwhm_px_median"] < 20:
                fwhm = float(st["fwhm_px_median"])
        except Exception:  # measuring is a nicety; GraXpert still runs with a typical DWARF value
            pass
        psf = round(min(8.0, max(1.5, fwhm or 3.6)), 2)
        models = (["deconv-stellar", "deconv-obj"] if "cluster" in rec["kind"] else ["deconv-obj", "deconv-stellar"])
        done, src = [], current
        for i, model in enumerate(models):
            # The object model leaves stars alone and the stellar one sharpens only stars, so both run, one after
            # the other (C 33 in 0.11.1 ran the object model alone and its stars came out soft).
            strength = rec["deconv"] if model == "deconv-obj" else min(rec["deconv"], 0.3)
            out, _ = _gx_run(j, exe, src, work, t, [model, "-strength", f"{strength:.2f}", "-psfsize", f"{psf}"],
                             f"result_deconv{'' if i == 0 else '2'}", 5, batch=True, timeout=1800)
            if out:
                src = out
                done.append(f"{'object (nebula and galaxy detail)' if model == 'deconv-obj' else 'stellar (star cores)'}"
                            f" at {strength:.2f}")
        if done:
            j.outputs["deconvolved"] = str(src)
            j.step(5, "done", f"GraXpert AI deconvolution: {' then '.join(done)}, for stars measured at {psf} px "
                              f"FWHM{'' if fwhm else ' (typical DWARF value; measuring failed)'}.")
            return src
    j.step(5, "skipped", "Skipped: this GraXpert's command line has no deconvolution (GraXpert 4.0 dropped it; 3.1 "
                         "has it), and other deconvolution does more harm than good on the DWARF's data.")
    return current


def _denoise(j: _Job, exe: str | None, current: Path, work: Path, tune: list[str], t: dict, rec: dict) -> Path:
    j.step(6, "running")
    note, out = "", None
    if exe:
        out, _ = _gx_run(j, exe, current, work, t, ["denoising", "-strength", f"{rec['denoise']:.2f}"],
                         "result_dn", 6, batch=True)
        if out:
            note = f"GraXpert AI denoise, strength {rec['denoise']:.2f} (for {rec['minutes']:.0f} min of data)"
    if not out:
        mod = min(1.0, rec["denoise"] + 0.2)
        ok, _ = _siril_try(j, work, tune, [f'load "{current.name}"', f"denoise -mod={mod:.2f}"], "result_dn", "denoise", 3600)
        if ok:
            out, note = work / "result_dn.fits", f"Siril's denoise (NL-Bayes), modulation {mod:.2f}"
    src = out or current
    sigma = 2.5 if rec["duo"] else 1.5
    final = work / "result_dnc.fits"
    finishing.chroma_denoise(src, final, sigma)
    note = (note + "; " if note else "") + (f"colour noise smoothed (σ {sigma})" +
                                             (", stronger for Duo-Band, where only the red pixels see Hα" if rec["duo"] else ""))
    j.outputs["denoised"] = str(final)
    j.step(6, "done", note[:1].upper() + note[1:] + ".")
    return final


STRETCHES = ("autostretch", "asinh", "ght", "autoghs", "modasinh", "mtf", "linstretch")


def _stretch(j: _Job, a: dict, plan: dict, current: Path, work: Path, tune: list[str], rec: dict, calibrated: bool) -> Path:
    """Linear to display. The plan's own Siril commands run first (on linear data); then, unless the plan brings
    its own stretch, a colour-keeping stretch (finishing.colour_stretch) and a saturation boost that leaves the
    sky alone. 0.9.0 used Siril's unlinked auto-stretch here, which Siril's docs say undoes colour calibration."""
    j.step(7, "running")
    handled = {"platesolve", "pcc", "spcc", "crop", "save", "savetif", "rmgreen", "subsky", "denoise", "starnet", "rl"}
    extra = [c for c in a["assessment"].get("siril_script", []) if c.get("command") not in handled
             and tools._ARG_OK.match((c.get("args") or "").strip())]
    notes = []
    pre = []
    if plan["rmgreen"] and not rec["duo"] and not calibrated:
        pre.append("rmgreen 0")  # after SPCC/PCC the green is already right; SCNR would only shift the balance
        notes.append("removed green (average neutral)")
    elif rec["duo"]:
        notes.append("kept the green, since OIII from the Duo-Band filter is green-blue")
    # The plan's own stretch and saturation are left out: on M 45 a plan's linked stretch to a 20% sky replaced
    # the tuned one. Its other linear-data commands still run.
    skipped = [c["command"] for c in extra if c["command"] in STRETCHES or c["command"] == "satu"]
    extra = [c for c in extra if c["command"] not in STRETCHES and c["command"] != "satu"]
    if skipped:
        notes.append(f"left out the plan's {', '.join(sorted(set(skipped)))} in favour of the tuned stretch")
    own_stretch = False
    linear_extra = extra

    def siril(lines: list[str], out_name: str) -> tuple[int, str]:
        fin = work / "overview_stretch.ssf"
        fin.write_text("\n".join(["requires 1.2.0", *tune, "set32bits", f'load "{current.name}"', "save result_linear", *lines,
                                  f"save {out_name}", *([f"savetif {out_name}"] if out_name == "result_stretched" else []),
                                  ""]), encoding="utf-8")
        return j.run(_siril_args(fin, work), work, 3600)

    out = work / "result_stretched.tif"
    for f in (out, work / "result_stretched.fits"):
        f.unlink(missing_ok=True)
    cmds = pre + [f"{c['command']} {(c.get('args') or '').strip()}".rstrip() for c in linear_extra]
    if own_stretch:
        code, tail = siril(cmds, "result_stretched")
        if code == 0 and out.exists():
            notes.append("ran the plan's " + ", ".join(c["command"] for c in linear_extra))
            msg = "; ".join(notes)
            j.outputs["stretched"] = str(out)
            j.step(7, "done", msg[:1].upper() + msg[1:] + ".")
            return out
        j.log("The plan's stretch didn't run in Siril; using the standard stretch instead.")
        notes.append("the plan's own stretch didn't run in Siril, so it used the standard one")
        cmds, linear_extra = pre, []
    source = current
    if cmds:
        code, tail = siril(cmds, "result_prestretch")
        if code == 0 and (work / "result_prestretch.fits").exists():
            source = work / "result_prestretch.fits"
            if linear_extra:
                notes.append("ran the plan's " + ", ".join(c["command"] for c in linear_extra))
        else:
            j.log("A plan command failed in Siril; leaving it out.")
            notes.append("the plan's " + ", ".join(c["command"] for c in linear_extra) + " didn't run in Siril")
    else:
        siril([], "result_linear")  # keep result_linear.fits for the record, as before
    img, header = finishing.read_fits_rgb(source)
    # Duo-Band nebulae are mostly faint gas: a slightly brighter sky shows more of it. Broadband stays darker.
    target = 0.14 if rec["duo"] else 0.12
    img, info = finishing.colour_stretch(img, target_bg=target)
    stars = finishing.star_mask(img)
    img = finishing.cap_star_colour(finishing.saturation(img, 0.35, stars=stars), stars)
    finishing.write_fits_rgb(work / "result_stretched.fits", img, header)
    import tifffile
    import numpy as np
    tifffile.imwrite(str(out), (np.clip(img, 0, 1) * 65535 + 0.5).astype("uint16"), photometric="rgb")
    notes.append(f"colour-keeping stretch to a {int(target * 100)}% sky (brightness stretched, colour ratios kept, "
                 "sky noise kept grey); saturation +35% on the subject, not the sky or the stars; star colour capped "
                 "so stars keep a tint instead of turning into red or cyan discs")
    msg = "; ".join(notes)
    j.outputs["stretched"] = str(out)
    j.step(7, "done", msg[:1].upper() + msg[1:] + ".")
    return out


def _starnet(j: _Job, src: Path, dst: Path) -> bool:
    exe = tools.find_tool("starnet")
    if not exe:
        return False
    dst.unlink(missing_ok=True)
    for args in ([exe, "-i", str(src), "-o", str(dst)], [exe, str(src), str(dst), "256"]):
        j.run(args, Path(exe).parent, 1800)  # StarNet looks for its weights next to itself
        if dst.exists():
            return True
    return False


def _finish(j: _Job, stretched: Path, work: Path, proc: Path, found: dict, rec: dict) -> dict:
    j.step(8, "running")
    img = finishing.read_tif(stretched)
    header = None
    try:
        header = finishing.read_fits_rgb(work / "result_stretched.fits")[1]
    except OSError:
        pass
    notes = []
    if rec["stars"] <= 0:
        notes.append("left the stars alone, as they're the subject")
    elif _starnet(j, stretched, work / "result_starless.tif"):
        img = finishing.reduce_stars(img, rec["stars"], finishing.read_tif(work / "result_starless.tif"))
        notes.append(f"StarNet separated the stars, which were dimmed by {int(rec['stars'] * 60)}% "
                     "(the starless image is kept too)")
    else:  # erosion flattened and fringed the stars in 0.8.0, so without StarNet they're left as they are
        notes.append("stars left as they are: install StarNet++ (Setup links it) for star reduction")
    img, gate = finishing.quality_gate(img)
    notes += gate
    out = {"linear": str(work / "result_linear.fits"), "folder": str(work), "frames": found["count"],
           "cleanup_bytes": _folder_bytes(proc)}
    if rec["duo"] and _hoo_main(rec):
        # Duo-Band on emission nebulae: Hα/OIII is the natural palette for this filter, so HOO is the main result.
        hoo = finishing.hoo(img)
        hoo, _ = finishing.quality_gate(finishing.cap_star_colour(hoo, finishing.star_mask(hoo)))
        saved = finishing.save_final(hoo, work / "result_final", header)
        out["natural"] = finishing.save_final(img, work / "result_natural", header)["tif"]
        notes.append("Duo-Band, so the main result is the HOO palette (Hα red, OIII teal); the natural-colour "
                     "version is saved as result_natural")
    else:
        saved = finishing.save_final(img, work / "result_final", header)
        if rec["duo"]:
            notes.append("shot through the Duo-Band filter, which blocks most of the starlight this target shines by, "
                         "so the natural-colour version is the main result; the Astro filter suits it better next time")
    out.update(final=saved["tif"], files=saved)
    notes.append("saved TIFF (16-bit), PNG, JPG and FITS")
    msg = "; ".join(notes)
    j.outputs["final"] = saved["tif"]
    j.step(8, "done", msg[:1].upper() + msg[1:] + ".")
    return out


def _hoo_main(rec: dict) -> bool:
    """HOO suits gas that glows in Hα and OIII. Reflection nebulae (M 45), galaxies and clusters shine by starlight,
    so their natural-colour version stays the main result even when shot through the Duo-Band filter."""
    return not any(w in (rec.get("kind") or "") for w in ("reflection", "galaxy", "cluster"))


def _fill_empty(src: Path, dst: Path) -> str:
    """Full-frame stacks: corners the turning field never covered come out as zeros. Fill them with the
    background level so background extraction and the stretch aren't thrown by black wedges."""
    import numpy as np
    from astropy.io import fits
    with fits.open(src) as hdul:
        data, header = np.asarray(hdul[0].data, dtype="float32"), hdul[0].header.copy()
    empty = np.all(data == 0, axis=0) if data.ndim == 3 else data == 0
    share = float(empty.mean())
    if share:
        planes = data if data.ndim == 3 else data[None]
        for c in planes:
            c[empty] = np.median(c[~empty]) if (~empty).any() else 0
    for k in ("BZERO", "BSCALE"):
        header.remove(k, ignore_missing=True)
    fits.PrimaryHDU(data=data, header=header).writeto(dst, overwrite=True)
    h, w = data.shape[-2:]
    return (f"Kept the full {w}×{h} frame, no crop" +
            (f"; {share * 100:.1f}% of it (corners the turning sky never covered) was filled with the background level."
             if share > 0.0005 else "."))


def _crop(src: Path, dst: Path, margin: int) -> None:
    from astropy.io import fits
    with fits.open(src) as hdul:
        hdu = hdul[0]
        data, header = hdu.data, hdu.header.copy()
        h, w = data.shape[-2:]
        m = max(0, min(margin, h // 4, w // 4))
        data = data[..., m:h - m, m:w - m] if m else data
        for key in ("CRPIX1", "CRPIX2"):
            if key in header:
                header[key] -= m
        fits.PrimaryHDU(data=data, header=header).writeto(dst, overwrite=True)


def _folder_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def cleanup(folder: str) -> dict:
    """Delete Siril's working copies (the process folder) after a run. The frames and results stay."""
    work = Path(folder).resolve()
    root = work_root().resolve()
    if root not in work.parents:
        raise ProcessError("That isn't a processing folder.")
    proc = work / "process"
    freed = _folder_bytes(proc)
    shutil.rmtree(proc, ignore_errors=True)
    return {"freed_bytes": freed}


# ---------------------------------------------------------------- LAST SESSION

def unfinished() -> list[dict]:
    """Auto-apply runs that didn't finish (stopped, failed, or cut off by closing the app or an update),
    newest first, plus any run still going."""
    roots = [r for r in (work_root(), finished_root()) if r and r.is_dir()]
    out = []
    for f in (f for root in roots for f in root.glob(f"*/{RUN_FILE}")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        live = tools.JOBS.get(r.get("job_id") or "")
        if live and live["state"] == "running":
            r["status"] = "running"
        elif r.get("status") == "running":
            r["status"] = "interrupted"  # the app was closed, updated or the computer restarted mid-run
        if r.get("status") == "done":
            continue
        nxt = next((s["label"] for s in r.get("steps", []) if s["state"] not in ("done", "skipped")), None)
        out.append({"folder": str(f.parent), "session": r.get("session"), "photo": r.get("photo"),
                    "assessment_id": r.get("assessment_id"), "status": r["status"], "progress": r.get("progress", 0),
                    "updated": r.get("updated"), "next_step": nxt, "error": r.get("error"), "job_id": r.get("job_id"),
                    "frames": r.get("frames")})
    return sorted(out, key=lambda r: r.get("updated") or "", reverse=True)


def resume(folder: str) -> str:
    """Carry on a run from its first unfinished step. A run that's still going is just reattached."""
    work = Path(folder).resolve()
    if work_root().resolve() not in work.parents:
        raise ProcessError("That isn't a processing folder.")
    try:
        r = json.loads((work / RUN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ProcessError("That run's record (run.json) couldn't be read.")
    live = tools.JOBS.get(r.get("job_id") or "")
    if live and live["state"] == "running":
        return r["job_id"]
    return start(int(r["assessment_id"]), bool(r.get("include_failed")), resume=r, full_frame=r.get("full_frame", True))
