"""Auto-apply a processing plan: restack every raw frame of the session behind the photo, then run the plan.

The DWARF's own stack (stacked-*.fits) is what the Assess tab measures. Its raw frames sit in the same session
folder on the telescope (and in the laptop library only if they were copied). Auto-apply:

1. finds every raw frame of that session: the library folder, the work folder, the DWARF over USB-C, or the DWARF
   over Wi-Fi. Frames the DWARF rejected (failed_*) are left out unless asked for;
2. copies the missing ones into the processing folder (Settings), after checking there's room;
3. Siril: debayer, register (2-pass), keep the best frames (FWHM and roundness from the plan), stack with
   Winsorized rejection, 32-bit FITS;
4. crops the stacking edges;
5. GraXpert background extraction (and AI denoise if the plan says so);
6. Siril: plate solve and photometric colour calibration (skipped with a note if there's no internet for the
   star catalogue), remove green only if the plan doesn't say to skip it, then the plan's remaining Siril
   commands, and saves FITS, TIFF and JPG.

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

from . import config, dwarf, history, hwprofile, scope, tools

FRAME = re.compile(r"\.(fits?|fts)$", re.I)
NOT_A_FRAME = re.compile(r"^(stacked|thumbnail|result_|overview_)|_(gx|siril|crop|bg|dn)", re.I)
# Debayered 16-bit RGB is three times a raw frame; Siril keeps the converted and the registered copies.
SPACE_FACTOR = 7.0


class ProcessError(Exception):
    pass


def work_root() -> Path:
    custom = (config.load().get("work_folder") or "").strip()
    return Path(custom).expanduser() if custom else dwarf.library_dir() / "_processing"


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

    for folder, source in ((lights, "processing folder"), (local, "laptop library")):
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
    return {"session": name, "frames": shown, "count": len(shown), "failed": failed, "dwarf_reached": remote,
            "bytes": total, "bytes_to_copy": to_copy, "bytes_needed": int(need), "bytes_free": free,
            "fits": free > need, "work_folder": str(work_root() / _safe(name))}


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
           "rmgreen": None, "focal": 150, "pixel": 2.0}
    for s in a.get("steps", []):
        blob = " ".join([s.get("action", ""), s.get("menu_path", "")] + [f"{x['name']} {x['value']}" for x in s.get("settings", [])])
        low = blob.lower()
        if "fwhm" in low and (m := re.search(r"best\s*(\d{2})\s*%", low)):
            out["keep_percent"] = min(95, max(40, int(m.group(1))))
        if m := re.search(r"roundness\D{0,12}(0\.\d+)", low):
            out["roundness"] = min(0.95, max(0.3, float(m.group(1))))
        if "rejection" in low and (m := re.search(r"sigma\s*([\d.]+)\s*/\s*([\d.]+)", low)):
            out["sigma_low"], out["sigma_high"] = float(m.group(1)), float(m.group(2))
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
WEIGHTS = [25, 55, 1, 9, 4, 6]
STEPS = ["Find and copy the raw frames", "Siril: debayer, register, select frames, stack",
         "Crop the stacking edges", "GraXpert: background extraction", "Siril: plate solve and colour calibration",
         "Siril: finish the plan and save"]


def start(assessment_id: int, include_failed: bool = False, resume: dict | None = None) -> str:
    a = history.get_assessment(int(assessment_id))
    if not a:
        raise ProcessError("Assess the photo first, so there's a plan to apply.")
    found = find_frames(a["file_path"], include_failed)
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
    meta = {"assessment_id": int(assessment_id), "include_failed": include_failed, "session": found["session"],
            "photo": a["file_path"], "frames": found["count"],
            "started": (resume or {}).get("started") or time.strftime("%Y-%m-%d %H:%M:%S")}
    j = _Job(STEPS, Path(found["work_folder"]), meta)
    threading.Thread(target=_run, args=(j, a, found, resume), daemon=True).start()
    return j.id


def _run(j: _Job, a: dict, found: dict, resume: dict | None = None) -> None:
    try:
        out = _pipeline(j, a, found, resume)
        if not j.stopped():
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


def _pipeline(j: _Job, a: dict, found: dict, resume: dict | None = None) -> dict:
    plan = _plan_settings(a["assessment"])
    work = Path(found["work_folder"])
    prev_steps = (resume or {}).get("steps") or []
    prev_out = (resume or {}).get("outputs") or {}

    def reuse(n: int, key: str | None) -> str | bool | None:
        """For LAST SESSION: a step that finished last time and whose file is still there isn't redone."""
        if n >= len(prev_steps) or prev_steps[n].get("state") not in ("done", "skipped"):
            return None
        path = prev_out.get(key) if key else None
        if key and prev_steps[n]["state"] == "done" and not (path and Path(path).exists()):
            return None  # its file is gone: redo the step
        j.step(n, prev_steps[n]["state"], (prev_steps[n].get("note") or "") + " (kept from the last session)")
        if path:
            j.outputs[key] = path
        return path or True

    lights, proc = work / "lights", work / "process"
    try:
        shot = json.loads((_session_of(Path(a["file_path"]))[1] / "shotsInfo.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        shot = {}

    # 1. frames
    j.weights[0] = round(WEIGHTS[0] * found["bytes_to_copy"] / max(1, found["bytes"]))
    t = hwprofile.tuning()
    tune = [f"setcpu {t['siril_threads']}", f"setmem {t['siril_mem_ratio']}"]
    j.log(hwprofile.profile().get("summary", ""))
    j.step(0, "running", f"{found['count']} frames for {found['session']}" +
           (f" ({found['failed']} the DWARF rejected are included)" if found["failed"] else "") + ".")
    copied = _copy_frames(j, found, lights)
    j.step(0, "done", f"{found['count']} frames ready in {lights} ({copied} copied).")

    # 2. restack
    stacked = work / "result_stack.fits"
    if reuse(1, "stacked"):
        pass
    else:
        _restack(j, plan, work, proc, tune)
        j.outputs["stacked"] = str(stacked)
        j.step(1, "done", f"Stacked: {stacked}")

    # 3. crop
    cropped = work / "result_crop.fits"
    if not reuse(2, "cropped"):
        j.step(2, "running")
        _crop(stacked, cropped, plan["crop_px"])
        j.outputs["cropped"] = str(cropped)
        j.step(2, "done", f"Cropped {plan['crop_px']} px off each edge.")

    # 4. GraXpert
    gx = a["assessment"].get("graxpert") or {}
    current = cropped
    exe = tools.find_graxpert()
    kept = reuse(3, "graxpert")
    if kept:
        current = Path(kept) if isinstance(kept, str) else current
    elif not exe:
        j.step(3, "skipped", "GraXpert wasn't found (set it in Settings), so background extraction was skipped.")
    else:
        current = _graxpert(j, gx, exe, current, work, t)

    # 5. plate solve + colour calibration (needs the online star catalogue; skip if it fails)
    kept = reuse(4, "calibrated")
    if kept:
        current = Path(kept) if isinstance(kept, str) else current
    else:
        current = _calibrate(j, plan, shot, current, work, tune)
    return _finish(j, a, plan, current, work, proc, found, tune)


def _restack(j: _Job, plan: dict, work: Path, proc: Path, tune: list[str]) -> None:
    j.step(1, "running", f"Keeping the best {plan['keep_percent']}% by FWHM with roundness above {plan['roundness']}; "
                         f"Winsorized rejection {plan['sigma_low']}/{plan['sigma_high']}.")
    if proc.exists():
        shutil.rmtree(proc, ignore_errors=True)
    script = work / "overview_stack.ssf"
    script.write_text("\n".join([
        "# Written by DWARF Post-Processing Hub: restack with frame selection",
        "requires 1.2.0", *tune, "setext fits", "set16bits",
        "cd lights", "convert light -debayer -out=../process", "cd ../process",
        "register light -2pass",
        f"seqapplyreg light -filter-wfwhm={plan['keep_percent']}% -filter-round={plan['roundness']} -framing=min",
        "set32bits",
        f"stack r_light rej w {plan['sigma_low']} {plan['sigma_high']} -norm=addscale -output_norm -32b -out=../result_stack",
        ""]), encoding="utf-8")
    (work / "result_stack.fits").unlink(missing_ok=True)
    code, tail = j.run(_siril_args(script, work), work, step=1,
                       phases=["convert", "register", "seqapplyreg", "stack"])
    if code != 0 or not (work / "result_stack.fits").exists():
        raise ProcessError("Siril's restack didn't finish:\n" + tail[-1500:])


def _graxpert(j: _Job, gx: dict, exe: str, current: Path, work: Path, t: dict) -> Path:
    j.step(3, "running")
    corr = gx.get("correction") or "Subtraction"
    smooth = float(gx.get("smoothing") if gx.get("smoothing") is not None else 0.2)
    gpu_args = ["-gpu", "true" if t["graxpert_gpu"] else "false"]
    args = [exe, str(current), "-cli", "-cmd", "background-extraction", "-correction", corr,
            "-smoothing", f"{smooth:.2f}", "-output", str(work / "result_bg")]
    code, tail = j.run(args + gpu_args, work, 3600, step=3, phases=["background"])
    if code != 0:  # an older GraXpert without the -gpu option
        code, tail = j.run(args, work, 3600)
    bg = tools._newest_with_stem(work, "result_bg")
    if code != 0 or not bg:
        raise ProcessError("GraXpert's background extraction failed:\n" + tail[-1200:])
    current = Path(bg)
    note = f"{corr}, smoothing {smooth:.2f}."
    if gx.get("run_denoise"):
        args = [exe, str(current), "-cli", "-cmd", "denoising", "-strength",
                f"{float(gx.get('denoise_strength') or 0.5):.2f}", "-output", str(work / "result_dn")]
        code, tail = j.run(args + gpu_args + ["-batch_size", str(t["graxpert_batch"])], work, 3600)
        if code != 0:
            code, tail = j.run(args, work, 3600)
        dn = tools._newest_with_stem(work, "result_dn")
        if code == 0 and dn:
            current, note = Path(dn), note + " AI denoise too."
    j.outputs["graxpert"] = str(current)
    j.step(3, "done", note)
    return current


def _calibrate(j: _Job, plan: dict, shot: dict, current: Path, work: Path, tune: list[str]) -> Path:
    j.step(4, "running")
    from .platesolve import hint_from_shot_info
    hint = hint_from_shot_info(shot)
    coords = f"{hint['ra']:.5f},{hint['dec']:.5f} " if "ra" in hint else ""
    cal = work / "overview_calibrate.ssf"
    cal.write_text("\n".join(["requires 1.2.0", *tune, f'load "{current.name}"',
                              f"platesolve {coords}-focal={plan['focal']} -pixelsize={plan['pixel']}", "pcc",
                              "save result_calibrated", ""]), encoding="utf-8")
    code, tail = j.run(_siril_args(cal, work), work, 1800)
    if code == 0 and (work / "result_calibrated.fits").exists():
        current = work / "result_calibrated.fits"
        j.outputs["calibrated"] = str(current)
        j.step(4, "done", "Plate solved and colour calibrated.")
    else:
        j.step(4, "skipped", "Plate solve or colour calibration didn't work (it needs internet for the star "
                             "catalogue), so colours were left as they are. Siril said: " +
                 (tail.strip().splitlines() or [""])[-1][:200])
    return current


def _finish(j: _Job, a: dict, plan: dict, current: Path, work: Path, proc: Path, found: dict, tune: list[str]) -> dict:

    # 6. the rest of the plan
    j.step(5, "running")
    handled = {"platesolve", "pcc", "spcc", "crop", "save", "savetif", "rmgreen", "subsky"}
    lines = ["requires 1.2.0", *tune, f'load "{current.name}"']
    notes = []
    if plan["rmgreen"]:
        lines.append("rmgreen 0")
        notes.append("removed green (average neutral)")
    else:
        notes.append("skipped Remove Green, as the plan says")
    extra = [c for c in a["assessment"].get("siril_script", []) if c.get("command") not in handled
             and tools._ARG_OK.match((c.get("args") or "").strip())]
    lines += ["save result_linear"]
    for c in extra:
        lines.append(f"{c['command']} {(c.get('args') or '').strip()}".rstrip())
    if not any(c["command"] in ("autostretch", "asinh", "ght") for c in extra):
        lines.append("autostretch")
    lines += ["save result_final", "savetif result_final", "savejpg result_final 95", ""]
    fin = work / "overview_finish.ssf"
    fin.write_text("\n".join(lines), encoding="utf-8")
    code, tail = j.run(_siril_args(fin, work), work, 3600)
    final = work / "result_final.tif"
    if (code != 0 or not final.exists()) and extra:
        # One of the plan's own commands didn't suit this Siril; save with a plain auto-stretch instead.
        j.log("A plan command failed in Siril; saving with a plain auto-stretch instead.")
        keep = [ln for ln in lines if not any(ln.startswith(c["command"] + " ") or ln == c["command"] for c in extra)]
        keep.insert(keep.index("save result_final"), "autostretch") if "autostretch" not in keep else None
        fin.write_text("\n".join(keep), encoding="utf-8")
        notes.append(f"the plan's {', '.join(c['command'] for c in extra)} didn't run in Siril, so it used a plain auto-stretch")
        extra = []
        code, tail = j.run(_siril_args(fin, work), work, 3600)
    if code != 0 or not final.exists():
        raise ProcessError("Siril couldn't finish the plan:\n" + tail[-1200:])
    if extra:
        notes.append("ran " + ", ".join(c["command"] for c in extra))
    msg = "; ".join(notes)
    j.step(5, "done", msg[:1].upper() + msg[1:] + ".")
    j.outputs["final"] = str(final)
    return {"final": str(final), "linear": str(work / "result_linear.fits"), "folder": str(work),
            "frames": found["count"], "cleanup_bytes": _folder_bytes(proc)}


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
    root = work_root()
    out = []
    if not root.is_dir():
        return out
    for f in root.glob(f"*/{RUN_FILE}"):
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
    return start(int(r["assessment_id"]), bool(r.get("include_failed")), resume=r)
