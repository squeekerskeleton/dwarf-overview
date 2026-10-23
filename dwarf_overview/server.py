"""Local web server for the overview page. Listens on this computer only."""
from __future__ import annotations

import json
import os
import secrets
import string
import sys
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__, assess, config, dwarf, dwarfium_web, history, imaging, lunar, platesolve, scope, tools, update, control, planner, projects, setupcheck, liveview, learn, netcheck, diagnose

RESTART_EXIT_CODE = 3  # start-windows.bat relaunches the app when it exits with this code


def _restart_soon() -> None:
    threading.Timer(1.0, lambda: os._exit(RESTART_EXIT_CODE)).start()

STATIC = Path(__file__).parent / "static"
TOKEN = secrets.token_urlsafe(24)
ANALYSES: dict[str, dict] = {}
ANALYSES_LOCK = threading.Lock()


def _capture_start(body: dict) -> dict:
    filt, exp, gain, frames = int(body["filter"]), float(body["exposure"]), int(body["gain"]), int(body["frames"])
    result = control.start_capture(filt, exp, gain, frames)
    cfg = config.load()
    if cfg.get("learn_enabled", True):
        try:
            learn.log_capture(str(body.get("target") or ""), str(body.get("kind") or ""), cfg.get("mount") or "altaz",
                              filt, exp, gain, frames, body.get("conditions") or {})
        except Exception:  # learning must never stop a capture
            pass
    return result


def _drives() -> list[str]:
    if sys.platform == "win32":
        return [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
    roots = ["/Volumes"] if sys.platform == "darwin" else ["/media", "/mnt", "/run/media"]
    return [r for r in roots if os.path.isdir(r)]


def browse(folder: str | None) -> dict:
    start = folder or config.load().get("start_folder") or str(Path.home())
    p = Path(start).expanduser()
    if not p.is_dir():
        p = Path.home()
    dirs, files = [], []
    try:
        for entry in sorted(p.iterdir(), key=lambda e: e.name.lower()):
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    dirs.append(entry.name)
                elif entry.suffix.lower() in imaging.IMAGE_EXTS | lunar.VIDEO_EXTS:
                    files.append({"name": entry.name, "size": entry.stat().st_size})
            except OSError:
                continue
    except OSError as e:
        return {"folder": str(p), "parent": str(p.parent), "dirs": [], "files": [], "error": str(e), "roots": _drives()}
    return {"folder": str(p), "parent": str(p.parent), "dirs": dirs, "files": files, "roots": _drives(),
            "has_shots_info": (p / "shotsInfo.json").is_file()}


class Handler(BaseHTTPRequestHandler):
    server_version = "DwarfOverview"

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    # ------------------------------------------------------------ plumbing
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        if code >= 400 and isinstance(obj, dict) and obj.get("error"):
            diagnose.record("app", urlparse(self.path).path, obj["error"])
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _live(self, camera: str) -> None:
        try:
            frames = liveview.frames(camera)
            first = next(frames)
        except (liveview.LiveError, StopIteration) as e:
            return self._send(503, str(e).encode(), "text/plain")
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            jpg = first
            while True:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                                 + b"\r\n\r\n" + jpg + b"\r\n")
                jpg = next(frames)
        except (OSError, liveview.LiveError, StopIteration):
            pass
        finally:
            frames.close()

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # ------------------------------------------------------------ routes
    def do_GET(self):
        if not self._host_ok():
            return self._send(403, b"Forbidden", "text/plain")
        url = urlparse(self.path)
        if url.path == "/":
            html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", TOKEN)
            return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/api/live.mjpg" and q.get("t") == TOKEN:
            return self._live(q.get("camera", "tele"))
        if self.headers.get("X-Token") != TOKEN:
            return self._send(403, b"Forbidden", "text/plain")
        if url.path == "/api/config":
            return self._json({**config.public_view(), "tools": tools.status(), "version": __version__})
        try:
            if url.path == "/api/update/check":
                return self._json(update.check())
            if url.path == "/api/update/backups":
                return self._json(update.backups())
            if url.path == "/api/best":
                a = history.get_assessment(int(q["assessment_id"]))
                target = a["target"] or a["assessment"].get("identification", {}).get("primary_target")
                return self._json({"target": target,
                                   "best": history.best_of_target(a["target"], a["mode"], a["id"])})
            if url.path == "/api/dwarfium/status":
                return self._json(dwarfium_web.status())
            if url.path == "/api/dwarf/sessions":
                return self._json({"sessions": dwarf.list_sessions(q.get("via", "wifi")),
                                   "library": str(dwarf.library_dir())})
            if url.path == "/api/plan":
                cfg = config.load()
                lat = float(q.get("lat") or cfg.get("site_lat") or "nan")
                lon = float(q.get("lon") or cfg.get("site_lon") or "nan")
                if lat != lat or lon != lon:
                    return self._json({"error": "Add your site's latitude and longitude in Settings first."}, 400)
                return self._json(planner.plan(q["date"], lat, lon, int(q.get("offset", 0)),
                                               mount=q.get("mount") or cfg.get("mount") or "altaz"))
            if url.path == "/api/presets":
                return self._json(learn.presets(q.get("mount") or config.load().get("mount") or "altaz"))
            if url.path == "/api/net/live":
                return self._json(netcheck.live())
            if url.path == "/api/learn/summary":
                return self._json(learn.summary())
            if url.path == "/api/control/status":
                return self._json(control.status())
            if url.path == "/api/control/camera":
                return self._json(control.camera_options(int(q.get("camera", 0))))
            if url.path == "/api/projects":
                return self._json({"projects": projects.list_projects(), "root": str(projects.root())})
            if url.path.startswith("/api/project/"):
                return self._json(projects.get(int(url.path.rsplit("/", 1)[1])))
            if url.path == "/api/setup/check":
                return self._json(setupcheck.check())
            if url.path == "/api/dwarf/usb":
                return self._json({"folders": scope.usb_folders()})
            if url.path == "/api/dwarf/session":
                return self._json(dwarf.session_files(q.get("root", ""), q.get("name", "")))
        except (update.UpdateError, dwarf.DwarfError, dwarfium_web.DwarfiumError, control.ControlError,
                projects.ProjectError, netcheck.NetError, ValueError, KeyError) as e:
            return self._json({"error": str(e)}, 400)
        if url.path == "/api/browse":
            return self._json(browse(q.get("folder")))
        if url.path == "/api/history":
            return self._json(history.list_recent(100))
        if url.path.startswith("/api/assessment/"):
            a = history.get_assessment(int(url.path.rsplit("/", 1)[1]))
            return self._json(a or {"error": "not found"}, 200 if a else 404)
        if url.path.startswith("/api/job/"):
            job = tools.JOBS.get(url.path.rsplit("/", 1)[1])
            return self._json({k: v for k, v in job.items() if k != "proc"} if job else {"error": "not found"},
                              200 if job else 404)
        return self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if not self._host_ok() or self.headers.get("X-Token") != TOKEN:
            return self._send(403, b"Forbidden", "text/plain")
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/config":
                updates = {k: v for k, v in body.items() if k in config.DEFAULTS}
                if body.get("api_key"):
                    updates["api_key"] = body["api_key"].strip()
                if body.get("clear_api_key"):
                    updates["api_key"] = ""
                config.save(updates)
                return self._json({**config.public_view(), "tools": tools.status(), "version": __version__})
            if path == "/api/analyze":
                result = imaging.analyze(body["path"], body.get("mode") or "auto")
                if config.load().get("learn_enabled", True):
                    try:
                        learn.record_measurement(result)
                    except Exception:  # learning must never break measuring
                        pass
                aid = uuid.uuid4().hex[:12]
                with ANALYSES_LOCK:
                    ANALYSES[aid] = result
                    while len(ANALYSES) > 20:
                        ANALYSES.pop(next(iter(ANALYSES)))
                return self._json({"analysis_id": aid, **result})
            if path == "/api/identify":
                aid = body["analysis_id"]
                with ANALYSES_LOCK:
                    analysis = ANALYSES.get(aid)
                if analysis is None:
                    return self._json({"error": "Measure the image again first."}, 400)
                info = analysis["metrics"].get("image", {})

                def run_identify():
                    field = platesolve.identify(analysis["path"], info.get("width", 0), info.get("height", 0),
                                                analysis.get("shot_info") or {},
                                                fits_y_up=info.get("format") != "FITS")
                    analysis["field"] = field
                    if config.load().get("learn_enabled", True):
                        try:
                            learn.record_measurement(analysis)
                        except Exception:
                            pass
                    return field

                return self._json({"job_id": tools.start_task(run_identify, "Plate-solving the star field")})
            if path == "/api/assess":
                with ANALYSES_LOCK:
                    analysis = ANALYSES.get(body["analysis_id"])
                if analysis is None:
                    return self._json({"error": "Analyse the image again first."}, 400)
                note = body.get("note", "")
                result, model = assess.assess(analysis, note)
                result["siril_script_text"] = tools.build_siril_script(analysis["path"], result.get("siril_script", []))
                shot_info = dict(analysis.get("shot_info") or {})
                subject = (analysis.get("field") or {}).get("subject")
                if subject and not history.shot_fields(shot_info)["target"]:
                    shot_info["target"] = subject
                assessment_id = history.add_assessment(analysis["path"], shot_info,
                                                       {**analysis["metrics"], "shot_info": shot_info}, result, model, note,
                                                       analysis.get("mode", "deepsky"), analysis["overview_jpeg_b64"])
                return self._json({"assessment_id": assessment_id, "model": model, **result})
            if path == "/api/compare":
                a = history.get_assessment(int(body["assessment_id"]))
                with ANALYSES_LOCK:
                    analysis = ANALYSES.get(body.get("analysis_id", ""))
                current = analysis["overview_jpeg_b64"] if analysis else None
                if current is None:
                    return self._json({"error": "Measure the image again first."}, 400)
                refs = []
                best = {r["id"]: r for r in history.best_of_target(a["target"], a["mode"], a["id"], limit=50)}
                for hid in body.get("history_ids", []):
                    r = best.get(int(hid))
                    if r and r["preview_b64"]:
                        refs.append((f"Your earlier shot #{r['id']} ({r['created'][:10]})", r["preview_b64"]))
                for pth in body.get("paths", []):
                    refs.append((f"Reference: {Path(pth).name}", imaging.analyze(pth, a["mode"])["overview_jpeg_b64"]))
                if not refs:
                    return self._json({"error": "Pick at least one image to compare with."}, 400)
                target = a["target"] or a["assessment"].get("identification", {}).get("primary_target", "")
                return self._json(assess.compare(current, refs[:3], target))
            if path == "/api/feedback":
                rating = body.get("rating")
                history.add_feedback(int(body["assessment_id"]), int(rating) if rating else None,
                                     body.get("settings_used", ""), body.get("notes", ""))
                return self._json({"ok": True})
            if path == "/api/dwarf/download":
                root, name, files = body["root"], body["name"], list(body.get("files") or [])
                return self._json({"job_id": tools.start_task(lambda: dwarf.download(root, name, files),
                                                              f"Copying {len(files)} file(s) from the DWARF")})
            if path == "/api/dwarfium/install":
                return self._json({"job_id": tools.start_task(dwarfium_web.install,
                                                              "Downloading Dwarfium from its official GitHub release")})
            if path == "/api/dwarfium/start":
                return self._json(dwarfium_web.start())
            if path == "/api/dwarfium/autofill":
                return self._json({"ready": dwarfium_web.prepare_autofill(), "url": dwarfium_web.web_url(),
                                   "page": dwarfium_web.HELPER_NAME})
            if path.startswith("/api/control/"):
                action = path[len("/api/control/"):]
                actions = {
                    "connect": lambda: control.connect(), "disconnect": lambda: control.disconnect(),
                    "goto": lambda: control.goto(float(body["ra_hours"]), float(body["dec"]), str(body.get("name") or "Target")),
                    "stop_goto": lambda: control.stop_goto(),
                    "joystick": lambda: control.joystick(body["angle"], body["length"]),
                    "joystick_stop": lambda: control.joystick_stop(),
                    "capture_start": lambda: _capture_start(body),
                    "capture_stop": lambda: control.stop_capture(),
                }
                if action not in actions:
                    return self._json({"error": "Unknown action."}, 404)
                return self._json(actions[action]())
            if path == "/api/projects/create":
                return self._json(projects.create(body["name"]))
            if path == "/api/project/notes":
                return self._json(projects.save_notes(int(body["id"]), str(body.get("notes", ""))))
            if path == "/api/project/add":
                return self._json(projects.add_photo(int(body["id"]), body["file"], body.get("title", ""),
                                                     body.get("preview_b64")))
            if path == "/api/project/photo":
                return self._json(projects.save_photo(int(body["photo_id"]), body.get("title"), body.get("notes")))
            if path == "/api/project/photo/remove":
                return self._json(projects.remove_photo(int(body["photo_id"])))
            if path == "/api/project/open":
                tools.open_in("folder", str(Path(projects.get(int(body["id"]))["folder"]) / "x"))
                return self._json({"ok": True})
            if path == "/api/setup/done":
                config.save({"setup_done": True})
                return self._json({"ok": True})
            if path == "/api/diagnose":
                shot = body.pop("screenshot_b64", None) or None
                use_claude = bool(body.pop("use_claude", True))
                return self._json({"job_id": tools.start_task(lambda: diagnose.run(body, shot, use_claude),
                                                              "Looking at what's going wrong")})
            if path == "/api/scope/away":
                holder: dict = {}
                note = lambda m: tools.JOBS.get(holder.get("id"), {"log": []})["log"].append(m)
                holder["id"] = tools.start_task(lambda: control.away(note),
                                                "AWAY: stopping everything and tucking the lens away")
                return self._json({"job_id": holder["id"]})
            if path == "/api/scope/away_record":
                if body.get("now"):
                    return self._json({"saved": control.record_away()})
                config.save({"away_pending": True})
                return self._json({"pending": True})
            if path == "/api/scope/reboot":
                return self._json(control.reboot())
            if path == "/api/live/start":
                return self._json(liveview.start(str(body.get("camera") or "tele")))
            if path == "/api/diagnose/fix":
                return self._json(diagnose.apply(str(body["action"]), str(body.get("value") or "")))
            if path == "/api/net/diagnose":
                addrs = [str(a) for a in body.get("addresses") or []]
                return self._json({"job_id": tools.start_task(lambda: netcheck.diagnose(addrs),
                                                              "Testing every connection to the DWARF")})
            if path == "/api/net/join_hotspot":
                ssid, pw = str(body.get("ssid") or ""), str(body.get("password") or "")
                return self._json({"job_id": tools.start_task(lambda: netcheck.join_hotspot(ssid, pw),
                                                              "Joining the DWARF's hotspot")})
            if path == "/api/net/rejoin":
                return self._json(netcheck.rejoin_previous())
            if path == "/api/scope/scan":
                return self._json({"job_id": tools.start_task(scope.scan, "Searching the local network for your DWARF")})
            if path == "/api/scope/stop":
                out = {"jobs_stopped": tools.stop_jobs()}
                cfg = config.load()
                try:
                    out["telescope"] = scope.stop_all((cfg.get("dwarf_ip") or "").strip(),
                                                      int(cfg.get("dwarf_hardware_id") or 2))
                except (scope.ScopeError, OSError) as e:
                    out["telescope_error"] = str(e)
                return self._json(out)
            if path == "/api/update/install":
                result = update.install()
                _restart_soon()
                return self._json(result)
            if path == "/api/update/restore":
                result = update.restore(body["name"])
                _restart_soon()
                return self._json(result)
            if path == "/api/open":
                tools.open_in(body["tool"], body["path"])
                return self._json({"ok": True})
            if path == "/api/run/graxpert":
                a = history.get_assessment(int(body["assessment_id"]))
                gx = dict(a["assessment"]["graxpert"])
                gx.update({k: body[k] for k in ("smoothing", "denoise_strength", "correction", "run_background",
                                                 "run_denoise") if k in body})
                return self._json({"job_id": tools.run_graxpert(body.get("path") or a["file_path"], gx)})
            if path == "/api/run/siril":
                a = history.get_assessment(int(body["assessment_id"]))
                script = body.get("script") or a["assessment"].get("siril_script_text")
                return self._json({"job_id": tools.run_siril_script(body.get("path") or a["file_path"], script)})
        except (assess.AssessmentError, update.UpdateError, dwarf.DwarfError, dwarfium_web.DwarfiumError,
                scope.ScopeError, control.ControlError, projects.ProjectError, netcheck.NetError, diagnose.DiagnoseError, liveview.LiveError, KeyError) as e:
            return self._json({"error": str(e)}, 400)
        except (FileNotFoundError, ValueError, KeyError) as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        return self._send(404, b"Not found", "text/plain")


def main() -> None:
    port = int(os.environ.get("DWARF_OVERVIEW_PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"DWARF Overview {__version__} is running at {url}")
    print(f"Data and history are stored in {config.data_dir()}")
    print("Close this window (or press Ctrl+C) to stop it.")
    control.away_watcher()
    if not os.environ.get("DWARF_OVERVIEW_NO_BROWSER"):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
