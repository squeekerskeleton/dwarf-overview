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

from . import __version__, assess, config, dwarf, history, imaging, lunar, tools, update

RESTART_EXIT_CODE = 3  # start-windows.bat relaunches the app when it exits with this code


def _restart_soon() -> None:
    threading.Timer(1.0, lambda: os._exit(RESTART_EXIT_CODE)).start()

STATIC = Path(__file__).parent / "static"
TOKEN = secrets.token_urlsafe(24)
ANALYSES: dict[str, dict] = {}
ANALYSES_LOCK = threading.Lock()


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
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

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
        if self.headers.get("X-Token") != TOKEN:
            return self._send(403, b"Forbidden", "text/plain")
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
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
            if url.path == "/api/dwarf/sessions":
                return self._json({"sessions": dwarf.list_sessions(), "library": str(dwarf.library_dir())})
            if url.path == "/api/dwarf/session":
                return self._json(dwarf.session_files(q.get("root", ""), q.get("name", "")))
        except (update.UpdateError, dwarf.DwarfError) as e:
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
            return self._json(job or {"error": "not found"}, 200 if job else 404)
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
                aid = uuid.uuid4().hex[:12]
                with ANALYSES_LOCK:
                    ANALYSES[aid] = result
                    while len(ANALYSES) > 20:
                        ANALYSES.pop(next(iter(ANALYSES)))
                return self._json({"analysis_id": aid, **result})
            if path == "/api/assess":
                with ANALYSES_LOCK:
                    analysis = ANALYSES.get(body["analysis_id"])
                if analysis is None:
                    return self._json({"error": "Analyse the image again first."}, 400)
                note = body.get("note", "")
                result, model = assess.assess(analysis, note)
                result["siril_script_text"] = tools.build_siril_script(analysis["path"], result.get("siril_script", []))
                assessment_id = history.add_assessment(analysis["path"], analysis.get("shot_info") or {},
                                                       analysis["metrics"], result, model, note,
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
        except (assess.AssessmentError, update.UpdateError, dwarf.DwarfError) as e:
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
    if not os.environ.get("DWARF_OVERVIEW_NO_BROWSER"):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
