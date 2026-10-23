"""Diagnose: show Claude what you're seeing in the app, get the likely cause and one-click fixes.

A report is built from the app's own state: version, settings (the API key and passwords are left out),
which programs were found, the connection to the DWARF, the control link, Dwarfium, recent errors from
the app and the page, recent background jobs, and optionally a screenshot of the page. It's saved under
the data folder (diagnostics\\<time>) so it can also be read later, then sent to Claude with your own
API key. Claude can only suggest fixes from a fixed list; each one runs only when you click it.
"""
from __future__ import annotations

import base64
import json
import platform
import shutil
import sys
import threading
from collections import deque
from datetime import datetime

import anthropic

from . import __version__, config, tools

_ERRORS: deque = deque(maxlen=60)
_LOCK = threading.Lock()
SECRET_KEYS = {"api_key"}

FIXES = {
    "find_dwarf": "Search the network for the DWARF and save its address",
    "test_connections": "Test every connection to the DWARF and use the steadiest",
    "connect_control": "Connect the app's control link (takes control from other apps)",
    "disconnect_control": "Disconnect the app's control link (hands control back to Dwarfium or the phone app)",
    "set_dwarf_ip": "Save a specific DWARF address",
    "set_mount": "Switch the mount setting between Alt-Az and EQ",
    "open_tab": "Open one of the app's tabs",
    "open_settings": "Open Settings",
    "open_setup": "Open the Setup page (software check and downloads)",
    "reload_page": "Reload the page",
}
CLIENT_SIDE = {"open_tab", "open_settings", "open_setup", "reload_page"}

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["summary", "likely_causes", "steps", "fixes", "needs_update"],
    "properties": {
        "summary": {"type": "string"},
        "likely_causes": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                          "required": ["cause", "evidence", "confidence"],
                          "properties": {"cause": {"type": "string"}, "evidence": {"type": "string"},
                                         "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}}},
        "steps": {"type": "array", "items": {"type": "string"}},
        "fixes": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                  "required": ["action", "label", "why", "value"],
                  "properties": {"action": {"type": "string", "enum": sorted(FIXES)},
                                 "label": {"type": "string"}, "why": {"type": "string"},
                                 "value": {"type": "string"}}}},
        "needs_update": {"type": "string"},
    },
}

PROMPT = """You are the built-in troubleshooter for DWARF Overview, a Windows desktop helper (Python server plus a page \
in the browser) for a DWARF 3 smart telescope. Tabs: Assess (measure a photo, Claude plans processing in GraXpert/Siril), \
Plan & Control (night planner, goto, camera settings, W/A/S/D joystick, live view over RTSP port 554), Telescope \
(Dwarfium embedded, served at 127.0.0.1:8000 with its DwarfiumProxy helper), From DWARF (sessions over Wi-Fi from the \
DWARF's file server, or USB-C as a drive), History, Projects, Network (live connection status and tests).

DWARF facts: it answers device info on HTTP port 8082 and takes control over a WebSocket on port 9900; only ONE app \
can be in charge at a time (the DWARFLAB phone app, Dwarfium and this app's Connect each take the master lock and can \
knock each other off, which looks like a connect/disconnect loop). Its own hotspot is always 192.168.88.1; on home \
Wi-Fi (STA mode) the router gives it an address. USB-C is files only. Goto needs the site latitude/longitude in Settings.

You get a report of the app's state and, if included, a screenshot of what the user sees. Work out what is wrong \
from the evidence, most likely cause first, and quote the evidence (an error message, a status value). Give short, \
plain steps the user can do. Offer fixes only from the allowed actions, and only ones that address the problem; put \
the needed value in "value" (an IP for set_dwarf_ip, "altaz" or "eq" for set_mount, a tab name for open_tab: work, \
control, scope, dwarf, history, projects, net), otherwise an empty string. If the problem is a bug in the app itself \
that needs a code change, say so in needs_update in one sentence the user can pass on; otherwise leave it empty. \
Never ask for or repeat passwords or API keys. Write for a beginner, briefly."""


def record(source: str, where: str, message: str) -> None:
    """Remember an error for the next diagnosis (the server calls this for every failed request)."""
    with _LOCK:
        _ERRORS.append({"time": datetime.now().isoformat(timespec="seconds"), "source": source,
                        "where": str(where)[:200], "message": str(message)[:1000]})


def _safe_config() -> dict:
    cfg = config.public_view()
    return {k: v for k, v in cfg.items() if k not in SECRET_KEYS}


def collect(client: dict | None = None) -> dict:
    from . import control, dwarfium_web, netcheck, setupcheck
    client = client or {}
    report = {"app_version": __version__, "time": datetime.now().isoformat(timespec="seconds"),
              "system": {"os": platform.platform(), "python": sys.version.split()[0]},
              "settings": _safe_config(),
              "user_says": str(client.get("problem") or "")[:2000],
              "page": {k: client.get(k) for k in ("tab", "visible_messages", "errors", "url", "browser")}}
    for name, fn in (("programs", tools.status), ("network", netcheck.live), ("control_link", control.status),
                     ("dwarfium", dwarfium_web.status)):
        try:
            report[name] = fn()
        except Exception as e:  # the report must always be built
            report[name] = {"error": f"{type(e).__name__}: {e}"}
    try:
        report["software_check"] = [{"name": i["name"], "found": i["found"], "level": i["level"]}
                                    for i in setupcheck.check()["items"]]
    except Exception as e:
        report["software_check"] = {"error": str(e)}
    report["recent_jobs"] = [{"label": (j.get("log") or [""])[0], "state": j.get("state"), "error": j.get("error"),
                              "last_log": (j.get("log") or [])[-3:]} for j in list(tools.JOBS.values())[-8:]]
    with _LOCK:
        report["app_errors"] = list(_ERRORS)[-25:]
    try:
        from . import control as _control
        report["dwarf_command_log"] = _control.log_tail(120)
    except Exception:
        pass
    return report


def save(report: dict, screenshot_b64: str | None) -> str:
    root = config.data_dir() / "diagnostics"
    folder = root / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if screenshot_b64:
        (folder / "screen.jpg").write_bytes(base64.b64decode(screenshot_b64))
    for old in sorted(p for p in root.iterdir() if p.is_dir())[:-10]:  # keep the last 10
        shutil.rmtree(old, ignore_errors=True)
    return str(folder)


def ask(report: dict, screenshot_b64: str | None) -> dict:
    key = config.api_key()
    if not key:
        raise DiagnoseError("Add your Anthropic API key in Settings first (the report was still saved).")
    cfg = config.load()
    content = []
    if screenshot_b64:
        content += [{"type": "text", "text": "Screenshot of what the user sees:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": screenshot_b64}}]
    content.append({"type": "text", "text": "App report:\n" + json.dumps(report, indent=1, default=str)[:60000]})
    try:
        response = anthropic.Anthropic(api_key=key).messages.create(
            model=cfg["model"], max_tokens=6000, system=PROMPT, messages=[{"role": "user", "content": content}],
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}})
    except anthropic.APIConnectionError:
        raise DiagnoseError("Couldn't reach Claude: is the laptop online? (If it's on the DWARF's hotspot without a "
                            "cable, it has no internet.) The report was saved.")
    except anthropic.APIError as e:
        raise DiagnoseError(f"The diagnosis request failed: {getattr(e, 'message', e)}")
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        out = json.loads(text)
    except ValueError:
        raise DiagnoseError("Couldn't read the diagnosis. Try again.")
    out["fixes"] = [f for f in out.get("fixes", []) if f.get("action") in FIXES]
    for f in out["fixes"]:
        f["client_side"] = f["action"] in CLIENT_SIDE
    return out


def run(client: dict, screenshot_b64: str | None, use_claude: bool = True) -> dict:
    report = collect(client)
    folder = save(report, screenshot_b64)
    result = {"saved_to": folder}
    if use_claude:
        result["diagnosis"] = ask(report, screenshot_b64)
    return result


def apply(action: str, value: str = "") -> dict:
    """Run one fix the user clicked. Only the server-side actions in FIXES are allowed."""
    from . import control, netcheck, scope
    if action == "find_dwarf":
        found = scope.scan()
        if not found["devices"]:
            raise DiagnoseError("No DWARF answered on the network.")
        d = found["devices"][0]
        config.save({"dwarf_ip": d["ip"], "dwarf_hardware_id": d["hardware_id"]})
        return {"done": f"Found {d['name']} at {d['ip']} and saved it."}
    if action == "test_connections":
        r = netcheck.diagnose(None)
        return {"done": f"Using {r['best']}." if r["best"] else "The DWARF didn't answer on any connection."}
    if action == "connect_control":
        r = control.connect()
        return {"done": f"Connected to {r.get('name') or r.get('ip')}."}
    if action == "disconnect_control":
        control.disconnect()
        return {"done": "The app's control link is disconnected; other apps can take control now."}
    if action == "set_dwarf_ip":
        if not scope._is_local(value):
            raise DiagnoseError("That isn't a local network address.")
        config.save({"dwarf_ip": value})
        return {"done": f"Saved the DWARF address {value}."}
    if action == "set_mount":
        if value not in ("altaz", "eq"):
            raise DiagnoseError("The mount must be altaz or eq.")
        config.save({"mount": value})
        return {"done": f"Mount set to {'EQ' if value == 'eq' else 'Alt-Az'}."}
    raise DiagnoseError("That fix isn't available.")


class DiagnoseError(Exception):
    pass

