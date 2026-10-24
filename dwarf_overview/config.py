"""Settings and data folder. Everything stays on this computer."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULTS = {
    "model": "claude-opus-5-5",
    "effort": "medium",
    "siril_path": "",
    "graxpert_path": "",
    "pipp_path": "",
    "autostakkert_path": "",
    "registax_path": "",
    "imppg_path": "",
    "other_app_name": "",
    "other_app_path": "",
    "start_folder": "",
    "dwarfium_path": "",
    "dwarf_hardware_id": 2,
    "site_lat": "",
    "site_lon": "",
    "site_name": "",
    "projects_folder": "",
    "setup_done": False,
    "theme_primary": "",
    "theme_secondary": "",
    "mount": "altaz",
    "previous_wifi": "",
    "learn_enabled": True,
    "auto_apply": True,
    "dwarf_ip": "",
    "library_folder": "",
    "work_folder": "",
    "update_url": "",
    "dwarfium_web_url": "",
    "astap_path": "",
    "dwarfium_standalone_dir": "",
}


def data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        d = base / "DwarfOverview"
    elif sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / "DwarfOverview"
    else:
        d = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "dwarf-overview"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path() -> Path:
    return data_dir() / "config.json"


def load() -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(_path().read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return cfg


def save(updates: dict) -> dict:
    cfg = load()
    for key, value in updates.items():
        if key in DEFAULTS or key == "api_key":
            cfg[key] = value
    p = _path()
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return cfg


def api_key() -> str | None:
    """The environment variable wins; otherwise the key saved from the settings panel."""
    return os.environ.get("ANTHROPIC_API_KEY") or load().get("api_key") or None


def public_view() -> dict:
    cfg = load()
    cfg.pop("api_key", None)
    cfg["has_api_key"] = api_key() is not None
    cfg["api_key_source"] = "environment" if os.environ.get("ANTHROPIC_API_KEY") else ("settings" if cfg["has_api_key"] else None)
    cfg["data_dir"] = str(data_dir())
    return cfg
