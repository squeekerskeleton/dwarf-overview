"""Filters screwed or held in front of the DWARF 3's lens.

The DWARF 3 writes only its own filter wheel position (VIS / Astro / Duo-Band) into its files, so a filter held in
front of the lens is invisible in the data. The owner says which one was used for a session (Auto-apply, "Filter in
front of the lens"), and the choice is kept per session in external_filters.json in the app's data folder.

Each filter lists what the camera's red and green/blue pixels see through it, which is what Siril's SPCC narrowband
mode needs (-rwl/-rbw for red, -gwl/-gbw and -bwl/-bbw for green and blue, in nm). Behind an external filter the
DWARF's own filter must be Astro (UV/IR cut): the Duo-Band would block the external filter's other line.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config

# The DWARF 3's own Duo-Band (also in autoprocess.DUO_BAND): Hα about 18 nm, OIII about 30 nm.
DUO_BAND = {"key": "duo", "name": "DWARF 3 Duo-Band (Hα + OIII)", "short": "Duo-Band", "red": "Hα", "palette": "HOO",
            "rwl": 656.28, "rbw": 18, "gwl": 500.7, "gbw": 30, "bwl": 500.7, "bbw": 30}

EXTERNAL = {
    # SVBONY SV220: SII 671.6 nm and OIII 500.7 nm, 7 nm each. Red pixels see SII, green and blue see OIII.
    "sv220": {"key": "sv220", "name": "SVBONY SV220 (SII + OIII, 7 nm)", "short": "SV220 SII+OIII", "red": "SII",
              "palette": "SOO", "rwl": 671.6, "rbw": 7, "gwl": 500.7, "gbw": 7, "bwl": 500.7, "bbw": 7},
}


def _path() -> Path:
    return config.data_dir() / "external_filters.json"


def _load() -> dict:
    try:
        d = json.loads(_path().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def for_session(session: str) -> str:
    """The external filter key set for this session, or "" for none."""
    key = _load().get(session or "", "")
    return key if key in EXTERNAL else ""


def set_for_session(session: str, key: str) -> str:
    if not session:
        raise ValueError("No session to set the filter for.")
    key = (key or "").strip().lower()
    if key and key not in EXTERNAL:
        raise ValueError(f"Unknown filter {key!r}; known: {', '.join(EXTERNAL)}.")
    d = _load()
    if key:
        d[session] = key
    else:
        d.pop(session, None)
    p = _path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return key


def choices() -> list[dict]:
    return [{"key": "", "name": "None (only the DWARF's own filter)"}] + [
        {"key": k, "name": f["name"]} for k, f in EXTERNAL.items()]


def passbands(external: str, duo: bool) -> dict | None:
    """The narrowband passbands for SPCC: the external filter's, else the Duo-Band's, else None (broadband)."""
    if external in EXTERNAL:
        return EXTERNAL[external]
    return DUO_BAND if duo else None
