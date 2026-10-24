"""First-time setup and the pre-start software check.

Lists everything DWARF Post-Processing Hub uses, whether it's on this computer, and where
to download it (official sites only). Nothing is installed silently; the page
opens the download pages and checks again afterwards. Dwarfium's browser
edition is the exception: the app installs that itself, with a checksum check.

Run `python -m dwarf_overview.setupcheck` for the same check in a console;
the Windows launcher runs it before starting the app.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from . import config

PACKAGES = [("anthropic", "anthropic"), ("numpy", "numpy"), ("PIL", "pillow"), ("astropy", "astropy"),
            ("tifffile", "tifffile"), ("cv2", "opencv-python-headless")]


def _packages() -> list[str]:
    missing = []
    for mod, pip_name in PACKAGES:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pip_name)
    return missing


def _astap_db(astap: str | None) -> bool:
    if not astap:
        return False
    folder = Path(astap).parent
    return any(folder.glob("d50*")) or any(folder.glob("d80*")) or any(folder.glob("v50*")) or any(folder.glob("h18*"))


def check() -> dict:
    from . import dwarfium_web, tools
    t = tools.status()
    missing_pkgs = _packages()
    inst = dwarfium_web.installed()
    items = [
        {"key": "python", "name": "Python 3.10 or newer", "level": "required", "found": sys.version_info >= (3, 10),
         "detail": sys.version.split()[0], "url": "https://www.python.org/downloads/",
         "why": "Runs this app. Tick \"Add python.exe to PATH\" when installing."},
        {"key": "packages", "name": "Python libraries", "level": "required", "found": not missing_pkgs,
         "detail": "all installed" if not missing_pkgs else "missing: " + ", ".join(missing_pkgs),
         "why": "The launcher installs these by itself; if any are missing, close the app and run the launcher again."},
        {"key": "api_key", "name": "Anthropic API key", "level": "required", "found": bool(config.api_key()),
         "url": "https://console.anthropic.com/settings/keys", "action": "settings",
         "why": "Lets Claude assess your photos. Paste it in Settings; it stays on this computer."},
        {"key": "siril", "name": "Siril 1.2 or newer", "level": "recommended", "found": bool(t.get("siril") and t.get("siril_cli")),
         "detail": t.get("siril") or "", "url": "https://siril.org/download/",
         "why": "Stacking and processing; the app runs its suggested scripts."},
        {"key": "graxpert", "name": "GraXpert", "level": "recommended", "found": bool(t.get("graxpert")),
         "detail": t.get("graxpert") or "", "url": "https://github.com/Steffenhir/GraXpert/releases/latest",
         "why": "Gradient removal and denoising, run automatically with the suggested settings."},
        {"key": "astap", "name": "ASTAP plate solver + D50 star database", "level": "recommended",
         "found": bool(t.get("astap")) and _astap_db(t.get("astap")),
         "detail": (t.get("astap") or "") + ("" if _astap_db(t.get("astap")) or not t.get("astap") else " (star database missing)"),
         "url": "https://www.hnsky.org/astap.htm", "why": "Names the subject of each photo from its stars. Install ASTAP, then the D50 database from the same page."},
        {"key": "starnet", "name": "StarNet++ v2 (command-line version)", "level": "recommended", "found": bool(t.get("starnet")),
         "detail": t.get("starnet") or "", "url": "https://www.starnetastro.com/download/",
         "why": "Auto-apply uses it to separate the stars for a clean star reduction. Unzip it into Program Files\\StarNet (or set its location in Settings); without it a simpler star reduction is used."},
        {"key": "dwarfium", "name": "Dwarfium (browser edition)", "level": "recommended", "found": bool(inst),
         "detail": inst["version"] if inst else "", "action": "install_dwarfium",
         "why": "Full telescope control in the Telescope tab. The app downloads and verifies it for you."},
        {"key": "pipp", "name": "PIPP", "level": "optional", "found": bool(t.get("pipp")), "detail": t.get("pipp") or "",
         "url": "https://sites.google.com/site/astropipp/downloads", "why": "Moon: crops and sorts video frames."},
        {"key": "autostakkert", "name": "AutoStakkert!", "level": "optional", "found": bool(t.get("autostakkert")),
         "detail": t.get("autostakkert") or "", "url": "https://www.autostakkert.com/wp/download/",
         "why": "Moon: stacks the best frames of a video."},
        {"key": "registax", "name": "RegiStax", "level": "optional", "found": bool(t.get("registax")),
         "detail": t.get("registax") or "", "url": "https://www.astronomie.be/registax/download.html",
         "why": "Moon: wavelet sharpening."},
        {"key": "imppg", "name": "ImPPG", "level": "optional", "found": bool(t.get("imppg")), "detail": t.get("imppg") or "",
         "url": "https://greatattractor.github.io/imppg/", "why": "Moon: deconvolution sharpening."},
    ]
    required_ok = all(i["found"] for i in items if i["level"] == "required")
    return {"items": items, "required_ok": required_ok, "setup_done": bool(config.load().get("setup_done")),
            "windows": sys.platform == "win32"}


def main() -> int:
    """Console check for the launcher: exit code 1 only if something required to start is missing."""
    missing = _packages()
    if sys.version_info < (3, 10):
        print(f"DWARF Post-Processing Hub needs Python 3.10 or newer; this is {sys.version.split()[0]}.")
        print("Get it from https://www.python.org/downloads/ and run the launcher again.")
        return 1
    if missing:
        print("These Python libraries didn't install: " + ", ".join(missing))
        print("Check the internet connection and run the launcher again.")
        return 1
    try:
        report = check()
    except Exception as e:  # the check itself must never stop the app from starting
        print(f"Software check skipped ({e}).")
        return 0
    todo = [i for i in report["items"] if not i["found"] and i["level"] != "optional"]
    if todo:
        print("Software check: the app will start, but these are missing (the Setup page links each one):")
        for i in todo:
            print(f"  - {i['name']} ({i['level']})")
    else:
        print("Software check: everything needed is installed.")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
