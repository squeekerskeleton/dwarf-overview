"""Build a release: DWARF-Overview-<version>.zip plus latest.json for the built-in updater.

Usage: python scripts/make_release.py <owner/repo> [notes]
Then commit releases/<zip> and latest.json to the repo's main branch; the app picks the update up from there.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dwarf_overview import __version__  # noqa: E402

SKIP = {".venv", ".git", "releases", "__pycache__"}


def main():
    repo = sys.argv[1]
    notes = sys.argv[2] if len(sys.argv) > 2 else ""
    dist = ROOT / "releases"
    dist.mkdir(exist_ok=True)
    zip_path = dist / f"DWARF-Overview-{__version__}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT)
            if p.is_file() and not SKIP & set(rel.parts) and rel.name not in ("latest.json", ".gitignore"):
                z.write(p, "DWARF Overview/" + rel.as_posix())
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    manifest = {"version": __version__, "notes": notes, "sha256": digest,
                "zip_url": f"https://raw.githubusercontent.com/{repo}/main/releases/{zip_path.name}"}
    (ROOT / "latest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(zip_path, digest)


if __name__ == "__main__":
    main()
