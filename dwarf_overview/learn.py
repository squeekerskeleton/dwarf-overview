"""Camera-setting recommendations: a published baseline, refined by your own results ("Calibrate & learn").

Baseline (what every recommendation starts from):
  - DWARFLAB's DWARF 3 manual: 15-60 s exposures (EQ mode needed at 30 s and above), gain 60-80
    (minimum 40 for deep-sky telephoto), 200-400 frames.
  - Community guides (dwarfastro.com settings guide, New Forest Astro filter guide): Astro filter for
    galaxies, clusters and reflection nebulae; Duo-band for emission and planetary nebulae and for bright
    Moon or light pollution; Alt-Az subs capped at 15 s; avoid gain 100+ for deep sky.

Learning: with "Calibrate & learn" on, every capture started from the app is logged with its conditions
(target, filter, exposure, gain, Moon, altitude). When a session is measured on the Assess tab, its
results (frames kept vs taken, signal-to-noise, star size and roundness, and your star rating) are
matched back to the settings that produced them. Once there are at least two results for the same kind
of target and filter, the settings that did best for you are recommended, and the reasons are shown.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone

from . import history

SOURCES = {
    "manual": "DWARF 3 manual (DWARFLAB): 15-60 s, gain 60-80, 200-400 frames",
    "dwarfastro": "dwarfastro.com settings guide",
    "nfa": "New Forest Astro filter guide",
}

# kind -> mount -> settings. Filter index: 0 VIS, 1 Astro, 2 Duo-band.
BASE = {
    "emission": {"label": "Emission nebula", "filter": 2, "filter_name": "Duo-band", "gain": 80,
                 "exposure": {"altaz": 15, "eq": 60}, "frames": 300,
                 "why": "Duo-band isolates H-alpha and OIII, so it works even under a bright Moon. Gain 80-90.",
                 "sources": ["dwarfastro", "nfa", "manual"]},
    "planetary": {"label": "Planetary nebula", "filter": 2, "filter_name": "Duo-band", "gain": 80,
                  "exposure": {"altaz": 15, "eq": 30}, "frames": 240,
                  "why": "Small and bright in OIII; Duo-band, centre it in the frame.", "sources": ["nfa", "manual"]},
    "galaxy": {"label": "Galaxy", "filter": 1, "filter_name": "Astro", "gain": 60,
               "exposure": {"altaz": 15, "eq": 60}, "frames": 400,
               "why": "Galaxies shine across the spectrum, so use Astro (Duo-band blocks most of their light). Needs a dark, Moon-free sky.",
               "sources": ["dwarfastro", "nfa", "manual"]},
    "reflection": {"label": "Reflection or dusty nebula", "filter": 1, "filter_name": "Astro", "gain": 60,
                   "exposure": {"altaz": 15, "eq": 60}, "frames": 400,
                   "why": "Reflected starlight is broadband: Astro filter and a dark sky.", "sources": ["nfa", "manual"]},
    "cluster": {"label": "Star cluster", "filter": 1, "filter_name": "Astro", "gain": 50,
                "exposure": {"altaz": 15, "eq": 30}, "frames": 200,
                "why": "Astro filter, gain 50-60; clusters are bright, so fewer frames are needed.",
                "sources": ["dwarfastro", "nfa"]},
    "mixed": {"label": "Mixed deep sky", "filter": 1, "filter_name": "Astro", "gain": 60,
              "exposure": {"altaz": 15, "eq": 30}, "frames": 300,
              "why": "A safe all-round starting point.", "sources": ["dwarfastro", "manual"]},
}
PLANNER_KIND = {"E": "emission", "P": "planetary", "G": "galaxy", "R": "reflection", "C": "cluster"}
FILTERS = {"vis": 0, "astro": 1, "duo-band": 2, "duoband": 2, "duo": 2}

SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (file TEXT PRIMARY KEY, created TEXT NOT NULL, target TEXT, filter INTEGER,
  exposure REAL, gain INTEGER, keep REAL, snr REAL, fwhm REAL, elong REAL, object_type TEXT);
CREATE TABLE IF NOT EXISTS capture_log (id INTEGER PRIMARY KEY, created TEXT NOT NULL, target TEXT, kind TEXT,
  mount TEXT, filter INTEGER, exposure REAL, gain INTEGER, frames INTEGER, conditions TEXT);
"""


def _db():
    con = history._conn()
    con.executescript(SCHEMA)
    return con


def log_capture(target: str, kind: str, mount: str, filter_index: int, exposure: float, gain: int, frames: int,
                conditions: dict | None = None) -> None:
    with _db() as con:
        con.execute("INSERT INTO capture_log (created, target, kind, mount, filter, exposure, gain, frames, conditions) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(timespec="seconds"), target, kind, mount, filter_index,
                     exposure, gain, frames, json.dumps(conditions or {})))


def record_measurement(analysis: dict) -> None:
    """Called whenever a deep-sky photo is measured: free data for Calibrate & learn."""
    if analysis.get("mode") != "deepsky":
        return
    f = history.shot_fields(analysis.get("shot_info") or {})
    shot = analysis.get("shot_info") or {}
    filt = FILTERS.get(str(f["filter"] or "").strip().lower().replace(" ", ""))
    try:
        exp, gain = float(f["exposure"]), int(float(f["gain"]))
    except (TypeError, ValueError):
        return
    if filt is None:
        return
    taken, stacked = shot.get("shotsTaken"), shot.get("shotsStacked")
    m = analysis.get("metrics") or {}
    stars = m.get("stars") or {}
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO measurements (file, created, target, filter, exposure, gain, keep, snr, fwhm, "
                    "elong, object_type) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (analysis.get("path"), datetime.now(timezone.utc).isoformat(timespec="seconds"), f["target"], filt,
                     exp, gain, float(stacked) / float(taken) if taken and stacked else None,
                     (m.get("background") or {}).get("snr_proxy_p99_5"), stars.get("fwhm_px_median"),
                     stars.get("elongation_median"), (analysis.get("field") or {}).get("subject_type")))


def _norm_target(t: str | None) -> str:
    return "".join((t or "").lower().split())


def _results() -> list[dict]:
    """Every measured deep-sky session with its settings and how it turned out."""
    out = []
    with _db() as con:
        rows = con.execute(
            "SELECT a.id, a.target, a.filter, a.exposure, a.gain, a.frames, a.metrics_json, a.assessment_json, "
            "(SELECT rating FROM feedback f WHERE f.assessment_id = a.id ORDER BY f.id DESC LIMIT 1), a.file_path "
            "FROM assessments a WHERE a.mode = 'deepsky'").fetchall()
        logs = con.execute("SELECT target, kind FROM capture_log").fetchall()
        meas = con.execute("SELECT file, target, filter, exposure, gain, keep, snr, fwhm, elong, object_type "
                           "FROM measurements").fetchall()
        rated = {r[0]: r[1] for r in con.execute(
            "SELECT a.file_path, (SELECT rating FROM feedback f WHERE f.assessment_id = a.id ORDER BY f.id DESC LIMIT 1) "
            "FROM assessments a").fetchall()}
    kind_by_target = {_norm_target(t): k for t, k in logs if t}
    seen = set()
    for m in meas:
        kind = kind_by_target.get(_norm_target(m[1])) or _kind_from_text(m[9] or "")
        seen.add(m[0])
        out.append({"target": m[1], "kind": kind, "filter": m[2], "exposure": m[3], "gain": m[4], "keep": m[5],
                    "rating": rated.get(m[0]), "score": _score(rated.get(m[0]), m[6], m[5], m[7], m[8])})
    for r in rows:
        if r[9] in seen:  # already counted from its measurement
            continue
        seen.add(r[9])
        try:
            metrics, assessment = json.loads(r[6] or "{}"), json.loads(r[7] or "{}")
            exp, gain = float(r[3]), int(float(r[4]))
        except (TypeError, ValueError):
            continue
        filt = FILTERS.get(str(r[2] or "").strip().lower().replace(" ", ""))
        if filt is None:
            continue
        shot = metrics.get("shot_info") or {}
        taken, stacked = shot.get("shotsTaken"), shot.get("shotsStacked")
        keep = (float(stacked) / float(taken)) if taken and stacked else None
        stars = metrics.get("stars") or {}
        kind = kind_by_target.get(_norm_target(r[1])) or _kind_from_text(
            (assessment.get("identification") or {}).get("object_type", ""))
        score = _score(r[8], metrics.get("background", {}).get("snr_proxy_p99_5"), keep,
                       stars.get("fwhm_px_median"), stars.get("elongation_median"))
        out.append({"target": r[1], "kind": kind, "filter": filt, "exposure": exp, "gain": gain, "keep": keep,
                    "rating": r[8], "score": score})
    return out


def _kind_from_text(text: str) -> str | None:
    t = (text or "").lower()
    for word, kind in (("planetary", "planetary"), ("supernova", "emission"), ("emission", "emission"),
                       ("h ii", "emission"), ("galaxy", "galaxy"), ("reflection", "reflection"), ("dark", "reflection"),
                       ("cluster", "cluster"), ("nebula", "emission")):
        if word in t:
            return kind
    return None


def _score(rating, snr, keep, fwhm, elong) -> float:
    """0-1. Your rating counts most; otherwise frames kept, signal-to-noise and star shape."""
    parts = []
    if rating:
        parts.append((float(rating) / 5, 3.0))
    if keep is not None:
        parts.append((max(0.0, min(1.0, (keep - 0.6) / 0.4)), 1.5))
    if snr:
        parts.append((max(0.0, min(1.0, float(snr) / 40)), 1.0))
    if fwhm:
        parts.append((max(0.0, min(1.0, (7 - float(fwhm)) / 4)), 0.5))
    if elong:
        parts.append((max(0.0, min(1.0, (1.5 - float(elong)) / 0.5)), 0.5))
    if not parts:
        return 0.5
    return sum(v * w for v, w in parts) / sum(w for _, w in parts)


def recommend(kind: str, mount: str = "altaz", moon_illum: float | None = None) -> dict:
    kind = kind if kind in BASE else "mixed"
    mount = "eq" if mount == "eq" else "altaz"
    b = BASE[kind]
    rec = {"kind": kind, "label": b["label"], "filter": b["filter"], "filter_name": b["filter_name"],
           "exposure": b["exposure"][mount], "gain": b["gain"], "frames": b["frames"],
           "reasons": [b["why"]] + [SOURCES[s] for s in b["sources"]], "learned": False, "samples": 0}
    if mount == "altaz":
        rec["reasons"].append("Alt-Az mode: subs are limited to 15 s on the DWARF 3.")
    if kind in ("galaxy", "reflection") and moon_illum is not None and moon_illum > 0.6:
        rec["reasons"].append("Bright Moon tonight: expect washed-out backgrounds; save these for a darker night if you can.")

    same = [r for r in _results() if r["kind"] == kind and r["filter"] == rec["filter"]
            and (r["exposure"] >= 30 if mount == "eq" else r["exposure"] <= 15)]
    rec["samples"] = len(same)
    if len(same) >= 2:
        combos: dict[tuple, list] = {}
        for r in same:
            combos.setdefault((r["exposure"], r["gain"]), []).append(r)
        best = max(combos.items(), key=lambda kv: (statistics.mean(x["score"] for x in kv[1]), len(kv[1])))
        (exp, gain), runs = best
        mean = statistics.mean(x["score"] for x in runs)
        base_runs = combos.get((rec["exposure"], rec["gain"]))
        if (exp, gain) != (rec["exposure"], rec["gain"]) and (
                not base_runs or mean > statistics.mean(x["score"] for x in base_runs) + 0.05):
            rec["exposure"], rec["gain"], rec["learned"] = exp, gain, True
            rec["reasons"].insert(0, f"Your results: {exp:g} s at gain {gain} did best on {b['label'].lower()} targets "
                                     f"({len(runs)} session{'s' if len(runs) > 1 else ''}, score {mean:.2f}).")
        keeps = [r["keep"] for r in runs if r["keep"] is not None]
        if keeps and statistics.mean(keeps) < 0.8 and rec["exposure"] > 10:
            rec["exposure"] = 10 if rec["exposure"] <= 15 else 30 if rec["exposure"] <= 60 else 60
            rec["learned"] = True
            rec["reasons"].insert(0, f"Your sessions kept only {statistics.mean(keeps):.0%} of frames, "
                                     "so a shorter exposure is suggested to cut trailing and wind rejects.")
    return rec


def presets(mount: str = "altaz", moon_illum: float | None = None) -> dict:
    """One recommended preset per kind of target, in the shape the Plan & Control page uses."""
    out = {}
    for kind in BASE:
        r = recommend(kind, mount, moon_illum)
        out[kind] = {"label": r["label"], "filter": r["filter"], "filter_name": r["filter_name"],
                     "exposure": r["exposure"], "gain": r["gain"], "frames": r["frames"], "learned": r["learned"],
                     "samples": r["samples"], "reasons": r["reasons"], "note": r["reasons"][0]}
    return out


def summary() -> dict:
    results = _results()
    with _db() as con:
        logged = con.execute("SELECT COUNT(*) FROM capture_log").fetchone()[0]
    by_kind = {}
    for r in results:
        if r["kind"]:
            by_kind.setdefault(r["kind"], 0)
            by_kind[r["kind"]] += 1
    return {"captures_logged": logged, "sessions_measured": len(results), "by_kind": by_kind,
            "sources": list(SOURCES.values())}
