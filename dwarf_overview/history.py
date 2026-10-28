"""SQLite log of every assessment and how it turned out.

Each assessment stores the measurements, the shot details and Claude's advice.
Feedback (a rating, what you actually used, notes) is attached afterwards.
Recent rows are summarised and sent with the next request, so the advice
learns which settings worked for your skies and your scope.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT NOT NULL,
    file_path TEXT NOT NULL,
    target TEXT,
    filter TEXT,
    exposure TEXT,
    gain TEXT,
    frames TEXT,
    session_note TEXT,
    metrics_json TEXT NOT NULL,
    assessment_json TEXT NOT NULL,
    model TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT NOT NULL,
    mode TEXT NOT NULL,
    target TEXT,
    target_type TEXT,
    folder TEXT,
    version TEXT,
    metrics_json TEXT NOT NULL,
    rating INTEGER,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id INTEGER NOT NULL REFERENCES assessments(id),
    created TEXT NOT NULL,
    rating INTEGER,
    settings_used TEXT,
    notes TEXT
);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.data_dir() / "history.db")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(assessments)")}
    if "mode" not in cols:
        conn.execute("ALTER TABLE assessments ADD COLUMN mode TEXT NOT NULL DEFAULT 'deepsky'")
    if "preview_b64" not in cols:
        conn.execute("ALTER TABLE assessments ADD COLUMN preview_b64 TEXT")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def shot_fields(shot_info: dict) -> dict:
    """Pull the common fields out of shotsInfo.json, tolerating key-name differences between firmware versions."""
    def pick(*names):
        for n in names:
            for k, v in shot_info.items():
                if k.lower() == n.lower() and v not in (None, ""):
                    return str(v)
        return None

    return {
        "target": pick("target", "targetName", "object"),
        "filter": pick("ir", "filter", "irCut"),
        "exposure": pick("exp", "exposure", "expTime"),
        "gain": pick("gain"),
        "frames": pick("shotsStacked", "stackedCount", "shotsTaken", "frames"),
    }


def add_assessment(file_path: str, shot_info: dict, metrics: dict, assessment: dict, model: str,
                   session_note: str = "", mode: str = "deepsky", preview_b64: str | None = None) -> int:
    f = shot_fields(shot_info)
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO assessments (created, file_path, target, filter, exposure, gain, frames, session_note,"
            " metrics_json, assessment_json, model, mode, preview_b64) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), file_path, f["target"], f["filter"], f["exposure"], f["gain"], f["frames"], session_note,
             json.dumps(metrics), json.dumps(assessment), model, mode, preview_b64))
        return int(cur.lastrowid)


def best_of_target(target: str | None, mode: str, exclude_id: int | None = None, limit: int = 3) -> list[dict]:
    """Your own best earlier shots of a target: highest rating first, then Claude's score."""
    rows = [r for r in list_recent(500) if (r["mode"] or "deepsky") == mode and r["id"] != exclude_id
            and (not target or (r["target"] or "").lower() == target.lower())]
    rows.sort(key=lambda r: (r["rating"] or 0, r["quality_score"] or 0), reverse=True)
    out = []
    with _conn() as c:
        for r in rows[:limit]:
            prev = c.execute("SELECT preview_b64 FROM assessments WHERE id = ?", (r["id"],)).fetchone()[0]
            out.append({**r, "preview_b64": prev})
    return out


def get_assessment(assessment_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM assessments WHERE id = ?", (assessment_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["metrics"] = json.loads(d.pop("metrics_json"))
    d["assessment"] = json.loads(d.pop("assessment_json"))
    d.pop("preview_b64", None)
    return d


def add_feedback(assessment_id: int, rating: int | None, settings_used: str, notes: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO feedback (assessment_id, created, rating, settings_used, notes) VALUES (?,?,?,?,?)",
                  (assessment_id, _now(), rating, settings_used, notes))


def list_recent(limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT a.id, a.created, a.mode, a.file_path, a.target, a.filter, a.exposure, a.gain, a.frames, a.metrics_json,"
            " a.assessment_json, f.rating, f.settings_used, f.notes FROM assessments a"
            " LEFT JOIN feedback f ON f.id = (SELECT MAX(id) FROM feedback WHERE assessment_id = a.id)"
            " ORDER BY a.id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        m = json.loads(r["metrics_json"])
        a = json.loads(r["assessment_json"])
        out.append({
            "id": r["id"], "created": r["created"], "mode": r["mode"], "file": r["file_path"], "target": r["target"],
            "filter": r["filter"], "exposure": r["exposure"], "gain": r["gain"], "frames": r["frames"],
            "quality_score": a.get("quality_score"), "summary": a.get("overall_summary", ""),
            "fwhm_px": (m.get("stars") or {}).get("fwhm_px_median"),
            "elongation": (m.get("stars") or {}).get("elongation_median"),
            "gradient_pct": (m.get("gradient") or {}).get("spread_percent_of_background"),
            "snr": (m.get("background") or {}).get("snr_proxy_p99_5"),
            "cast": (m.get("color") or {}).get("background_cast"),
            "sharpness": ((m.get("detail") or {}).get("sharpness_index")
                          or (m.get("frame_quality") or {}).get("sharpness_p90")),
            "lunar_kind": "video" if "video" in m else ("stack" if "disk" in m else None),
            "good_frames_pct": (m.get("frame_quality") or {}).get("frames_within_15pct_of_best_percent"),
            "limb_clipped_pct": (m.get("exposure") or {}).get("saturated_disk_percent"),
            "halo_sigma": m.get("sharpening_halo_sigma"),
            "plan": "; ".join(f"{s.get('tool')}: {s.get('action')} ("
                              + ", ".join(f"{x.get('name')}={x.get('value')}" for x in s.get("settings", [])[:4]) + ")"
                              for s in a.get("steps", [])[:5]),
            "graxpert": a.get("graxpert"), "learned_note": a.get("learned_note", ""),
            "rating": r["rating"], "settings_used": r["settings_used"], "notes": r["notes"],
        })
    return out


def context_for(shot_info: dict, mode: str = "deepsky", limit: int = 12) -> str:
    """Compact text history for the prompt: same mode only, same target and filter first, then the most recent."""
    rows = [r for r in list_recent(200) if (r["mode"] or "deepsky") == mode]
    if not rows:
        return f"No previous {'lunar' if mode == 'moon' else 'deep-sky'} sessions logged yet."
    f = shot_fields(shot_info)

    def score(r):
        return (2 if f["target"] and r["target"] == f["target"] else 0) + (1 if f["filter"] and r["filter"] == f["filter"] else 0)

    chosen = sorted(rows, key=lambda r: (score(r), r["id"]), reverse=True)[:limit]
    lines = [f"{len(rows)} previous assessments logged. Most relevant {len(chosen)}:"]
    for r in chosen:
        gx = r["graxpert"] or {}
        if mode == "moon":
            line = (f"- #{r['id']} {r['created'][:10]} lunar {r['lunar_kind']} exp={r['exposure'] or '?'} "
                    f"gain={r['gain'] or '?'} | sharpness={r['sharpness']} good_frames={r['good_frames_pct']}% "
                    f"limb_clipped={r['limb_clipped_pct']}% halo={r['halo_sigma']} | score={r['quality_score']} | "
                    f"advised: {r['plan']}")
            if r["rating"] is not None or r["settings_used"] or r["notes"]:
                line += (f" | USER RESULT rating={r['rating']}/5 used=\"{r['settings_used'] or ''}\" "
                         f"notes=\"{r['notes'] or ''}\"")
            if r["learned_note"]:
                line += f" | note: {r['learned_note']}"
            lines.append(line)
            continue
        line = (f"- #{r['id']} {r['created'][:10]} {r['target'] or '?'} filter={r['filter'] or '?'} "
                f"exp={r['exposure'] or '?'}s gain={r['gain'] or '?'} frames={r['frames'] or '?'} | "
                f"FWHM={r['fwhm_px']}px elong={r['elongation']} gradient={r['gradient_pct']}% snr={r['snr']} "
                f"cast={r['cast']} | score={r['quality_score']} | advised GraXpert smoothing={gx.get('smoothing')} "
                f"correction={gx.get('correction')} denoise={gx.get('denoise_strength')}")
        if r["rating"] is not None or r["settings_used"] or r["notes"]:
            line += f" | USER RESULT rating={r['rating']}/5 used=\"{r['settings_used'] or ''}\" notes=\"{r['notes'] or ''}\""
        if r["learned_note"]:
            line += f" | note: {r['learned_note']}"
        lines.append(line)
    rated = [r for r in rows if r["rating"] is not None]
    if rated:
        avg = sum(r["rating"] for r in rated) / len(rated)
        lines.append(f"Average user rating so far: {avg:.1f}/5 over {len(rated)} rated sessions.")
    return "\n".join(lines)


# ---------------------------------------------------------------- Auto-apply and MOON STACK runs
# Every finished run is measured and logged, so the next run of the same kind of target can say whether it beat the
# best so far, and the advice can see which runs the owner liked. The owner's rating outranks the measurements.

KEY = {"deepsky": ("contrast", "nebula-to-sky-noise contrast"), "moon": ("sharpness", "fine-detail sharpness")}
LABELS = {"contrast": "contrast", "sky_noise": "sky noise", "sky": "sky level", "flatness": "background unevenness",
          "sharpness": "sharpness", "noise": "noise", "clipped_pct": "clipped %"}
LOWER_BETTER = {"sky_noise", "flatness", "noise", "clipped_pct"}


def add_run(mode: str, target: str, target_type: str, folder: str, version: str, metrics: dict) -> int:
    with _conn() as c:
        cur = c.execute("INSERT INTO runs (created, mode, target, target_type, folder, version, metrics_json) VALUES (?,?,?,?,?,?,?)",
                        (_now(), mode, target, target_type, folder, version, json.dumps(metrics)))
        return int(cur.lastrowid)


def rate_run(run_id: int, rating: int | None, notes: str = "") -> None:
    with _conn() as c:
        c.execute("UPDATE runs SET rating = ?, notes = ? WHERE id = ?", (rating, notes, run_id))


def runs(mode: str | None = None, limit: int = 200) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM runs" + (" WHERE mode = ?" if mode else "") + " ORDER BY id DESC LIMIT ?",
                         ((mode, limit) if mode else (limit,))).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["metrics"] = json.loads(d.pop("metrics_json"))
        out.append(d)
    return out


def best_run(mode: str, target_type: str, exclude_id: int | None = None) -> dict | None:
    """The best earlier run of this kind of target: the owner's highest rating first, then the key measurement."""
    key = KEY.get(mode, ("contrast",))[0]
    rows = [r for r in runs(mode) if r["id"] != exclude_id and (r["target_type"] or "") == (target_type or "")
            and r["metrics"].get(key) is not None]
    if not rows:
        return None
    return max(rows, key=lambda r: (r["rating"] or 0, r["metrics"][key]))


def compare_note(mode: str, target_type: str, metrics: dict, run_id: int) -> str:
    """One line on how this run measures up against the best earlier run of the same kind of target."""
    best = best_run(mode, target_type, exclude_id=run_id)
    key, key_label = KEY.get(mode, ("contrast", "contrast"))
    if best is None or metrics.get(key) is None:
        return f"first measured {target_type or mode} run, so it's the one to beat"
    a, b = metrics[key], best["metrics"][key]
    verdict = "better than" if a > b * 1.02 else "about the same as" if a >= b * 0.98 else "below"
    rest = [f"{LABELS[k]} {metrics[k]:.4g} against {best['metrics'][k]:.4g}" for k in LABELS
            if k != key and metrics.get(k) is not None and best["metrics"].get(k) is not None][:3]
    who = f"{best['target'] or '?'}, {best['created'][:10]}" + (f", you rated it {best['rating']}/5" if best["rating"] else "")
    return (f"{key_label} {a:.4g}, {verdict} the best earlier {target_type or mode} run ({who}: {b:.4g})"
            + (f"; {', '.join(rest)}" if rest else ""))


def runs_context(mode: str, limit: int = 8) -> str:
    rows = runs(mode, limit)
    if not rows:
        return "No Auto-apply or MOON STACK runs logged yet."
    lines = [f"Recent {'MOON STACK' if mode == 'moon' else 'Auto-apply'} runs (measured by the app; the rating is the owner's):"]
    for r in rows:
        m = ", ".join(f"{k}={v:.4g}" for k, v in r["metrics"].items() if isinstance(v, (int, float)))
        lines.append(f"- {r['created'][:10]} {r['target'] or '?'} ({r['target_type'] or '?'}, app {r['version']}): {m}"
                     + (f" | OWNER RATING {r['rating']}/5" if r["rating"] else "") + (f" notes=\"{r['notes']}\"" if r["notes"] else ""))
    return "\n".join(lines)
