"""Plan a night: which targets are best for a given date and site, and when.

Everything is worked out on this computer: Sun and Moon positions use the
standard low-precision formulas (good to a fraction of a degree, plenty for
planning), and each target's altitude is checked every 10 minutes through the
night. Targets are scored on how long they sit high in a dark sky, how bright
and close the Moon is, and whether the DWARF 3's filters can cut through it
(the Duo-band filter handles moonlight on emission nebulae; galaxies need a
dark sky).

Times are returned in UTC; the page shows them in the laptop's local time.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from . import learn, platesolve

STEP_MIN = 10
MIN_ALT = 25.0

# Camera presets come from learn.py: the published DWARF 3 baseline, refined by Calibrate & learn.
# (catalogue name, display name or None to use the catalogue's, kind, size override arcmin, ra, dec)
# kind: E emission/SNR (Duo-band), P planetary (Duo-band, small), G galaxy, R reflection/dust, C cluster.
TARGETS = [
    ("NGC7000", "North America Nebula (NGC 7000)", "E", None), ("IC5070", "Pelican Nebula (IC 5070)", "E", None),
    ("IC1805", "Heart Nebula (IC 1805)", "E", 150), ("IC1848", "Soul Nebula (IC 1848)", "E", 120),
    ("IC405", "Flaming Star Nebula (IC 405)", "E", None), ("IC410", "Tadpoles Nebula (IC 410)", "E", None),
    ("NGC2237", "Rosette Nebula (NGC 2237)", "E", None), ("NGC2264", "Cone Nebula & Christmas Tree (NGC 2264)", "E", 60),
    ("IC434", "Horsehead & Flame (IC 434)", "E", None), ("NGC1499", "California Nebula (NGC 1499)", "E", None),
    ("NGC6960", "Western Veil (NGC 6960)", "E", 70), ("NGC6992", "Eastern Veil (NGC 6992)", "E", 70),
    ("NGC6888", "Crescent Nebula (NGC 6888)", "E", None), ("IC1396", "Elephant's Trunk (IC 1396)", "E", 170),
    ("NGC7635", "Bubble Nebula (NGC 7635)", "E", None), ("NGC281", "Pacman Nebula (NGC 281)", "E", None),
    ("NGC7380", "Wizard Nebula (NGC 7380)", "E", None), ("IC5146", "Cocoon Nebula (IC 5146)", "R", None),
    ("NGC2174", "Monkey Head Nebula (NGC 2174)", "E", None), ("IC443", "Jellyfish Nebula (IC 443)", "E", 50),
    ("IC1318", "Sadr region (IC 1318)", "E", 180, 305.5, 40.25), ("NGC6914", "NGC 6914", "R", 40, 306.18, 42.48),
    ("NGC6820", "NGC 6820", "E", 30, 295.62, 23.08), ("Sh2-155", "Cave Nebula (Sh2-155)", "E", 50, 344.2, 62.62),
    ("NGC2359", "Thor's Helmet (NGC 2359)", "P", None), ("IC2177", "Seagull Nebula (IC 2177)", "E", 120),
    ("M42", "Orion Nebula (M42)", "E", None), ("M8", "Lagoon Nebula (M8)", "E", None), ("M20", "Trifid Nebula (M20)", "E", None),
    ("M16", "Eagle Nebula (M16)", "E", None), ("M17", "Omega Nebula (M17)", "E", None), ("M1", "Crab Nebula (M1)", "P", None),
    ("M27", "Dumbbell Nebula (M27)", "P", None), ("M57", "Ring Nebula (M57)", "P", None), ("M97", "Owl Nebula (M97)", "P", None),
    ("NGC7293", "Helix Nebula (NGC 7293)", "P", None), ("M76", "Little Dumbbell (M76)", "P", None),
    ("NGC7023", "Iris Nebula (NGC 7023)", "R", 40), ("M45", "Pleiades (M45)", "R", 110), ("M78", "M78", "R", None),
    ("NGC1977", "Running Man Nebula (NGC 1977)", "R", 40), ("IC2118", "Witch Head Nebula (IC 2118)", "R", 180, 75.5, -7.9),
    ("IC348", "IC 348 and dust", "R", 40), ("IC63", "Ghost of Cassiopeia (IC 63)", "E", 20),
    ("M31", "Andromeda Galaxy (M31)", "G", None), ("M33", "Triangulum Galaxy (M33)", "G", None),
    ("M51", "Whirlpool Galaxy (M51)", "G", None), ("M81", "Bode's Galaxy (M81)", "G", 40), ("M82", "Cigar Galaxy (M82)", "G", None),
    ("M101", "Pinwheel Galaxy (M101)", "G", None), ("M106", "M106", "G", None), ("M63", "Sunflower Galaxy (M63)", "G", None),
    ("M64", "Black Eye Galaxy (M64)", "G", None), ("M65", "Leo Triplet (M65, M66, NGC 3628)", "G", 40),
    ("NGC891", "NGC 891", "G", None), ("NGC253", "Sculptor Galaxy (NGC 253)", "G", None), ("NGC2403", "NGC 2403", "G", None),
    ("NGC4565", "Needle Galaxy (NGC 4565)", "G", None), ("NGC4631", "Whale Galaxy (NGC 4631)", "G", None),
    ("M104", "Sombrero Galaxy (M104)", "G", None), ("NGC6946", "Fireworks Galaxy (NGC 6946)", "G", None),
    ("NGC7331", "NGC 7331", "G", None), ("M74", "Phantom Galaxy (M74)", "G", None), ("M94", "M94", "G", None),
    ("M13", "Hercules Cluster (M13)", "C", None), ("M92", "M92", "C", None), ("M3", "M3", "C", None), ("M15", "M15", "C", None),
    ("M2", "M2", "C", None), ("M5", "M5", "C", None), ("M22", "M22", "C", None), ("M11", "Wild Duck Cluster (M11)", "C", None),
    ("NGC869", "Double Cluster (NGC 869/884)", "C", 70), ("M35", "M35", "C", None), ("M37", "M37", "C", None),
    ("M44", "Beehive Cluster (M44)", "C", 95), ("NGC7789", "Caroline's Rose (NGC 7789)", "C", None),
    ("NGC457", "Owl Cluster (NGC 457)", "C", None), ("M52", "M52", "C", None),
]

KIND_INFO = {
    "E": ("Emission nebula", "duo"), "P": ("Planetary nebula / remnant", "small"),
    "G": ("Galaxy", "astro"), "R": ("Reflection / dusty nebula", "astro"), "C": ("Star cluster", "cluster"),
}


def _targets() -> list[dict]:
    out = []
    for entry in TARGETS:
        name, label, kind, size = entry[:4]
        if len(entry) > 4:
            ra, dec, mag = entry[4], entry[5], None
        else:
            o = platesolve.lookup_name(name)
            if not o:
                continue
            ra, dec, mag = o["ra"], o["dec"], o["mag"]
            size = size or o["size_arcmin"]
        out.append({"id": name, "name": label, "kind": kind, "ra": ra, "dec": dec, "size": size or 10, "mag": mag})
    return out


# ---------------------------------------------------------------- Sun, Moon and altitude

def _jd(t: datetime) -> float:
    return t.timestamp() / 86400.0 + 2440587.5


def _lst(jd: float, lon: float) -> float:
    return (280.46061837 + 360.98564736629 * (jd - 2451545.0) + lon) % 360


def _ecl_to_eq(lam: float, beta: float, eps: float) -> tuple[float, float]:
    l, b, e = map(math.radians, (lam, beta, eps))
    ra = math.degrees(math.atan2(math.sin(l) * math.cos(e) - math.tan(b) * math.sin(e), math.cos(l))) % 360
    dec = math.degrees(math.asin(math.sin(b) * math.cos(e) + math.cos(b) * math.sin(e) * math.sin(l)))
    return ra, dec


def sun(jd: float) -> tuple[float, float, float]:
    n = jd - 2451545.0
    L, g = (280.460 + 0.9856474 * n) % 360, math.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)
    ra, dec = _ecl_to_eq(lam, 0.0, 23.439 - 0.0000004 * n)
    return ra, dec, lam


def moon(jd: float) -> tuple[float, float, float, float]:
    """Moon RA, Dec, ecliptic longitude and latitude (low precision, about 0.2 degrees)."""
    T = (jd - 2451545.0) / 36525
    r = lambda d: math.radians(d % 360)
    Lp, Mp, F = 218.316 + 481267.881 * T, r(134.963 + 477198.867 * T), r(93.272 + 483202.019 * T)
    D, M = r(297.850 + 445267.111 * T), r(357.529 + 35999.050 * T)
    lam = (Lp + 6.289 * math.sin(Mp) + 1.274 * math.sin(2 * D - Mp) + 0.658 * math.sin(2 * D)
           + 0.214 * math.sin(2 * Mp) - 0.186 * math.sin(M) - 0.114 * math.sin(2 * F))
    beta = 5.128 * math.sin(F) + 0.281 * math.sin(Mp + F) + 0.278 * math.sin(Mp - F) + 0.173 * math.sin(2 * D - F)
    ra, dec = _ecl_to_eq(lam, beta, 23.439 - 0.0000004 * (jd - 2451545.0))
    return ra, dec, lam % 360, beta


def moon_illumination(jd: float) -> float:
    _, _, sun_lam = sun(jd)
    _, _, lam, beta = moon(jd)
    elong = math.acos(math.cos(math.radians(beta)) * math.cos(math.radians(lam - sun_lam)))
    return (1 - math.cos(elong)) / 2


def alt_az(ra: float, dec: float, lat: float, lst: float) -> tuple[float, float]:
    h, d, p = math.radians(lst - ra), math.radians(dec), math.radians(lat)
    alt = math.asin(math.sin(p) * math.sin(d) + math.cos(p) * math.cos(d) * math.cos(h))
    az = math.atan2(-math.sin(h) * math.cos(d), math.cos(p) * math.sin(d) - math.sin(p) * math.cos(d) * math.cos(h))
    return math.degrees(alt), math.degrees(az) % 360


def separation(ra1, dec1, ra2, dec2) -> float:
    a = (math.sin(math.radians(dec1)) * math.sin(math.radians(dec2))
         + math.cos(math.radians(dec1)) * math.cos(math.radians(dec2)) * math.cos(math.radians(ra1 - ra2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, a))))


def _compass(az: float) -> str:
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((az + 22.5) // 45) % 8]


# ---------------------------------------------------------------- the plan

def plan(date: str, lat: float, lon: float, utc_offset_min: int, limit: int = 25, mount: str = "altaz") -> dict:
    """date is the evening's local date (YYYY-MM-DD); utc_offset_min is local minus UTC, in minutes."""
    local_noon = datetime.strptime(date, "%Y-%m-%d").replace(hour=12, tzinfo=timezone.utc)
    start = local_noon - timedelta(minutes=utc_offset_min)
    samples = []
    for i in range(0, 24 * 60 // STEP_MIN + 1):
        t = start + timedelta(minutes=i * STEP_MIN)
        jd = _jd(t)
        lst = _lst(jd, lon)
        s_ra, s_dec, _ = sun(jd)
        m_ra, m_dec, _, _ = moon(jd)
        m_alt, _ = alt_az(m_ra, m_dec, lat, lst)
        samples.append({"t": t, "jd": jd, "lst": lst, "sun_alt": alt_az(s_ra, s_dec, lat, lst)[0],
                        "moon": (m_ra, m_dec), "moon_alt": m_alt - 0.95 * math.cos(math.radians(m_alt))})

    dark_limit = -18.0 if any(s["sun_alt"] < -18 for s in samples) else -12.0
    dark = [s for s in samples if s["sun_alt"] < dark_limit]
    if not dark:
        return {"date": date, "dark": None, "targets": [], "schedule": [], "presets": learn.presets(mount),
                "note": "The sky doesn't get properly dark on this night at this latitude."}
    mid = dark[len(dark) // 2]
    illum = moon_illumination(mid["jd"])
    moon_up = [s for s in dark if s["moon_alt"] > 0]
    presets = learn.presets(mount, illum)

    results = []
    for tg in _targets():
        kind_label = KIND_INFO[tg["kind"]][0]
        preset_key = learn.PLANNER_KIND[tg["kind"]]
        if tg["kind"] == "P" and tg["size"] >= 10:
            preset_key = "emission"
        narrowband = presets[preset_key]["filter"] == 2
        per_sample = []
        for s in dark:
            alt, az = alt_az(tg["ra"], tg["dec"], lat, s["lst"])
            if alt < MIN_ALT:
                per_sample.append(0.0)
                continue
            q = math.sin(math.radians(alt))
            if s["moon_alt"] > 0:
                sep = separation(tg["ra"], tg["dec"], *s["moon"])
                glare = illum * max(0.0, 1 - sep / 120) + illum * 0.35 * min(1.0, s["moon_alt"] / 30)
                q *= max(0.05, 1 - glare * (0.35 if narrowband else 0.8 if preset_key == "cluster" else 1.3))
            per_sample.append(q)
        good = [q for q in per_sample if q > 0]
        if not good:
            continue
        # Best contiguous window above the altitude limit.
        best, run, best_run = (0, 0), None, 0.0
        for i, q in enumerate(per_sample + [0.0]):
            if q > 0 and run is None:
                run = i
            elif q <= 0 and run is not None:
                total = sum(per_sample[run:i])
                if total > best_run:
                    best, best_run = (run, i - 1), total
                run = None
        hours = len(good) * STEP_MIN / 60
        size = tg["size"]
        fit = 1.0 if 12 <= size <= 180 else 0.75 if 6 <= size < 12 or 180 < size <= 300 else 0.45
        if tg["kind"] == "P":
            fit = max(fit, 0.8)
        score = sum(good) * STEP_MIN / 60 * fit
        peak_i = max(range(len(per_sample)), key=lambda i: per_sample[i])
        peak_alt, peak_az = alt_az(tg["ra"], tg["dec"], lat, dark[peak_i]["lst"])
        min_moon_sep = min((separation(tg["ra"], tg["dec"], *s["moon"]) for s in moon_up), default=None)
        preset = presets[preset_key]
        win_s = (best[1] - best[0] + 1) * STEP_MIN * 60
        results.append({
            "id": tg["id"], "name": tg["name"], "kind": kind_label, "ra_hours": round(tg["ra"] / 15, 5),
            "dec": round(tg["dec"], 4), "size_arcmin": size, "score": round(score, 2), "hours_up": round(hours, 1),
            "window_start": dark[best[0]]["t"].isoformat(), "window_end": (dark[best[1]]["t"] + timedelta(minutes=STEP_MIN)).isoformat(),
            "peak_alt": round(peak_alt), "peak_dir": _compass(peak_az), "peak_time": dark[peak_i]["t"].isoformat(),
            "moon_sep": round(min_moon_sep) if min_moon_sep is not None else None, "preset": preset_key,
            "frames": int(min(720, max(30, win_s * 0.8 // (preset["exposure"] + 3)))),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[:limit]
    return {"date": date, "lat": lat, "lon": lon,
            "dark": {"start": dark[0]["t"].isoformat(), "end": (dark[-1]["t"] + timedelta(minutes=STEP_MIN)).isoformat(),
                     "kind": "astronomical" if dark_limit == -18 else "nautical (no full darkness tonight)"},
            "moon": {"illumination": round(illum * 100), "up_hours": round(len(moon_up) * STEP_MIN / 60, 1)},
            "targets": top, "schedule": _schedule(top, dark, lat), "presets": presets,
            "mount": mount}


def _schedule(top: list[dict], dark: list[dict], lat: float) -> list[dict]:
    """Split the dark hours between the best few targets, at least an hour each."""
    picks = top[:8]
    by_id = {t["id"]: t for t in picks}
    tg = {t["id"]: _targets_by_id()[t["id"]] for t in picks}
    slot_len = 3  # 30-minute slots
    slots = []
    for i in range(0, len(dark), slot_len):
        chunk = dark[i:i + slot_len]
        best, best_q = None, 0.0
        for tid, t in tg.items():
            q = sum(max(0.0, math.sin(math.radians(alt_az(t["ra"], t["dec"], lat, s["lst"])[0])))
                    for s in chunk if alt_az(t["ra"], t["dec"], lat, s["lst"])[0] >= MIN_ALT)
            q *= by_id[tid]["score"]
            if q > best_q:
                best, best_q = tid, q
        slots.append([best, chunk[0]["t"], chunk[-1]["t"] + timedelta(minutes=STEP_MIN)])
    runs = []
    for tid, a, b in slots:
        if runs and runs[-1][0] == tid:
            runs[-1][2] = b
        else:
            runs.append([tid, a, b])
    # Fold blocks shorter than an hour into a neighbour so there's time to set up and stack.
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (tid, a, b) in enumerate(runs):
            if (b - a) < timedelta(minutes=60):
                j = i - 1 if i > 0 else i + 1
                if j < i:
                    runs[j][2] = b
                else:
                    runs[j][1] = a
                runs.pop(i)
                changed = True
                break
    return [{"id": tid, "name": by_id[tid]["name"] if tid else "Nothing well placed",
             "start": a.isoformat(), "end": b.isoformat()} for tid, a, b in runs]


def _targets_by_id() -> dict:
    return {t["id"]: t for t in _targets()}
