"""Ask Claude to assess an image and plan the processing."""
from __future__ import annotations

import json

import anthropic

from . import config, history

SIRIL_COMMANDS = ["autostretch", "subsky", "rmgreen", "platesolve", "spcc", "pcc", "denoise", "starnet",
                  "asinh", "ght", "satu", "crop", "rl", "unsharp", "savetif", "save"]

# What tuning on this owner's own DWARF 3 data measured (2026-10-27/28). Kept in the prompts so Claude's advice agrees with
# what Auto-apply and MOON STACK do, and with what was found to work; update it as new rounds teach something.
DEEP_FINDINGS = """

Measured on this owner's DWARF 3 data (C 33 Eastern Veil, Duo-band, 218 min of 15 s subs), in many measured rounds \
against the DWARF's own processed picture of the same data. Trust these over generic advice:
- The app's Auto-apply already does: Siril restack (keep about 88 to 95% x roundness), GraXpert background extraction \
(smoothing 0.4 for nebulae: 0.2 left a dark moat around C 33), plate solve + SPCC, GraXpert AI deconvolution \
(object 0.5, stellar up to 0.3 at the measured FWHM), GraXpert denoise (0.7 under 1 h, 0.6 under 3 h, 0.5 above) plus colour \
noise smoothing. Recommend changes to these only when the measurements show a reason.
- For Duo-band emission nebulae with StarNet installed, the finish that won is layered: StarNet removes the stars \
right after a colour-keeping arcsinh stretch; background flattening, the nebula lift, sky smoothing and saturation are \
done on the starless layer; the stars are screened back tightened. On C 33 this reached nebula-to-sky-noise contrast \
74 (natural) and 79 (HOO), against 58 for the DWARF's own picture and 32 for a single-image finish.
- Targets that looked best: sky background about 8.5 to 9% (11 to 12% looked washed grey), nebula about 0.22 above the sky \
for strong H-alpha, sky noise about 0.003. H-alpha should be brick red: green about 0.5 of red on H-alpha areas (0.8 \
turned it orange-yellow, 0.9 yellow); OIII teal-blue. For HOO, OIII scaled to about 0.75 of H-alpha's strength is \
balanced (1.1 went cyan).
- The edges of long alt-az sessions (field-rotation wedges) are noisier and glow: they need their own flattening and \
smoothing; a subject mask must not count glow touching the frame border as nebula.
- Stars: reduce lightly (about 18%) in rich Milky Way fields; 30 to 42% left the Cygnus field looking empty. Keep star \
colour (white, blue, orange) but remove a shared tint.
- Measure sky and noise on the darker half of the image, never a whole-image median: a big nebula counts as sky \
otherwise. Over-smoothing the sky (grain below about 0.05 of its level) looks plastic.
- Pictures saved before app 0.14.0 were upside down compared with the DWARF's own (FITS rows run bottom-up).
Capture lessons for capture_tips: the Duo-band filter only for emission and planetary nebulae (it starves reflection \
nebulae like M 45, galaxies and clusters, which shine by starlight: use Astro); when stars are elongated across the \
whole frame, check wind and that the DWARF is level before blaming tracking."""

MOON_FINDINGS = """

Measured on this owner's DWARF 3 Moon videos (4K H.265, 2026-04-30 and later), scored by fine-detail sharpness, \
high-frequency content and detail-to-noise. Trust these over generic advice:
- The app's MOON STACK already does this, and it gave about 3x the fine detail of the owner's AutoStakkert stack of the \
same video at a better detail-to-noise ratio (36 against 21): best 5% of frames (5% beat 10% and 20%; at least 50), \
reference averaged from the 8 sharpest, global rotation + shift then 128 px patches, sharper frames weighted higher, \
then highland white balance, a gentle edge-preserving denoise, Richardson-Lucy 30 iterations (15 before the denoise was lightened) with a Gaussian PSF \
sigma 1.2 px, three-scale detail gains (fine +80%, medium +50%, coarse +20%), the limb protected, a soft highlight shoulder.
- Tone and colour (2026-10-23, the owner found a disc median of 0.60 with 0.7% near white "blown out and poor"): \
white point at the disc's 99.97th percentile, gamma to a disc median of about 0.48, soft shoulder from 0.82. Natural \
'mineral Moon' colour: large-scale colour ratios with the brightness- and position-linked part (twilight scatter, channel \
response) fitted out, highlands neutral, kept faint (0.8x; 3x and 1.5x read as teal/pink false colour to the owner) and faded over \
the outer 40 px of the limb; disc median 0.42 with a gentle local contrast lift; a denoise set from the measured noise \
(bilateral sigmaColor about 13x the green-channel noise MAD: 0.015 on 2026-10-23, 0.0225 on the noisier 2026-04-30), \
none where a channel clipped. The owner wants a TRUE black sky (0, not 1-2/255 of noise) and a near-grey Moon like \
DWARFLAB's own sample images. Take detail from green when red is near clipping.
- So in AutoStakkert terms: stack about 5% for this camera, alignment point size hardly matters (seeing shifts were \
about 0.3 px), and do NOT drizzle or upsample the 4K video: the 8-bit H.265 frames hold no sub-pixel detail (1.5x \
measured worse). ImPPG equivalent: sigma about 1.2, light iterations, after a light denoise.
- Never stack a softer video or photo session in with a sharper one: on 2026-04-30 adding a video whose median frame \
was 0.58x as sharp cut the stack's sharpness from 0.102 to 0.075. MOON STACK now leaves such sources out on its own; \
HD video and the raw 1920x1080 FITS only go in when they don't cost resolution. The DWARF 3 tele's 1080p video and raw \
frames are a CROP at the same pixel scale as its 4K video (Moon radius 313 vs 323 px on 2026-10-23), not a downscaled frame: \
never upscale them 2x; match the Moon's measured size.
- Cubic-warped frames can go slightly negative at the limb; clip before any division (an app bug once made a black Moon).
Capture lessons for capture_tips: prefer 4K video over 1080p; keep the whole disc well inside the frame (frames where it \
touches the edge are dropped); check focus between videos (on 2026-04-30 one video was half as sharp as the next); \
several shorter videos in the steadiest seeing beat one long one; if the red channel clips on the bright highlands \
(it did on 13% of the disc on 2026-10-23), lower exposure or gain about one stop; shoot higher than dusk twilight when possible."""

SYSTEM_PROMPT = """You are an astrophotography processing coach built into a desktop helper for a DWARF 3 smart telescope \
(150 mm f/4.3 telephoto, Sony IMX678 colour sensor, about 2.75 arcsec per pixel, alt-az mount with field rotation, \
built-in filters VIS / Astro (UV-IR cut) / Duo-band (H-alpha + OIII)). The owner processes with free tools: \
GraXpert (AI background extraction and AI denoise), Siril (plate solve, colour calibration, stretching, scripts), \
StarNet (star removal) and a final polish in an image editor.

You receive: local measurements of the image (authoritative for numbers), a whole-frame preview and a 100% centre crop \
(both auto-stretched for display only if the data is linear, so do not mistake the display stretch for the real state), \
the DWARF shotsInfo.json if found, and a log of the owner's previous sessions with their own ratings.

How to read the measurements:
- gradient.spread_percent_of_background above ~10% or spread_in_noise_sigmas above ~5 means background extraction is needed. \
The 8x8 tile grid shows the shape: a smooth ramp suits GraXpert AI with Subtraction; vignetting (dark corners, bright centre) suits Division.
- stars.elongation_median above ~1.25 or elongated_percent above ~40 means trailing or wind; edge-only elongation means field \
rotation or optics, not tracking. Field rotation is normal on an alt-az mount for long sessions near the zenith.
- fwhm_px_median around 2.5 to 4 px is normal for DWARF 3; above 5 px suggests poor seeing, focus or dew.
- color.background_cast 'green' is normal for a raw one-shot-colour stack; fix with colour calibration then rmgreen.
- stretch.state 'linear' means everything up to and including denoise and star removal must happen before stretching.
- snr_proxy is relative only; compare against the owner's history rather than absolute thresholds.

Standard order for linear data: crop stacking edges -> GraXpert background extraction -> Siril plate solve + SPCC \
(or PCC) colour calibration -> rmgreen if still green -> GraXpert AI denoise (light, linear) -> StarNet star removal \
(optional, for nebulae and galaxies) -> stretch (GHT or asinh, starless and stars separately if star-removed) -> \
saturation -> recombine stars -> final polish. Skip steps the measurements show are unnecessary and say why. \
For an already-stretched JPG/PNG, recommend light corrections only and suggest using the FITS next time.

Settings ranges you may recommend: GraXpert smoothing 0.0 to 1.0 (0.4 for nebulae and 0.5 to 0.6 for galaxies and \
clusters on this owner's data, see the findings below), correction Subtraction or Division, denoise strength 0.0 to 1.0 (0.3 to 0.6 typical). Give exact values, the \
menu path in the app, and one line of why. Use the owner's history: when they rated a past setting poorly or noted \
what they changed, adjust toward what worked for them, and say you did.

siril_script: commands for siril-cli that run AFTER GraXpert, on its output. Use only these commands: \
""" + ", ".join(SIRIL_COMMANDS) + """. Args are the text after the command name, for example \
{"command": "subsky", "args": "1 -samples=20"}. Do not include load, cd or file paths; the app adds them. \
Leave the list empty if Siril is not needed. Write for a beginner who is keen to learn, in plain language.

identification: name the main target. When a plate solve is present it is authoritative: use its subject and \
in_field list (positions are already worked out). Otherwise use shotsInfo target and RA/Dec and confirm from the image, and \
list other catalogued objects visible in the frame with where they sit (the DWARF 3 telephoto frame is about \
2.9 x 1.6 degrees). Only list objects you can actually see or that certainly fall in this field; set confidence honestly.

spectral_overview: the DWARF 3 records three broad colour channels, not a spectrum. Combine the measured \
color_emission ratios with what is known about this object's light: for emission nebulae the H-alpha 656 nm, \
H-beta 486 nm, OIII 496/501 nm and SII 672 nm lines; for galaxies and clusters starlight continuum plus any \
H II regions; for reflection nebulae scattered blue starlight. Mark each line as seen in this image or not \
(H-alpha shows as red, OIII as teal; with the Duo-band filter red is essentially H-alpha and green+blue OIII). \
Describe the star colour mix (blue hot stars to red cool stars) in plain terms. Note that colours are uncalibrated \
until SPCC/PCC is run.""" + DEEP_FINDINGS

MOON_PROMPT = """You are a lunar imaging coach built into a desktop helper for a DWARF 3 smart telescope \
(150 mm f/4.3 telephoto, Sony IMX678 colour sensor, 2 um pixels, about 2.75 arcsec per pixel, so the full Moon is \
roughly 650 px across; the wide-angle camera is not used for the Moon). The owner captures the Moon as video or bursts \
and processes on Windows with lunar tools, typically PIPP (convert/centre/crop video, reject bad frames), AutoStakkert \
(analyse, pick best frames, align with alignment points, stack, optional drizzle), RegiStax 6 (wavelet sharpening) or \
ImPPG (Lucy-Richardson deconvolution, unsharp mask, tone curve), and an image editor for the final touches. They may \
also use Siril for deconvolution (rl) and unsharp masking.

You receive local measurements (authoritative for numbers), previews, possibly shotsInfo.json, and the owner's past \
lunar sessions with their own ratings. Two kinds of input:
1) A VIDEO (or SER): metrics are per-frame sharpness. frames_within_15pct_of_best_percent and the p90/p10 spread show \
how much seeing varied. Recommend the AutoStakkert 'frames to stack' as a percentage tied to these numbers (steady \
seeing: 25 to 50%; poor, highly variable seeing: 5 to 15%), quality estimator settings (Laplace delta, noise robust 3 to 5), \
alignment point size (25 to 50 for this image scale; smaller only with good seeing), surface vs planet alignment \
(surface for the Moon), and whether 1.5x drizzle is worth it (only with many good frames). If drift or edge-touching is \
high, recommend PIPP centring/cropping first. If brightness_variation is high (clouds), recommend PIPP quality filtering.
2) A STACKED IMAGE: sharpness_index is relative (compare with history), saturated_disk_percent above ~0.1 means the \
bright limb or highlands are clipped (fix at capture: lower exposure or gain), sharpening_halo_sigma above ~4 means a dark \
ring outside the limb from over-sharpening (back off). Recommend RegiStax wavelet layers (linked, Gaussian, default \
denoise per layer) with concrete layer values, OR ImPPG deconvolution sigma (1.0 to 2.0 px) and iterations (30 to 80), \
then gentle unsharp mask, and a tone curve. For colour data, mention the 'mineral Moon' saturation boost option.

Give exact values, the menu path in each program, and one line of why. Use the owner's history: when they rated past \
settings poorly or noted what they changed, adjust toward what worked for them and say so. For the graxpert object set \
run_background and run_denoise to false (GraXpert is not for lunar work). siril_script: leave empty unless Siril \
deconvolution genuinely helps; if used, only these commands: """ + ", ".join(SIRIL_COMMANDS) + """, e.g. \
{"command": "rl", "args": "-alpha=3000 -iters=10"}; no load, cd or paths. capture_tips should cover exposure, gain, \
frame count, focus and timing (Moon altitude, seeing) for next time. Write for a beginner who is keen to learn, in \
plain language.

identification: primary_target is the Moon with its phase; also_in_frame lists the named craters, maria, mountain \
ranges and rilles that are clearly visible, especially along the terminator, with where they are in the frame.

spectral_overview: moonlight is reflected sunlight (a solar spectrum with Fraunhofer absorption lines), so there are no \
emission lines; list lines as an empty array. In summary, explain the subtle colour differences of lunar soils \
(blue titanium-rich basalts like Mare Tranquillitatis, orange-brown titanium-poor basalts like Mare Serenitatis) and \
whether this image's colour data could show them in a 'mineral Moon' saturation boost. star_colours: 'none in frame'.""" + MOON_FINDINGS

_STR = {"type": "string"}
_NUM = {"type": "number"}

ASSESSMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["overall_summary", "quality_score", "identification", "spectral_overview", "issues", "steps",
                 "graxpert", "siril_script", "capture_tips", "learned_note"],
    "properties": {
        "overall_summary": _STR,
        "quality_score": {"type": "integer", "description": "1 to 10"},
        "issues": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["area", "severity", "finding"],
            "properties": {
                "area": {"type": "string", "enum": ["gradient", "noise", "stars", "color", "stretch", "clipping",
                                                    "framing", "tracking", "focus", "seeing",
                                                    "sharpening", "exposure", "other"]},
                "severity": {"type": "string", "enum": ["none", "minor", "moderate", "major"]},
                "finding": _STR,
            }}},
        "steps": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["tool", "action", "menu_path", "settings", "why"],
            "properties": {
                "tool": {"type": "string", "enum": ["GraXpert", "Siril", "StarNet", "PIPP", "AutoStakkert", "RegiStax",
                                                    "ImPPG", "Image editor", "Next capture"]},
                "action": _STR,
                "menu_path": _STR,
                "settings": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["name", "value"],
                    "properties": {"name": _STR, "value": _STR}}},
                "why": _STR,
            }}},
        "graxpert": {
            "type": "object", "additionalProperties": False,
            "required": ["run_background", "correction", "smoothing", "run_denoise", "denoise_strength"],
            "properties": {
                "run_background": {"type": "boolean"},
                "correction": {"type": "string", "enum": ["Subtraction", "Division"]},
                "smoothing": _NUM,
                "run_denoise": {"type": "boolean"},
                "denoise_strength": _NUM,
            }},
        "siril_script": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["command", "args", "comment"],
            "properties": {"command": {"type": "string", "enum": SIRIL_COMMANDS}, "args": _STR, "comment": _STR}}},
        "identification": {
            "type": "object", "additionalProperties": False,
            "required": ["primary_target", "object_type", "constellation", "confidence", "description", "also_in_frame"],
            "properties": {
                "primary_target": _STR,
                "object_type": _STR,
                "constellation": _STR,
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "description": {"type": "string", "description": "2 to 4 sentences: what it is, distance, size, why it looks the way it does."},
                "also_in_frame": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["name", "where", "note"],
                    "properties": {"name": _STR, "where": {"type": "string", "description": "position in the frame, e.g. upper left"},
                                   "note": _STR}}},
            }},
        "spectral_overview": {
            "type": "object", "additionalProperties": False,
            "required": ["summary", "lines", "star_colours"],
            "properties": {
                "summary": {"type": "string", "description": "What kind of light this object emits and what the image's colours show."},
                "lines": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["name", "wavelength_nm", "seen_in_image", "note"],
                    "properties": {"name": _STR, "wavelength_nm": _NUM,
                                   "seen_in_image": {"type": "string", "enum": ["yes", "weak", "no", "not measurable"]},
                                   "note": _STR}}},
                "star_colours": _STR,
            }},
        "capture_tips": {"type": "array", "items": _STR},
        "learned_note": {"type": "string",
                         "description": "One sentence worth remembering for future sessions, e.g. a recurring issue."},
    },
}


class AssessmentError(Exception):
    pass


def _clamp(v, lo, hi, default):
    try:
        return min(hi, max(lo, float(v)))
    except (TypeError, ValueError):
        return default


def _user_content(analysis: dict, session_note: str) -> list:
    metrics = analysis["metrics"]
    text = (
        "Measurements (JSON):\n" + json.dumps(metrics) +
        "\n\nPlate solve / catalogue identification:\n" + (json.dumps(analysis["field"]) if analysis.get("field") else "not available") +
        "\n\nDWARF shotsInfo.json:\n" + (json.dumps(analysis.get("shot_info")) if analysis.get("shot_info") else "not found") +
        "\n\nPrevious sessions:\n" + history.context_for(analysis.get("shot_info") or {}, analysis.get("mode", "deepsky")) +
        "\n\n" + history.runs_context(analysis.get("mode", "deepsky")) +
        "\n\nOwner's note for this session: " + (session_note or "none") +
        "\n\n" + (analysis.get("preview_note") or "Image 1 is the whole frame, image 2 is a 100% crop"
                   + (" (both display-stretched; the data itself is linear)." if analysis.get("display_stretched") else "."))
        + ("\nInput type: lunar " + analysis.get("kind", "image") + "." if analysis.get("mode") == "moon" else "") +
        "\nAssess the image and plan the processing."
    )
    img = lambda b64: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
    return [img(analysis["overview_jpeg_b64"]), img(analysis["crop_jpeg_b64"]), {"type": "text", "text": text}]


def assess(analysis: dict, session_note: str = "") -> tuple[dict, str]:
    key = config.api_key()
    if not key:
        raise AssessmentError("No Anthropic API key set. Add one in Settings, or set ANTHROPIC_API_KEY.")
    cfg = config.load()
    client = anthropic.Anthropic(api_key=key)
    request = dict(
        model=cfg["model"],
        max_tokens=16000,
        system=MOON_PROMPT if analysis.get("mode") == "moon" else SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_content(analysis, session_note)}],
        output_config={"effort": cfg.get("effort") or "medium",
                       "format": {"type": "json_schema", "schema": ASSESSMENT_SCHEMA}},
    )
    try:
        try:
            # If a safety classifier declines, the API retries on a suitable model within the same call.
            response = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                                   **request)
        except anthropic.BadRequestError as e:
            if "fallback" not in str(e).lower():
                raise
            response = client.messages.create(**request)
    except anthropic.AuthenticationError:
        raise AssessmentError("The API key was rejected. Check it in Settings.")
    except anthropic.PermissionDeniedError:
        raise AssessmentError("This API key can't use that model. Check your Anthropic Console plan.")
    except anthropic.NotFoundError:
        raise AssessmentError(f"Model {cfg['model']!r} not found. Change it in Settings.")
    except anthropic.RateLimitError:
        raise AssessmentError("Rate limited by the API. Wait a minute and try again.")
    except anthropic.BadRequestError as e:
        raise AssessmentError(f"The API rejected the request: {e.message}")
    except anthropic.APIStatusError as e:
        raise AssessmentError(f"API error {e.status_code}. Try again shortly.")
    except anthropic.APIConnectionError:
        raise AssessmentError("Couldn't reach the API. Check your internet connection.")

    if response.stop_reason == "refusal":
        raise AssessmentError("Claude declined this request.")
    if response.stop_reason == "max_tokens":
        raise AssessmentError("The answer was cut off. Try again.")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        raise AssessmentError("Empty response from the API.")
    try:
        result = json.loads(text)
    except ValueError:
        raise AssessmentError("Couldn't read the API's answer. Try again.")

    gx = result.get("graxpert") or {}
    gx["smoothing"] = _clamp(gx.get("smoothing"), 0.0, 1.0, 0.1)
    gx["denoise_strength"] = _clamp(gx.get("denoise_strength"), 0.0, 1.0, 0.5)
    result["graxpert"] = gx
    result["quality_score"] = int(_clamp(result.get("quality_score"), 1, 10, 5))
    return result, response.model


COMPARE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["verdict", "ranking", "biggest_gap", "next_steps"],
    "properties": {
        "verdict": _STR,
        "ranking": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["label", "rank", "score", "strengths", "weaknesses"],
            "properties": {"label": _STR, "rank": {"type": "integer"}, "score": {"type": "integer", "description": "1 to 10"},
                           "strengths": _STR, "weaknesses": _STR}}},
        "biggest_gap": {"type": "string", "description": "The one thing that most separates the owner's image from the best one."},
        "next_steps": {"type": "array", "items": _STR},
    },
}

COMPARE_PROMPT = """You compare astrophotos of the same target for the owner of a DWARF 3 smart telescope. Image 1 is \
the owner's current image. The others are reference images they chose (their own earlier best shots, or photos \
they saved from other DWARF users). Judge only what is visible: detail, noise, star shape, colour balance, \
background, framing and processing. Rank all images (rank 1 = best), score each 1 to 10 on the same scale, and be \
encouraging but honest. Keep in mind references may have far more integration time; say so when that explains the gap. \
A reference may be the DWARF's own processed picture of the same data: on this owner's C 33 that picture had a darker \
sky (8.6%) and more vivid H-alpha than the app's 0.13 finish but a less even, warm-cast background; the app's layered \
finish (0.15 on) beat it on contrast, flatness and neutral sky. Name which of these the images show."""


def compare(current_b64: str, refs: list[tuple[str, str]], target: str) -> dict:
    """refs: (label, jpeg base64) pairs, at most 3."""
    key = config.api_key()
    if not key:
        raise AssessmentError("No Anthropic API key set. Add one in Settings, or set ANTHROPIC_API_KEY.")
    cfg = config.load()
    img = lambda b64: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
    content = [{"type": "text", "text": "Image 1 (label 'Your image'):"}, img(current_b64)]
    for i, (label, b64) in enumerate(refs[:3], start=2):
        content += [{"type": "text", "text": f"Image {i} (label '{label}'):"}, img(b64)]
    content.append({"type": "text", "text": f"Target: {target or 'unknown'}. Compare and rank these images."})
    client = anthropic.Anthropic(api_key=key)
    try:
        response = client.messages.create(
            model=cfg["model"], max_tokens=8000, system=COMPARE_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": COMPARE_SCHEMA}})
    except anthropic.APIError as e:
        raise AssessmentError(f"The comparison request failed: {getattr(e, 'message', e)}")
    if response.stop_reason in ("refusal", "max_tokens"):
        raise AssessmentError("The comparison didn't complete. Try again.")
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return json.loads(text)
    except ValueError:
        raise AssessmentError("Couldn't read the comparison. Try again.")
