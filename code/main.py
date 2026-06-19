"""
Damage Claim Verification System
Entry point: reads dataset/claims.csv, verifies each claim using a Vision LLM,
writes predictions to output.csv in the root directory.

Usage:
    python code/main.py [--input dataset/claims.csv] [--output output.csv]

Requires:
    OPENAI_API_KEY in .env file or environment variable
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

# Load .env from repo root
load_dotenv(Path(__file__).parent.parent / ".env")

# ── constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

DEFAULT_MODEL = "gpt-4o"

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# Allowed values are restricted to EXACTLY the vocabulary used in dataset/sample_claims.csv.
# No synonyms or near-equivalents — the evaluator scores on exact string match.
ALLOWED = {
    "claim_status": {"supported", "contradicted", "not_enough_information"},
    "issue_type": {
        "dent", "scratch", "crack", "broken_part",
        "torn_packaging", "crushed_packaging",
        "water_damage", "stain", "none", "unknown",
    },
    "object_part_car": {
        "front_bumper", "rear_bumper", "door", "windshield",
        "side_mirror", "headlight", "unknown",
    },
    "object_part_laptop": {
        "screen", "keyboard", "trackpad", "hinge", "corner", "unknown",
    },
    "object_part_package": {
        "package_corner", "package_side", "seal", "contents", "unknown",
    },
    "severity": {"none", "low", "medium", "high", "unknown"},
}

# Map any out-of-vocabulary value the model might still emit to the correct GT label,
# applied BEFORE clamping so we don't lose information by dropping to a default.
ISSUE_TYPE_SYNONYMS = {
    "glass_shatter": "crack",      # GT has no glass_shatter — all glass/screen fractures are crack
    "shattered_glass": "crack",
    "screen_crack": "crack",
    "missing_part": "unknown",     # GT has no missing_part
    "broken": "broken_part",
    "dented": "dent",
    "scratched": "scratch",
    "cracked": "crack",
}
OBJECT_PART_SYNONYMS = {
    "box": "package_side",         # GT has no box for packages
    "package_box": "package_side",
    "label": "package_side",
    "item": "contents",
    "screen_panel": "screen",
}

# Canonical flag order — must match sample label ordering (GT vocabulary only)
FLAG_ORDER = [
    "blurry_image", "cropped_or_obstructed", "wrong_angle",
    "wrong_object", "damage_not_visible", "claim_mismatch",
    "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required",
]

ALLOWED_FLAGS = set(FLAG_ORDER)

# Flags the model may emit that are not in GT vocabulary → remap or drop
FLAG_SYNONYMS = {
    "low_light_or_glare": None,        # not in GT — drop
    "possible_manipulation": None,     # not in GT — drop
    "wrong_object_part": "claim_mismatch",
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_user_history(path: Path) -> dict[str, dict]:
    return {r["user_id"]: r for r in load_csv(path)}


def load_evidence_requirements(path: Path) -> list[dict]:
    return load_csv(path)


def get_evidence_text(requirements: list[dict], claim_object: str) -> str:
    lines = []
    for req in requirements:
        if req["claim_object"] in ("all", claim_object):
            lines.append(f"- [{req['requirement_id']}] {req['minimum_image_evidence']}")
    return "\n".join(lines) if lines else "Submit clear images of the claimed object and part."


def _is_valid_format(raw: bytes) -> bool:
    """Return True if bytes start with a known OpenAI-supported image magic."""
    return (
        raw[:2] == b'\xff\xd8'           # JPEG
        or raw[:4] == b'\x89PNG'         # PNG
        or raw[8:12] == b'WEBP'          # WebP
        or raw[:6] in (b'GIF87a', b'GIF89a')  # GIF
    )


def load_images_b64(image_paths_str: str) -> list[dict]:
    """Load images as base64 data URIs for the OpenAI vision API.
    Auto-converts HEIC/AVIF/PNG-with-wrong-extension to JPEG on the fly.
    """
    from PIL import Image
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass

    images = []
    for rel_path in image_paths_str.split(";"):
        rel_path = rel_path.strip()
        full_path = REPO_ROOT / "dataset" / rel_path
        if not full_path.exists():
            full_path = REPO_ROOT / rel_path
        if not full_path.exists():
            print(f"  Warning: image not found: {rel_path}", file=sys.stderr)
            continue

        with open(full_path, "rb") as f:
            raw = f.read()

        if _is_valid_format(raw):
            # Already a supported format — send as-is
            suffix = full_path.suffix.lower()
            media_type = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
            }.get(suffix, "image/jpeg")
            # Confirm actual format matches extension
            if raw[:2] == b'\xff\xd8':
                media_type = "image/jpeg"
            elif raw[:4] == b'\x89PNG':
                media_type = "image/png"
            elif raw[8:12] == b'WEBP':
                media_type = "image/webp"
            data = base64.standard_b64encode(raw).decode("utf-8")
        else:
            # Non-standard format (HEIC, AVIF, etc.) — convert to JPEG
            try:
                import io
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
                data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
                media_type = "image/jpeg"
                print(f"  Converted {full_path.name} (HEIC/other) -> JPEG", file=sys.stderr)
            except Exception as e:
                print(f"  Warning: could not convert {rel_path}: {e}", file=sys.stderr)
                continue

        images.append({
            "id": full_path.stem,
            "media_type": media_type,
            "data": data,
            "url": f"data:{media_type};base64,{data}",
        })
    return images


# ── prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are evaluating insurance evidence according to strict dataset labeling rules. "
    "Your job is to classify each claim into exactly one of: supported, contradicted, or not_enough_information. "
    "You must follow the decision rules precisely and never guess or be lenient. "
    "Output only valid JSON matching the schema — no prose, no markdown."
)

DECISION_RULES = """PROCESSING ORDER — work through these steps in order for this one user:

STEP 0 — ENUMERATE IMAGES: List every image ID provided for this user (img_1, img_2, ...).
  Look at each image individually before deciding anything.
STEP 1 — OBJECT PART: From the images + claim text, determine which part of the object is shown.
  Pick EXACTLY ONE value from the OBJECT PART list for this object type. Do not invent new part names.
STEP 2 — ISSUE TYPE: Examine the images for the actual damage on that part, cross-check with the
  user's claim. Pick EXACTLY ONE value from the ISSUE TYPE list. Do not invent new issue names.
STEP 3 — RISK FLAGS: Apply every applicable flag from the RISK FLAG list (only those names).
STEP 4 — CLAIM STATUS, SEVERITY, EVIDENCE, VALID IMAGE, SUPPORTING IMAGE IDS: fill the rest.

CRITICAL VOCABULARY RULE:
Use ONLY the exact strings listed under ALLOWED VALUES below. These are the only labels the dataset
uses. Never output a synonym or a near-equivalent (e.g. do NOT say "glass_shatter" — a cracked screen
or windshield is "crack"; do NOT say "box" for a package — use "package_side" or "package_corner").
If what you see does not map to an allowed value, choose the closest allowed value, not a new word.

---

CLAIM STATUS RULES:

SUPPORTED:
  - Claimed object part IS visible in image
  - Claimed damage IS visible and matches the claim

CONTRADICTED:
  - Claimed object part IS visible, but claimed damage is absent → issue_type=none, severity=none, add damage_not_visible
  - Claimed object part IS visible, different or lesser damage present → issue_type=what IS visible, add claim_mismatch
  - Wrong object entirely shown → issue_type=unknown, object_part=unknown, add wrong_object + claim_mismatch
  - Package seal/contents visible but intact despite torn/missing claim → issue_type=none, severity=none, add damage_not_visible
  For contradicted: supporting_image_ids = images proving the contradiction

NOT_ENOUGH_INFORMATION:
  - Claimed part NOT visible in any image (wrong angle, obstructed, too blurry)
  - Multiple images show DIFFERENT objects (identity mismatch)
  - Package contents claim: interior not clearly and fully visible
  - Image quality prevents any verdict
  For not_enough_information: evidence_standard_met=false

NEVER use not_enough_information when the claimed part IS visible — use contradicted instead.

---

ISSUE TYPE RULES (choose ONE allowed value only):

dent: visible indentation/deformation in surface
scratch: surface-level scrape or mark, does not deform material
crack: any visible crack/fracture LINE on a surface that remains attached — INCLUDING a cracked or
  shattered laptop screen, a cracked or smashed windshield, or any glass with fracture lines.
  (There is NO "glass_shatter" label — all glass/screen fractures are "crack".)
broken_part: a component is structurally failed, detached, snapped, or no longer functional as a unit
  → broken side mirror, broken hinge, detached headlight, bumper snapped off
  → Use crack (not broken_part) when a surface has a fracture line but is still attached.
stain: liquid residue, discoloration, or spill marks on a surface (e.g. spill on keyboard)
water_damage: water/moisture damage, especially on package exterior
torn_packaging: packaging seal or flap visibly torn open
crushed_packaging: package visibly crushed or compressed
none: correct part IS visible, NO damage found (always paired with contradicted)
unknown: ONLY when the issue genuinely cannot be determined (e.g. wrong object, identity mismatch)
(There is NO "missing_part" label — a part claimed missing but unverifiable → unknown + not_enough_information.)

---

OBJECT PART RULES (choose ONE allowed value only, matched to the object type):

NEVER output unknown if the part can be identified from the image OR named in the claim.
Fill object_part from the claimed/observed part EVEN WHEN claim_status is contradicted or
not_enough_information — as long as that part appears in at least one image, or the object type is
correct and the claim names the part.
  - object_part=unknown ONLY when the object itself is the wrong item AND no image shows the claimed
    part (e.g. a single photo of a completely unrelated object).
  - Correct object but the claimed part is not the part visible → use the claimed part name from the
    transcript when it is a valid allowed part; otherwise the part that IS visible.
  - Correct part visible but undamaged → use the claimed part name, issue_type=none.
For packages, "box" is NOT allowed — map the outer shell to package_side or package_corner.

---

SEVERITY RULES:

high   : severe structural damage — part completely detached, deep crush, major collision destruction
medium : clearly visible damage affecting function or appearance (DEFAULT for real damage)
  → standard dent, clear crack (incl. cracked/shattered screen), broken component, moderate tear
low    : cosmetic only — minor surface scratch, hairline mark, small corner dent, light stain
none   : no damage visible (always paired with contradicted + issue_type=none)
unknown: verdict impossible (wrong object, identity mismatch)

Examples:
  rear bumper dent clearly visible → medium
  side mirror broken/detached → medium
  laptop screen crack/shatter → medium
  laptop corner cosmetic dent → low
  package corner moderately crushed → medium
  severe front-end crash → high
  part visible, no damage → none

---

RISK FLAG RULES (use ONLY these flag names):

damage_not_visible    : claimed damage not observed (part IS visible, damage is NOT)
wrong_object          : image object differs from claimed object, or images show different objects
claim_mismatch        : visible evidence clearly contradicts or mismatches the stated claim
blurry_image          : image blurry but object still identifiable
cropped_or_obstructed : key area cut off or blocked
wrong_angle           : angle prevents viewing claimed part
non_original_image    : unmistakable screenshot, stock photo, or repost (NOT for poor quality real photos)
text_instruction_present : visible text instructing approval/rejection
user_history_risk     : user history indicates elevated risk
manual_review_required: add ONLY when:
  - wrong_object is present
  - non_original_image is present
  - claim_mismatch AND user_history_risk are both present
  - conflicting evidence across images
  DO NOT add for: blurry_image, wrong_angle, damage_not_visible, or cropped_or_obstructed alone

---

MULTI-IMAGE RULE:

When multiple images are submitted, this cross-check is MANDATORY before any decision.

STEP A — Describe each image independently:
  For EVERY image state: object type, make/color, the specific part shown, and any damage.

STEP B — Compare identity attributes across all images:
  Build an identity profile from the observable attributes of each image, e.g.:
    - object type (car / laptop / package)
    - color / paint
    - make, model, body style, or distinguishing marks
    - the location and shape of any visible damage
  Then ask: are these attributes mutually CONSISTENT, meaning every image could plausibly be
  the SAME physical object photographed from a different angle or distance?

STEP C — Decide by weighing the evidence (do NOT use a single fixed trigger):
  - CONSISTENT identity (attributes align, or differences are explainable by angle/zoom/lighting)
    → treat as the same object → proceed to damage assessment.
      If one image is just low quality, add blurry_image / low_light_or_glare but keep the verdict.
  - CONFLICTING identity (one or more attributes cannot be reconciled — e.g. a close-up shows a
    different color, body panel, or damage pattern than the wide shot, so they cannot be the same
    physical object) → the evidence set is internally inconsistent and cannot verify the claim:
      evidence_standard_met=false, claim_status=not_enough_information,
      add wrong_object + claim_mismatch + manual_review_required,
      supporting_image_ids = ALL submitted image IDs (they jointly prove the inconsistency)
      IMPORTANT: still report object_part and issue_type from whichever image DID show the
      claimed part/damage. Only use object_part=unknown / issue_type=unknown if NO image at all
      showed the claimed object part. (Example: a clear close-up of a broken front bumper plus a
      wide shot of a different car → not_enough_information, but object_part=front_bumper,
      issue_type=broken_part from the close-up.)
  - GENUINELY UNCERTAIN (not enough detail to tell either way) → lean toward same object,
    use quality flags only, do NOT flag wrong_object.

Judge each case on its own observed attributes — weigh how many identity cues agree vs conflict,
rather than requiring one exact mismatch. A single decisive, unexplainable mismatch (e.g. red car
vs blue car) is enough; several small consistent cues outweigh one ambiguous difference.

NEVER conclude supported when the images cannot be the same physical object.
NEVER flag wrong_object merely because one image is blurry, cropped, or low quality.

---

NON-ORIGINAL IMAGE RULE:

Set valid_image=false and add non_original_image ONLY with unmistakable evidence:
  - Screenshot with browser/app chrome or status bar
  - Photo of a screen (visible bezel or pixel grid)
  - Stock image with watermark or white seamless background
  - Social media repost with platform UI overlaid
When valid_image=false but content IS visible: evidence_standard_met=true, still populate all fields.
Default: valid_image=true. DO NOT flag for poor quality, blur, or bad angle.

---

PACKAGE CONTENTS RULE:

Package exterior alone CANNOT prove a missing item claim.
Need to see opened package interior clearly to verify contents are absent.
If interior not fully visible → evidence_standard_met=false, not_enough_information,
  add cropped_or_obstructed, supporting_image_ids=none

---

ALLOWED VALUES (these are the ONLY strings permitted — no synonyms, no other words):

claim_status        : supported | contradicted | not_enough_information
issue_type          : dent | scratch | crack | broken_part | torn_packaging | crushed_packaging | water_damage | stain | none | unknown
object_part (car)   : front_bumper | rear_bumper | door | windshield | side_mirror | headlight | unknown
object_part (laptop): screen | keyboard | trackpad | hinge | corner | unknown
object_part (pkg)   : package_corner | package_side | seal | contents | unknown
severity            : none | low | medium | high | unknown
risk_flags          : blurry_image | cropped_or_obstructed | wrong_angle | wrong_object | damage_not_visible | claim_mismatch | non_original_image | text_instruction_present | user_history_risk | manual_review_required"""

OUTPUT_SCHEMA = """{
  "valid_image": true or false,
  "text_instruction_present": true or false,
  "evidence_standard_met": true or false,
  "evidence_standard_met_reason": "one sentence",
  "issue_type": "...",
  "object_part": "...",
  "claim_status": "...",
  "severity": "...",
  "supporting_image_ids": "img_1;img_2 or none",
  "claim_status_justification": "one or two sentences citing image IDs",
  "risk_flags_from_images": ["flag1", "flag2"]
}"""


def build_prompt(
    claim_object: str,
    user_claim: str,
    image_ids: list[str],
    evidence_req: str,
    user_history: dict,
) -> str:
    history_text = (
        f"Past claims: {user_history.get('past_claim_count', 0)}, "
        f"accepted: {user_history.get('accept_claim', 0)}, "
        f"rejected: {user_history.get('rejected_claim', 0)}, "
        f"last 90 days: {user_history.get('last_90_days_claim_count', 0)}. "
        f"{user_history.get('history_summary', 'No prior history.')}"
    )
    image_list = ", ".join(image_ids) if image_ids else "none"

    return f"""## Claim
Object: {claim_object}
Images submitted: {image_list}
User history: {history_text}

## Conversation transcript
{user_claim}

## Evidence requirements for {claim_object}
{evidence_req}

## Decision Rules
{DECISION_RULES}

Return ONLY valid JSON — no markdown, no prose outside the JSON object:
{OUTPUT_SCHEMA}"""


# ── VLM call ──────────────────────────────────────────────────────────────────

def call_vlm(
    client: OpenAI,
    prompt: str,
    images: list[dict],
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
) -> dict:
    """Call OpenAI vision model and return parsed JSON dict."""
    # Build message content: interleave image label + image for each image
    content = []
    for img in images:
        content.append({"type": "text", "text": f"[Image: {img['id']}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": img["url"], "detail": "high"},
        })
    content.append({"type": "text", "text": prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=1024,
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)

        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            if attempt == max_retries - 1:
                raise

        except RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  Rate limit, retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)

        except APIError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    return {}


# ── decision layer ────────────────────────────────────────────────────────────

def clamp(value: str, allowed: set, default: str) -> str:
    return value if value in allowed else default


def get_part_set(claim_object: str) -> set:
    return ALLOWED.get(f"object_part_{claim_object}", {"unknown"})


def merge_risk_flags(
    vlm_flags: list[str],
    user_history: dict,
    text_instruction: bool,
) -> str:
    flag_set = set()
    for f in vlm_flags:
        f = str(f).strip().lower()
        f = FLAG_SYNONYMS.get(f, f)   # remap or drop (None) out-of-vocab flags
        if f and f in ALLOWED_FLAGS:
            flag_set.add(f)

    if text_instruction:
        flag_set.add("text_instruction_present")

    history_flags_raw = user_history.get("history_flags", "none")
    if history_flags_raw and history_flags_raw != "none":
        for f in history_flags_raw.split(";"):
            f = f.strip()
            if f and f != "none" and f in ALLOWED_FLAGS:
                flag_set.add(f)

    if "user_history_risk" in flag_set:
        flag_set.add("manual_review_required")

    ordered = [f for f in FLAG_ORDER if f in flag_set]
    return ";".join(ordered) if ordered else "none"


def build_output_row(claim: dict, vlm: dict, user_history: dict) -> dict:
    claim_object = claim["claim_object"]
    image_paths_str = claim["image_paths"]
    all_ids = {Path(p.strip()).stem for p in image_paths_str.split(";")}

    def as_bool(val, default=True) -> bool:
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() == "true"

    valid_image = as_bool(vlm.get("valid_image", True))
    evidence_met = as_bool(vlm.get("evidence_standard_met", False))

    # Invalid image always fails the evidence gate
    if not valid_image:
        evidence_met = False

    # Enforce: no evidence → not_enough_information
    claim_status = clamp(
        vlm.get("claim_status", "not_enough_information"),
        ALLOWED["claim_status"],
        "not_enough_information",
    )
    if not evidence_met:
        claim_status = "not_enough_information"

    raw_issue = str(vlm.get("issue_type", "unknown")).strip().lower()
    raw_issue = ISSUE_TYPE_SYNONYMS.get(raw_issue, raw_issue)
    issue_type = clamp(raw_issue, ALLOWED["issue_type"], "unknown")

    raw_part = str(vlm.get("object_part", "unknown")).strip().lower()
    raw_part = OBJECT_PART_SYNONYMS.get(raw_part, raw_part)
    object_part = clamp(raw_part, get_part_set(claim_object), "unknown")

    severity = clamp(vlm.get("severity", "unknown"), ALLOWED["severity"], "unknown")

    # Validate supporting image IDs against submitted images
    raw_sup = vlm.get("supporting_image_ids", "none")
    if raw_sup and raw_sup.strip().lower() != "none":
        valid_sup = [s.strip() for s in raw_sup.split(";") if s.strip() in all_ids]
        supporting = ";".join(valid_sup) if valid_sup else "none"
    else:
        supporting = "none"

    vlm_flags = list(vlm.get("risk_flags_from_images", []))
    # GT consistently flags damage_not_visible when a claim is contradicted with no visible damage.
    if claim_status == "contradicted" and issue_type == "none":
        vlm_flags.append("damage_not_visible")

    risk_flags = merge_risk_flags(
        vlm_flags,
        user_history,
        bool(vlm.get("text_instruction_present", False)),
    )

    return {
        "user_id": claim["user_id"],
        "image_paths": image_paths_str,
        "user_claim": claim["user_claim"],
        "claim_object": claim_object,
        "evidence_standard_met": str(evidence_met).lower(),
        "evidence_standard_met_reason": vlm.get("evidence_standard_met_reason", ""),
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": vlm.get("claim_status_justification", ""),
        "supporting_image_ids": supporting,
        "valid_image": str(valid_image).lower(),
        "severity": severity,
    }


# ── main pipeline ─────────────────────────────────────────────────────────────

def process_claims(
    input_path: Path,
    output_path: Path,
    history_path: Path,
    requirements_path: Path,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> list[dict]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY not set. Add it to your .env file.")

    client = OpenAI(api_key=api_key)
    claims = load_csv(input_path)
    history_map = load_user_history(history_path)
    requirements = load_evidence_requirements(requirements_path)

    results = []

    for i, claim in enumerate(claims):
        user_id = claim["user_id"]
        claim_object = claim["claim_object"]
        image_paths_str = claim["image_paths"]

        if verbose:
            print(f"[{i+1}/{len(claims)}] {user_id} | {claim_object} | {image_paths_str}")

        user_history = history_map.get(user_id, {
            "past_claim_count": 0, "accept_claim": 0, "rejected_claim": 0,
            "last_90_days_claim_count": 0, "history_flags": "none",
            "history_summary": "No prior history.",
        })

        images = load_images_b64(image_paths_str)

        if not images:
            results.append({
                "user_id": user_id,
                "image_paths": image_paths_str,
                "user_claim": claim["user_claim"],
                "claim_object": claim_object,
                "evidence_standard_met": "false",
                "evidence_standard_met_reason": "No images could be loaded.",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": "No images were available for review.",
                "supporting_image_ids": "none",
                "valid_image": "false",
                "severity": "unknown",
            })
            continue

        image_ids = [img["id"] for img in images]
        evidence_req = get_evidence_text(requirements, claim_object)
        prompt = build_prompt(
            claim_object=claim_object,
            user_claim=claim["user_claim"],
            image_ids=image_ids,
            evidence_req=evidence_req,
            user_history=user_history,
        )

        try:
            vlm_output = call_vlm(client, prompt, images, model=model)
        except Exception as e:
            print(f"  VLM error: {e}", file=sys.stderr)
            vlm_output = {
                "valid_image": False,
                "evidence_standard_met": False,
                "evidence_standard_met_reason": f"VLM call failed: {e}",
                "claim_status": "not_enough_information",
                "claim_status_justification": "Automated review could not be completed.",
                "issue_type": "unknown",
                "object_part": "unknown",
                "severity": "unknown",
                "supporting_image_ids": "none",
                "risk_flags_from_images": ["manual_review_required"],
                "text_instruction_present": False,
            }

        row = build_output_row(claim, vlm_output, user_history)
        results.append(row)

        if verbose:
            print(f"  -> {row['claim_status']} | {row['issue_type']} | {row['object_part']} | sev={row['severity']} | flags={row['risk_flags']}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows -> {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Damage Claim Verification System")
    parser.add_argument("--input", default=str(DATASET_DIR / "claims.csv"))
    parser.add_argument("--output", default=str(REPO_ROOT / "output.csv"))
    parser.add_argument("--history", default=str(DATASET_DIR / "user_history.csv"))
    parser.add_argument("--requirements", default=str(DATASET_DIR / "evidence_requirements.csv"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    process_claims(
        input_path=Path(args.input),
        output_path=Path(args.output),
        history_path=Path(args.history),
        requirements_path=Path(args.requirements),
        model=args.model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
