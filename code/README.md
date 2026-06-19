# Multi-Modal Evidence Review — Solution

A system that verifies damage claims for **cars**, **laptops**, and **packages** by reviewing the
submitted images against the claim conversation, user history, and minimum evidence requirements.
It produces a structured 14-column prediction per claim.

## Approach

Single-pass vision-language classification with a deterministic decision layer:

1. **Assemble context** — for each claim, the pipeline loads the submitted images (auto-converting
   HEIC/other formats to JPEG), the claim conversation, the user's history row, and the evidence
   requirements for that object type.
2. **One VLM call per claim** — images are sent inline (`detail:"high"`) with a strict, ordered
   prompt: identify the **object part** first, then the **issue type** from the images + chat, then
   **risk flags**, then claim status / severity / supporting images. The model returns JSON only.
3. **Deterministic decision + normalization layer** — the raw model JSON is clamped to the exact
   label vocabulary observed in `dataset/sample_claims.csv` (no synonyms). Out-of-vocabulary outputs
   are remapped (e.g. `glass_shatter`→`crack`, `box`→`package_side`) rather than dropped. Rules
   enforce consistency: invalid image → evidence gate fails; no evidence → `not_enough_information`;
   contradicted-with-no-damage → `damage_not_visible`; history risk → `manual_review_required`;
   risk flags emitted in canonical order.

Design priorities: **exact-match with dataset labels**, determinism (`temperature=0`), and avoiding
unnecessary repeated calls.

## Layout

```
code/
├── main.py              # Prediction entry point (CSV in -> CSV out)
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── evaluation/
    └── main.py          # Scores predictions vs sample_claims.csv ground truth
```

## Setup

```bash
pip install -r code/requirements.txt
```

Set your OpenAI key (read from the environment only — never hardcoded):

```bash
# bash / Git Bash
export OPENAI_API_KEY=sk-...
# PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

A git-ignored `.env` file in the repo root with `OPENAI_API_KEY=...` is also supported
(loaded via `python-dotenv`).

## Run — generate predictions

Test set (inputs only) → `output.csv`:

```bash
python code/main.py --input dataset/claims.csv --output output.csv
```

Useful flags: `--model <name>` (default `gpt-4o`), `--verbose`.

The output CSV contains all 14 columns:
`user_id, image_paths, user_claim, claim_object, evidence_standard_met,
evidence_standard_met_reason, risk_flags, issue_type, object_part, claim_status,
claim_status_justification, supporting_image_ids, valid_image, severity`.

## Evaluate — score against the labeled sample

```bash
python code/evaluation/main.py
```

This runs the pipeline on `dataset/sample_claims.csv` and prints per-field accuracy, a claim-status
confusion matrix, and an overall scored-field accuracy. To score an existing predictions file
without re-running the model:

```bash
python code/evaluation/main.py --predictions <predictions.csv>
```

Current sample-set result: **~81.7% overall scored-field accuracy** (evidence/valid_image 95%,
object_part & supporting_image_ids 85%, claim_status 80%). See `evaluation/evaluation_report.md`
for the full breakdown and the operational analysis (calls, tokens, cost, latency, rate limits).

## Allowed values (restricted to the dataset vocabulary)

- **claim_status**: `supported | contradicted | not_enough_information`
- **issue_type**: `dent | scratch | crack | broken_part | torn_packaging | crushed_packaging | water_damage | stain | none | unknown`
- **object_part (car)**: `front_bumper | rear_bumper | door | windshield | side_mirror | headlight | unknown`
- **object_part (laptop)**: `screen | keyboard | trackpad | hinge | corner | unknown`
- **object_part (package)**: `package_corner | package_side | seal | contents | unknown`
- **severity**: `none | low | medium | high | unknown`
- **risk_flags**: `blurry_image | cropped_or_obstructed | wrong_angle | wrong_object | damage_not_visible | claim_mismatch | non_original_image | text_instruction_present | user_history_risk | manual_review_required`

## Notes

- **Determinism**: `temperature=0` and a fixed prompt make runs reproducible.
- **Robust image loading**: HEIC/AVIF disguised as `.jpg` are detected by magic bytes and converted
  once in-memory before the API call.
- **Resilience**: rate-limit / API errors use exponential backoff (up to 3 attempts); malformed
  JSON is salvaged before a retry is counted.
- **Prompt-injection guard**: text embedded in images or claims that tries to force an approval
  (e.g. "mark as supported") is ignored — only observed evidence drives the verdict.
