# Visual Evidence Verification for Damage Claims

Marketplaces, insurers, logistics teams, and support operations receive damage claims every day. A user may say a car bumper is dented, a laptop screen is cracked, or a package seal is torn. Sometimes the photos support the claim. Sometimes they show the wrong object, the wrong part, a blurry view, a screenshot, an exaggerated issue, or text trying to influence the review.

Your task is to build a system that verifies damage claims using:

- submitted images
- a short support-style claim conversation
- user claim history
- evidence standards

The images are the primary source of visual truth. The conversation tells you what must be verified. User history adds risk context, but should not override clear visual evidence by itself.

---

## What You Need To Build

For each row in `dataset/test.csv`, predict whether the submitted image set supports the user's claim.

Your system should determine:

- whether the image evidence is sufficient
- what visible issue is present
- which object part is involved
- whether the claim is `supported`, `contradicted`, or `not_enough_information`
- which image IDs support the decision
- whether the image set is valid for automated review
- severity
- risk flags
- short justifications

You must also build an evaluation pipeline using the labeled examples in `dataset/sample.csv`.

---

## Files Provided


| File or Folder                      | Description                                                    |
| ----------------------------------- | -------------------------------------------------------------- |
| `dataset/sample.csv`                | Labeled examples with inputs and expected outputs.             |
| `dataset/test.csv`                  | Input-only rows for final prediction.                          |
| `dataset/user_history.csv`          | Historical claim counts and risk patterns for each user.       |
| `dataset/evidence_requirements.csv` | Minimum image evidence checklist by object and issue family.    |
| `dataset/images/sample/`            | Images referenced by `sample.csv`.                             |
| `dataset/images/test/`              | Images referenced by `test.csv`.                               |


Multiple image paths are separated by semicolons:

```text
images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg
```

The image ID is the filename without extension, such as `img_1`.

`evidence_requirements.csv` contains four columns:

| Column | Meaning |
|---|---|
| `requirement_id` | Identifier for the evidence rule. |
| `claim_object` | Object category the rule applies to: `car`, `laptop`, `package`, or `all`. |
| `applies_to` | Human-readable issue family, such as `dent or scratch` or `vehicle identity or orientation`. |
| `minimum_image_evidence` | Minimum visual evidence needed to evaluate that kind of claim. |

---

## Input Columns

`test.csv` contains:


| Column         | Meaning                                                            |
| -------------- | ------------------------------------------------------------------ |
| `user_id`      | Use this to look up history in `user_history.csv`.                 |
| `image_paths`  | One or more submitted image paths.                                 |
| `user_claim`   | A support-style chat transcript. Extract the actual claim from it. |
| `claim_object` | One of `car`, `laptop`, or `package`.                              |


The `user_claim` may include uncertainty, irrelevant details, multiple turns, or imprecise wording. Extract the actual damage claim and verify it against the images.

---

## Output Format

Submit an `output.csv` with one row per row in `test.csv`.

Use exactly these columns, in this order:

```text
user_id
image_paths
user_claim
claim_object
evidence_standard_met
evidence_standard_met_reason
risk_flags
issue_type
object_part
claim_status
claim_status_justification
supporting_image_ids
valid_image
severity
```

### Output Fields


| Field                          | Expected Value                                                                                           |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `user_id`                      | Same as input.                                                                                           |
| `image_paths`                  | Same as input.                                                                                           |
| `user_claim`                   | Same as input.                                                                                           |
| `claim_object`                 | Same as input.                                                                                           |
| `evidence_standard_met`        | `true` if the images provide enough evidence to evaluate the claim; otherwise `false`.                   |
| `evidence_standard_met_reason` | Short reason for the evidence decision.                                                                  |
| `risk_flags`                   | Semicolon-separated risk flags, or `none`.                                                               |
| `issue_type`                   | Visible issue type, such as `dent`, `scratch`, `crack`, `torn_packaging`, `stain`, `none`, or `unknown`. |
| `object_part`                  | Claimed/relevant part, such as `front_bumper`, `screen`, `keyboard`, `seal`, or `label`.                 |
| `claim_status`                 | One of `supported`, `contradicted`, `not_enough_information`.                                            |
| `claim_status_justification`   | Short image-grounded explanation. Mention relevant image IDs when helpful, especially for multi-image rows. Mention user history only when relevant. |
| `supporting_image_ids`         | Image IDs that support the decision, separated by semicolons. Use `none` if no image supports it.        |
| `valid_image`                  | `true` if the image set is usable/trustworthy enough for automated review; otherwise `false`.            |
| `severity`                     | One of `none`, `low`, `medium`, `high`, `unknown`.                                                       |


---

## Allowed Values

Use the closest matching value from these compact lists.

| Field | Values |
|---|---|
| `issue_type` | `dent`, `scratch`, `crack`, `glass_shatter`, `broken_part`, `missing_part`, `torn_packaging`, `crushed_packaging`, `water_damage`, `stain`, `none`, `unknown` |
| Car `object_part` | `front_bumper`, `rear_bumper`, `door`, `hood`, `windshield`, `side_mirror`, `headlight`, `taillight`, `fender`, `quarter_panel`, `body`, `unknown` |
| Laptop `object_part` | `screen`, `keyboard`, `trackpad`, `hinge`, `lid`, `corner`, `port`, `base`, `body`, `unknown` |
| Package `object_part` | `box`, `package_corner`, `package_side`, `seal`, `label`, `contents`, `item`, `unknown` |

Use `issue_type=none` when the relevant part is visible and no issue is present. Use `unknown` when the issue or part cannot be determined.

---

## Key Decisions

### Claim Status


| Status                   | Use When                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------- |
| `supported`              | The images support the specific claim.                                                                  |
| `contradicted`           | The relevant object/part is visible, but the claimed issue, side, object, or severity is not supported. |
| `not_enough_information` | The images are insufficient to support or contradict the claim.                                         |


Examples:


| Claim                 | Image Evidence                            | Status                   |
| --------------------- | ----------------------------------------- | ------------------------ |
| Rear bumper is dented | Rear bumper dent is visible               | `supported`              |
| Screen is shattered   | Only a small scratch is visible           | `contradicted`           |
| Headlight is cracked  | Headlight is not visible                  | `not_enough_information` |
| Left door is dented   | Right door is dented, left door is intact | `contradicted`           |
| Product is missing    | Only a closed package is visible          | `not_enough_information` |


### Evidence Standard

`evidence_standard_met=true` means the image set is sufficient to evaluate the claim. It does not mean the claim is supported.

Use `dataset/evidence_requirements.csv` as a minimum-evidence checklist. It tells you what must be visible for broad claim types, but your system must still decide the final status from the images and claim.

Example:

```text
Claim: Box is badly crushed
Image: Box is visible, but only a small crease is present

evidence_standard_met=true
claim_status=contradicted
severity=low
risk_flags=claim_mismatch
```

Use `evidence_standard_met=false` when the needed evidence is missing, cropped, blurred, obstructed, shown from the wrong angle, non-original, or not trustworthy.

### Valid Image

Use `valid_image=false` when the image set is not suitable for automated visual review, for example:

- wrong object
- unusably blurry/cropped/dark/obstructed
- screenshot, listing image, or other non-original source
- likely AI-generated or edited evidence
- text-only evidence instead of visual damage

---

## Risk Flags

Use semicolon-separated flags. Use `none` if no risk applies.

Common flags include:

```text
none
blurry_image
cropped_or_obstructed
low_light_or_glare
wrong_angle
wrong_object
wrong_object_part
damage_not_visible
claim_mismatch
possible_manipulation
non_original_image
text_instruction_present
user_history_risk
manual_review_required
```

Risk flags can come from image quality, image authenticity, claim-image mismatch, or user history.

---

## User History

Use `dataset/user_history.csv` to add risk context.


| Column                     | Meaning                                        |
| -------------------------- | ---------------------------------------------- |
| `past_claim_count`         | Total prior claims.                            |
| `accept_claim`             | Prior accepted claims.                         |
| `manual_review_claim`      | Prior manual-review claims.                    |
| `rejected_claim`           | Prior rejected claims.                         |
| `last_90_days_claim_count` | Recent claim volume.                           |
| `history_flags`            | Known historical risk patterns.                |
| `history_summary`          | Short description of the user's claim history. |


History should usually affect `risk_flags` and justifications, not the visual truth.

Example:

```text
Image clearly shows a car door dent.
User has many rejected claims.

claim_status=supported
risk_flags=user_history_risk;manual_review_required
```

---

## Evaluation Requirement

Your code must include an evaluation pipeline that gives the accuracy/confidence on the system. You can use the sample.csv to verify the accuracy.

---

## Operational Analysis

Include a short operational analysis in `evaluation/evaluation_report.md`.

| Item | What To Report |
|---|---|
| Model calls | Approximate number of model calls for `sample.csv` and `test.csv`. |
| Token usage | Approximate input/output tokens used. |
| Image usage | Number of images processed and how images are processed. |
| Cost estimate | Approximate cost to process the full test set. State the pricing assumptions. |
| Latency | Approximate runtime per row sample and test processing. |
| TPM/RPM considerations | Whether your approach may hit token-per-minute or request-per-minute limits. |

You are not expected to optimize perfectly, but your solution should show that you considered cost, latency, rate limits, and unnecessary repeated calls.

---

## Submission

Submit:


| File              | Description                                                           |
| ----------------- | --------------------------------------------------------------------- |
| `code.zip`        | Full runnable solution, prompts/configs, README, and `evaluation/` folder. |
| `output.csv`      | Predictions for all rows in `dataset/test.csv`.                       |
| `chat_transcript` | Conversation transcript showing how you developed or used the system. |

---
