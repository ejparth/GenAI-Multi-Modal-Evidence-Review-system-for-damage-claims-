# Evaluation Report — Multi-Modal Evidence Review

Model: `gpt-4o` (OpenAI vision), single-pass, `temperature=0`, JSON response format.

## 1. Accuracy on the labeled sample set (`dataset/sample_claims.csv`, 20 claims)

Predictions are generated independently from columns A–D only (user_id, image_paths,
user_claim, claim_object) and scored against the held ground-truth labels.

| Field | Accuracy | Correct/Total |
|---|---|---|
| evidence_standard_met | 95.0% | 19/20 |
| valid_image | 95.0% | 19/20 |
| claim_status | 80.0% | 16/20 |
| object_part | 85.0% | 17/20 |
| supporting_image_ids | 85.0% | 17/20 |
| severity | 70.0% | 14/20 |
| issue_type | 65.0% | 13/20 |
| risk_flags | 60.0% | 12/20 |
| **Overall (scored fields)** | **81.7%** | **98/120** |

Claim-status confusion (GT → prediction):
- supported → supported: 12
- not_enough_information → not_enough_information: 3
- contradicted → contradicted: 1
- contradicted → not_enough_information: 1
- contradicted → supported: 3

The remaining errors are vision-judgment calls on borderline-damage cases that also carry a
fraud signal (`user_history_risk` / `text_instruction_present`), where the model trusts faint
visible damage that the dataset labels as contradicted. The label vocabulary itself is now fully
constrained to the strings observed in `sample_claims.csv` (no synonyms), which removed an entire
class of exact-match misses (e.g. `glass_shatter`→`crack`, `box`→`package_side`).

## 2. Test set predictions (`dataset/claims.csv`, 44 claims)

Final predictions written to `output.csv` (all 14 columns). Distribution of the 44 predictions:

- claim_status: supported 17, contradicted 3, not_enough_information 24
- claim_object mix: car 18, laptop 13, package 13

No ground truth is provided for the test set, so accuracy cannot be scored here; the sample-set
metrics above are the proxy for expected quality.

## 3. Operational Analysis

### Model calls
- **1 model call per claim** (single-pass design — no separate extraction/verification passes).
- Sample set: **20 calls**. Test set: **44 calls**.
- Retries are only triggered on a transient error (rate limit / API error / malformed JSON),
  with up to 3 attempts per claim. No retries were needed in this run, so call count equals
  claim count: **64 calls total** across sample + test.

### Images processed
- Sample: 20 claims / **29 images**.
- Test: 44 claims / **82 images**.
- Total: **111 images**.
- 8 test images arrived as HEIC disguised as `.jpg`; these are detected by magic-byte inspection
  and transcoded to JPEG once, in-process, before the API call (no repeated conversion).

### Token usage (approximate, per call)
- Prompt text (system + decision rules + per-claim claim/history/requirements): **~2,400 input tokens**.
- Images: sent inline at `detail:"high"`. A typical claim photo costs **~765–1,100 tokens**;
  estimate **~1,000 tokens/image**.
- Output JSON: capped at `max_tokens=1024`; actual responses are **~300 tokens**.

| Set | Calls | Input text | Image tokens | Output tokens | Total input | Total output |
|---|---|---|---|---|---|---|
| Sample (20 claims, 29 imgs) | 20 | ~48k | ~29k | ~6k | **~77k** | **~6k** |
| Test (44 claims, 82 imgs) | 44 | ~106k | ~82k | ~13k | **~188k** | **~13k** |

### Approximate cost (full test set)
Pricing assumption — `gpt-4o`: **$2.50 / 1M input tokens**, **$10.00 / 1M output tokens**.
- Test input: ~188k × $2.50/1M ≈ **$0.47**
- Test output: ~13k × $10/1M ≈ **$0.13**
- **Test set total ≈ $0.60.**
- Sample set ≈ $0.25. One full dev + test cycle ≈ **$0.85**.

Cost scales linearly with image count, since images dominate the input token budget. A larger
test set of ~1,000 claims at the same ~1.9 images/claim would run **~$14**.

### Latency / runtime
- Measured test run: **308.8 s for 44 claims ≈ 7.0 s/claim** (wall clock, sequential).
- Sample set runs in roughly **140 s**.
- Latency is dominated by the vision call; multi-image claims (up to 3 images) are slightly slower.

### Rate limits (TPM/RPM), throttling, batching, caching, retries
- **Sequential processing.** Claims are processed one at a time. At ~7 s/claim this is ~8–9
  requests/min and ~25k tokens/min — comfortably under standard `gpt-4o` tier limits
  (typically thousands of RPM and hundreds of thousands of TPM), so no throttling was required
  at this scale.
- **Retry strategy.** `RateLimitError` and `APIError` are caught with **exponential backoff**
  (5·2^attempt s for rate limits, 2^attempt s for API errors), up to 3 attempts. Malformed JSON
  is salvaged with a regex `{...}` extraction before counting an attempt as failed.
- **Avoiding repeated work.** One call per claim — no redundant re-querying. HEIC→JPEG conversion
  happens once per image in-memory. `temperature=0` makes outputs deterministic, so a re-run does
  not change results and there is no need to sample a claim multiple times.
- **Batching / further optimization (considered, not implemented).** At this scale the run is
  cheap (<$1) and fast (~5 min), so added complexity is unwarranted. If the test set grew 10–100×,
  the natural levers would be: (a) concurrent requests with a bounded worker pool to cut wall-clock
  time, (b) a token-bucket limiter keyed to the account's published TPM/RPM, (c) an on-disk response
  cache keyed by `(claim_id, image hashes, prompt hash)` so re-runs and retries never re-bill
  identical inputs, and (d) dropping image `detail` to `"low"` for single-object close-ups to roughly
  halve image-token cost where full resolution is not needed.

### Reproducibility
- Deterministic decoding (`temperature=0`, fixed prompt).
- Secrets read only from the `OPENAI_API_KEY` environment variable (`.env`, git-ignored).
- Entry points unchanged: `python code/main.py --input <csv> --output <csv>` for prediction,
  `python code/evaluation/main.py` for scoring.
