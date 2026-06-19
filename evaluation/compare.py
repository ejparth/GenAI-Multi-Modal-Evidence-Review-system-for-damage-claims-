"""
Generates a side-by-side HTML comparison of ground truth vs fresh predictions.
Usage: python evaluation/compare.py
Output: evaluation/comparison_report.html
"""

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
GT_PATH   = REPO_ROOT / "dataset" / "sample_claims.csv"
PRED_PATH = REPO_ROOT / "evaluation" / "sample_predictions_fresh.csv"
OUT_PATH  = REPO_ROOT / "evaluation" / "comparison_report.html"

COMPARE_FIELDS = [
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

STRICT_FIELDS = {
    "evidence_standard_met", "valid_image", "claim_status",
    "issue_type", "object_part", "severity",
}

SET_FIELDS = {"risk_flags", "supporting_image_ids"}


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize(val):
    return val.strip().lower() if val else ""


def parse_set(val):
    v = normalize(val)
    if not v or v == "none":
        return set()
    return {x.strip() for x in v.split(";")}


def is_match(field, gt_val, pred_val):
    if field in SET_FIELDS:
        return parse_set(gt_val) == parse_set(pred_val)
    return normalize(gt_val) == normalize(pred_val)


def cell(gt_val, pred_val, field, scored):
    match = is_match(field, gt_val, pred_val) if scored else None
    if not scored:
        bg = "#f8f9fa"
        icon = ""
    elif match:
        bg = "#d4edda"
        icon = "&#10003; "
    else:
        bg = "#f8d7da"
        icon = "&#10007; "

    return (
        f'<td style="background:{bg};padding:6px 10px;vertical-align:top;font-size:12px">'
        f'<div><strong style="font-size:10px;color:#666">EXPECTED</strong><br>'
        f'<span>{icon}{_esc(gt_val)}</span></div>'
        f'<hr style="margin:4px 0;border-color:#ccc">'
        f'<div><strong style="font-size:10px;color:#666">PREDICTED</strong><br>'
        f'<span>{_esc(pred_val)}</span></div>'
        f'</td>'
    )


def _esc(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_html(gt_rows, pred_rows):
    total_scored = 0
    total_correct = 0

    rows_html = []
    for i, (gt, pred) in enumerate(zip(gt_rows, pred_rows)):
        row_correct = 0
        row_total = 0
        cells = []

        # Identity columns (unscored, grey)
        for col in ["user_id", "claim_object"]:
            cells.append(
                f'<td style="background:#e9ecef;padding:6px 10px;vertical-align:middle;font-size:12px">'
                f'{_esc(gt[col])}</td>'
            )
        # Claim summary (unscored)
        claim_preview = _esc(gt["user_claim"][:120]) + "..."
        cells.append(
            f'<td style="background:#e9ecef;padding:6px 10px;vertical-align:middle;font-size:11px;max-width:200px">'
            f'{claim_preview}</td>'
        )

        for field in COMPARE_FIELDS:
            gt_val   = gt.get(field, "")
            pred_val = pred.get(field, "")
            scored   = field in STRICT_FIELDS or field in SET_FIELDS
            if scored:
                m = is_match(field, gt_val, pred_val)
                row_correct += int(m)
                row_total   += 1
                total_correct += int(m)
                total_scored  += 1
            cells.append(cell(gt_val, pred_val, field, scored))

        pct = int(row_correct / row_total * 100) if row_total else 0
        row_bg = "#ffffff" if i % 2 == 0 else "#f9f9f9"
        score_color = "#28a745" if pct >= 80 else ("#fd7e14" if pct >= 50 else "#dc3545")
        score_td = (
            f'<td style="background:{row_bg};padding:6px;text-align:center;vertical-align:middle">'
            f'<span style="font-weight:bold;color:{score_color}">{pct}%</span><br>'
            f'<small style="color:#666">{row_correct}/{row_total}</small></td>'
        )
        rows_html.append(
            f'<tr style="background:{row_bg}">'
            f'<td style="background:{row_bg};padding:6px;text-align:center;vertical-align:middle;font-weight:bold">{i+1}</td>'
            + "".join(cells) + score_td + "</tr>"
        )

    overall_pct = round(total_correct / total_scored * 100, 1) if total_scored else 0

    header_cols = (
        ["#", "user_id", "claim_object", "claim (preview)"]
        + COMPARE_FIELDS
        + ["score"]
    )
    header_html = "".join(
        f'<th style="background:#343a40;color:#fff;padding:8px;white-space:nowrap;font-size:12px">{c}</th>'
        for c in header_cols
    )

    legend = (
        '<div style="margin-bottom:16px;font-family:sans-serif">'
        '<span style="background:#d4edda;padding:4px 10px;border-radius:4px;margin-right:8px">&#10003; Match</span>'
        '<span style="background:#f8d7da;padding:4px 10px;border-radius:4px;margin-right:8px">&#10007; Mismatch</span>'
        '<span style="background:#e9ecef;padding:4px 10px;border-radius:4px">Not scored</span>'
        '</div>'
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sample Predictions Comparison</title>
<style>
  body {{ font-family: sans-serif; padding: 24px; background: #f5f5f5; }}
  h1 {{ font-size: 20px; margin-bottom: 4px; }}
  .summary {{ background:#fff; border-radius:8px; padding:16px; margin-bottom:20px;
              display:inline-block; box-shadow:0 1px 4px rgba(0,0,0,.1); }}
  table {{ border-collapse: collapse; width: 100%; background: #fff;
           box-shadow: 0 1px 4px rgba(0,0,0,.1); border-radius: 8px; overflow: hidden; }}
  td, th {{ border: 1px solid #dee2e6; }}
  th {{ position: sticky; top: 0; z-index: 1; }}
  .wrap {{ overflow-x: auto; }}
</style>
</head>
<body>
<h1>Sample Claims — Ground Truth vs Fresh Predictions</h1>
<div class="summary">
  <strong>Overall scored-field accuracy: </strong>
  <span style="font-size:20px;font-weight:bold;color:{'#28a745' if overall_pct>=80 else '#fd7e14' if overall_pct>=60 else '#dc3545'}">{overall_pct}%</span>
  &nbsp;({total_correct}/{total_scored} fields correct across all 20 rows)
</div>
<br>
{legend}
<div class="wrap">
<table>
  <thead><tr>{header_html}</tr></thead>
  <tbody>{"".join(rows_html)}</tbody>
</table>
</div>
</body>
</html>"""


if __name__ == "__main__":
    gt_rows   = load_csv(GT_PATH)
    pred_rows = load_csv(PRED_PATH)
    html = build_html(gt_rows, pred_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report written -> {OUT_PATH}")
