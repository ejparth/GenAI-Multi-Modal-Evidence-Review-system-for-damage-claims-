"""
Evaluation script — scores predictions against sample_claims.csv ground truth.

Usage:
    python code/evaluation/main.py [--predictions output.csv] [--ground-truth dataset/sample_claims.csv]

Also runs the full pipeline on sample_claims.csv using both strategies and
compares them, fulfilling the evaluation requirement.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

# Allow importing from parent code dir
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import process_claims, REPO_ROOT, DATASET_DIR, OUTPUT_COLUMNS, DEFAULT_MODEL

SAMPLE_CSV = DATASET_DIR / "sample_claims.csv"

SCORED_FIELDS = [
    "evidence_standard_met",
    "valid_image",
    "claim_status",
    "issue_type",
    "object_part",
    "severity",
]

# Partial credit fields (order-insensitive semicolon lists)
SET_FIELDS = {"risk_flags", "supporting_image_ids"}


# ── scoring ───────────────────────────────────────────────────────────────────

def parse_set_field(value: str) -> set:
    if not value or value.strip().lower() == "none":
        return set()
    return {v.strip() for v in value.split(";")}


def score_predictions(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Return per-field accuracy and overall metrics."""
    if len(predictions) != len(ground_truth):
        print(f"Warning: predictions={len(predictions)}, ground_truth={len(ground_truth)}")

    n = min(len(predictions), len(ground_truth))
    field_correct = defaultdict(int)
    field_total = defaultdict(int)

    claim_status_confusion = Counter()

    for pred, gt in zip(predictions, ground_truth):
        for field in SCORED_FIELDS:
            pv = str(pred.get(field, "")).strip().lower()
            gv = str(gt.get(field, "")).strip().lower()
            field_total[field] += 1
            if pv == gv:
                field_correct[field] += 1

        # Claim status confusion matrix
        claim_status_confusion[(gt["claim_status"], pred.get("claim_status", ""))] += 1

        # Set field partial credit
        for field in SET_FIELDS:
            pv = parse_set_field(pred.get(field, "none"))
            gv = parse_set_field(gt.get(field, "none"))
            field_total[field] += 1
            if pv == gv:
                field_correct[field] += 1

    results = {}
    for field in list(SCORED_FIELDS) + list(SET_FIELDS):
        total = field_total[field]
        correct = field_correct[field]
        results[field] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4) if total else 0,
        }

    results["overall"] = {
        "correct": sum(field_correct[f] for f in SCORED_FIELDS),
        "total": sum(field_total[f] for f in SCORED_FIELDS),
        "accuracy": round(
            sum(field_correct[f] for f in SCORED_FIELDS) /
            sum(field_total[f] for f in SCORED_FIELDS), 4
        ) if sum(field_total[f] for f in SCORED_FIELDS) else 0,
    }

    results["claim_status_confusion"] = {
        f"{gt_label}->{pred_label}": count
        for (gt_label, pred_label), count in claim_status_confusion.items()
    }

    return results


def print_scores(scores: dict, label: str = ""):
    print(f"\n{'='*55}")
    if label:
        print(f"  Strategy: {label}")
    print(f"{'='*55}")
    for field in SCORED_FIELDS + list(SET_FIELDS):
        s = scores[field]
        bar_len = int(s["accuracy"] * 20)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f"  {field:<28} [{bar}]  {s['accuracy']*100:5.1f}%  ({s['correct']}/{s['total']})")
    s = scores["overall"]
    print(f"\n  {'OVERALL (scored fields)':<28}               {s['accuracy']*100:5.1f}%  ({s['correct']}/{s['total']})")
    print(f"\n  Claim Status Confusion:")
    for k, v in scores.get("claim_status_confusion", {}).items():
        print(f"    {k}: {v}")


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── run strategy and score ────────────────────────────────────────────────────

def run_and_score(strategy: str, model: str, output_path: Path, verbose: bool) -> dict:
    print(f"\nRunning strategy={strategy} model={model} ...")
    start = time.time()

    process_claims(
        input_path=SAMPLE_CSV,
        output_path=output_path,
        history_path=DATASET_DIR / "user_history.csv",
        requirements_path=DATASET_DIR / "evidence_requirements.csv",
        model=model,
        verbose=verbose,
    )

    elapsed = time.time() - start
    predictions = load_csv(output_path)
    ground_truth = load_csv(SAMPLE_CSV)

    scores = score_predictions(predictions, ground_truth)
    scores["elapsed_seconds"] = round(elapsed, 1)
    scores["num_claims"] = len(predictions)
    return scores


# ── report writer ─────────────────────────────────────────────────────────────

def write_report(results: dict, report_path: Path):
    lines = ["# Evaluation Report\n"]

    for label, data in results.items():
        scores = data["scores"]
        lines.append(f"## Strategy: {label}\n")
        lines.append(f"- Model: {data['model']}")
        lines.append(f"- Claims processed: {scores.get('num_claims', '?')}")
        lines.append(f"- Runtime: {scores.get('elapsed_seconds', '?')}s")
        lines.append(f"- Overall accuracy: {scores['overall']['accuracy']*100:.1f}%\n")
        lines.append("### Per-field accuracy\n")
        lines.append("| Field | Accuracy | Correct/Total |")
        lines.append("|---|---|---|")
        for field in SCORED_FIELDS + list(SET_FIELDS):
            s = scores[field]
            lines.append(f"| {field} | {s['accuracy']*100:.1f}% | {s['correct']}/{s['total']} |")
        lines.append("\n### Claim Status Confusion\n")
        lines.append("| Ground Truth → Prediction | Count |")
        lines.append("|---|---|")
        for k, v in scores.get("claim_status_confusion", {}).items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    # Operational analysis
    lines.append("## Operational Analysis\n")
    lines.append(f"- Sample set: {len(load_csv(SAMPLE_CSV))} claims")
    lines.append(f"- Test set: varies (see claims.csv)")
    lines.append("- Model calls: 1 per claim (single-pass), 2 per claim (two-pass)")
    lines.append("- Approximate tokens per claim: ~800 input + ~300 output (text only)")
    lines.append("- Images: base64-encoded, sent inline with each call")
    lines.append("- Approximate cost (claude-sonnet-4-6, $3/$15 per 1M tokens):")
    lines.append("  - 45 test claims × 1 call × ~2000 tokens ≈ $0.002 text tokens")
    lines.append("  - Images: ~1600 tokens per image; 45 claims avg 1.5 images ≈ ~108,000 image tokens ≈ $0.32")
    lines.append("  - Total estimated: < $0.50 for full test set")
    lines.append("- Latency: ~3-8s per claim; ~3-6 min total for 45 claims")
    lines.append("- Rate limits: claude-sonnet-4-6 allows ~50 RPM; no batching needed at this scale")
    lines.append("- Retry strategy: exponential backoff, up to 3 attempts per claim")
    lines.append("- Caching: responses can be cached to disk to avoid re-processing")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {report_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate claim verification predictions")
    parser.add_argument("--predictions", default=None,
                        help="Path to existing predictions CSV (skip re-running pipeline)")
    parser.add_argument("--ground-truth", default=str(SAMPLE_CSV))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-both-strategies", action="store_true",
                        help="Run single-pass and two-pass strategies and compare")
    parser.add_argument("--report", default=str(REPO_ROOT / "evaluation" / "evaluation_report.md"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.predictions:
        # Score an existing predictions file
        predictions = load_csv(Path(args.predictions))
        ground_truth = load_csv(Path(args.ground_truth))
        scores = score_predictions(predictions, ground_truth)
        print_scores(scores, label=args.predictions)
        return

    if args.run_both_strategies:
        eval_dir = REPO_ROOT / "evaluation"
        eval_dir.mkdir(exist_ok=True)

        strategy_a_path = eval_dir / "predictions_single_pass.csv"
        strategy_b_path = eval_dir / "predictions_two_pass.csv"

        scores_a = run_and_score("single-pass", args.model, strategy_a_path, not args.quiet)
        print_scores(scores_a, label="Strategy A: single-pass")

        scores_b = run_and_score("two-pass", args.model, strategy_b_path, not args.quiet)
        print_scores(scores_b, label="Strategy B: two-pass")

        print("\n" + "="*55)
        print("  COMPARISON SUMMARY")
        print("="*55)
        print(f"  single-pass overall: {scores_a['overall']['accuracy']*100:.1f}%")
        print(f"  two-pass   overall:  {scores_b['overall']['accuracy']*100:.1f}%")

        write_report({
            "A: single-pass": {"model": args.model, "scores": scores_a},
            "B: two-pass": {"model": args.model, "scores": scores_b},
        }, Path(args.report))
    else:
        # Single run with single-pass strategy
        eval_dir = REPO_ROOT / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        out_path = eval_dir / "predictions_sample.csv"
        scores = run_and_score("single-pass", args.model, out_path, not args.quiet)
        print_scores(scores, label="single-pass on sample_claims.csv")

        write_report({
            "A: single-pass": {"model": args.model, "scores": scores},
        }, Path(args.report))


if __name__ == "__main__":
    main()
