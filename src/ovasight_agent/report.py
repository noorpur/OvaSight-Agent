from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _fmt_metric(summary: dict, metric: str) -> str:
    x = summary.get(metric, {})
    if not isinstance(x, dict) or x.get("mean") is None:
        return "not available"
    return f"mean={x['mean']:.3f}, median={x['median']:.3f}, p10={x['p10']:.3f}, p90={x['p90']:.3f}"


def generate_report(cfg: dict) -> Path:
    results_dir = Path(cfg["outputs"]["results_dir"])
    reports_dir = Path(cfg["outputs"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "test_metrics.json"
    audit_path = results_dir / "data_audit.json"
    review_path = results_dir / "review_queue.csv"
    threshold_path = results_dir / "best_threshold.json"

    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    threshold = json.loads(threshold_path.read_text()) if threshold_path.exists() else {"threshold": 0.5}
    review_n = len(pd.read_csv(review_path)) if review_path.exists() else 0

    ready_flags = []
    dice_mean = metrics.get("dice", {}).get("mean") if isinstance(metrics.get("dice"), dict) else None
    empty_rate = metrics.get("empty_prediction_rate")
    if dice_mean is not None and dice_mean >= 0.75:
        ready_flags.append("Mean Dice is in a range that may justify expert review, pending distribution-level checks.")
    else:
        ready_flags.append("Mean Dice does not yet support moving directly to expert-facing validation without model improvement.")
    if empty_rate is not None and empty_rate <= cfg["qc"]["max_empty_prediction_rate"]:
        ready_flags.append("Empty-prediction rate is within the configured tolerance.")
    else:
        ready_flags.append("Empty-prediction rate needs attention before clinician review.")

    text = f"""# OvaSight-Agent Clinical-Readiness Report

I generated this report automatically after running the OvaSight pipeline on the real dataset available in `data/raw/MMOTU/`.

## Dataset audit

- Raw data root: `{audit.get('raw_root', 'not recorded')}`
- Paired image/mask cases found: **{audit.get('paired_cases', 'not recorded')}**
- Valid cases: **{audit.get('valid_cases', 'not recorded')}**
- Blank/near-blank masks flagged: **{audit.get('blank_masks', 'not recorded')}**

## Tuned decision threshold

The validation sweep selected a segmentation probability threshold of **{float(threshold.get('threshold', 0.5)):.2f}**. I tune this threshold on validation data because medical segmentation quality can change substantially when a fixed probability cutoff is used without calibration.

## Test-set segmentation metrics

- Dice: {_fmt_metric(metrics, 'dice')}
- IoU: {_fmt_metric(metrics, 'iou')}
- Sensitivity: {_fmt_metric(metrics, 'sensitivity')}
- Specificity: {_fmt_metric(metrics, 'specificity')}
- Precision: {_fmt_metric(metrics, 'precision')}
- Lesion-area relative error: {_fmt_metric(metrics, 'area_error')}
- Boundary-distance proxy: {_fmt_metric(metrics, 'boundary_distance_proxy')}
- Empty prediction rate: {metrics.get('empty_prediction_rate', 'not available')}

## Review queue

The agent flagged **{review_n}** cases for human review. Cases are sent to the review queue if they have low Dice, low predicted-mask confidence, or empty predictions.

## Interpretation

""" + "\n".join([f"- {x}" for x in ready_flags]) + """

This report should be read as a technical readiness screen, not a clinical validation claim. The next research step would be a blinded review with qualified gynecology/radiology experts using held-out data and predefined acceptance criteria.

## Figures to inspect

- `figures/training_curves.png`
- `figures/threshold_sweep.png`
- `figures/qualitative_grid.png`

## Limitations

- Performance depends on the exact MMOTU subset and split used.
- Public datasets may not represent local scanner distributions or patient populations.
- Segmentation masks are not a diagnosis.
- Before clinical use, the model would need expert review, external validation, bias analysis, and regulatory consideration.
"""
    out = reports_dir / "clinical_readiness_report.md"
    out.write_text(text, encoding="utf-8")
    return out
