# OvaSight-Agent Clinical-Readiness Report

I generated this report automatically after running the OvaSight pipeline on the real dataset available in `data/raw/MMOTU/`.

## Dataset audit

- Raw data root: `data/raw/MMOTU`
- Paired image/mask cases found: **1469**
- Valid cases: **1469**
- Blank/near-blank masks flagged: **0**

## Tuned decision threshold

The validation sweep selected a segmentation probability threshold of **0.40**. I tune this threshold on validation data because medical segmentation quality can change substantially when a fixed probability cutoff is used without calibration.

## Test-set segmentation metrics

- Dice: mean=0.737, median=0.827, p10=0.383, p90=0.953
- IoU: mean=0.629, median=0.704, p10=0.236, p90=0.909
- Sensitivity: mean=0.783, median=0.871, p10=0.413, p90=0.993
- Specificity: mean=0.968, median=0.983, p10=0.925, p90=0.997
- Precision: mean=0.767, median=0.873, p10=0.364, p90=0.985
- Lesion-area relative error: mean=0.472, median=0.210, p10=0.037, p90=1.000
- Boundary-distance proxy: mean=11.599, median=8.782, p10=2.747, p90=23.509
- Empty prediction rate: 0.004524886877828055

## Review queue

The agent flagged **40** cases for human review. Cases are sent to the review queue if they have low Dice, low predicted-mask confidence, or empty predictions.

## Interpretation

- Mean Dice does not yet support moving directly to expert-facing validation without model improvement.
- Empty-prediction rate is within the configured tolerance.

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
