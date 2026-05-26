# OvaSight-Agent Clinical-Readiness Report

I generated this report automatically after running the OvaSight pipeline on the real dataset available in `data/raw/MMOTU/`.

## Dataset audit

- Raw data root: `data/raw/MMOTU`
- Paired image/mask cases found: **1469**
- Valid cases: **1469**
- Blank/near-blank masks flagged: **0**

## Tuned decision threshold

The validation sweep selected a segmentation probability threshold of **0.55**. I tune this threshold on validation data because medical segmentation quality can change substantially when a fixed probability cutoff is used without calibration.

## Test-set segmentation metrics

- Dice: mean=0.747, median=0.847, p10=0.372, p90=0.950
- IoU: mean=0.639, median=0.735, p10=0.229, p90=0.904
- Sensitivity: mean=0.775, median=0.859, p10=0.402, p90=0.985
- Specificity: mean=0.972, median=0.984, p10=0.940, p90=0.997
- Precision: mean=0.779, median=0.870, p10=0.392, p90=0.983
- Lesion-area relative error: mean=0.384, median=0.188, p10=0.031, p90=0.821
- Boundary-distance proxy: mean=14.533, median=11.478, p10=3.909, p90=28.079
- Empty prediction rate: 0.0

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
