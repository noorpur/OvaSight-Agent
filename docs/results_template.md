# Results Template

Do not fill this page until the model has been run on real MMOTU data.

## Dataset

- Number of paired cases:
- Train/validation/test split:
- Image size:
- Excluded cases and reason:

## Training

- Best validation Dice:
- Epoch selected by early stopping:
- Device:

## Threshold tuning

- Selected threshold:
- Validation Dice at selected threshold:
- Validation IoU at selected threshold:

## Test results

| Metric | Mean | Median | P10 | P90 |
|---|---:|---:|---:|---:|
| Dice |  |  |  |  |
| IoU |  |  |  |  |
| Sensitivity |  |  |  |  |
| Specificity |  |  |  |  |
| Precision |  |  |  |  |
| Area error |  |  |  |  |

## Qualitative analysis

Discuss best, median, and worst cases. Include screenshots from `figures/qualitative_grid.png`.

## Clinical-readiness interpretation

State whether the model output quality is strong enough to justify expert review. Do not claim clinical validation.
