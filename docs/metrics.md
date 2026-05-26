# Metrics and Analysis Plan

## Dice coefficient

Dice measures overlap between predicted and ground-truth masks. It is widely used in medical segmentation because it handles foreground-background imbalance better than raw pixel accuracy.

## Intersection over Union

IoU is stricter than Dice and penalizes over-segmentation and under-segmentation in a different way. Reporting both helps avoid a single-metric mirage.

## Sensitivity and specificity

Sensitivity tells me how much of the true lesion area is captured. Specificity tells me how much background is correctly ignored.

## Precision

Precision estimates how much of the predicted lesion area is actually lesion. Low precision can indicate over-segmentation.

## Lesion-area error

Area error is clinically intuitive: if the model marks a region twice as large as the true lesion, the overlap score alone may not tell the whole story.

## Boundary-distance proxy

The boundary-distance proxy estimates how far the predicted boundary is from the true boundary. It is not a replacement for formal Hausdorff distance, but it gives a useful geometric failure signal.

## Threshold sweep

The agent writes `results/threshold_sweep.csv` and `figures/threshold_sweep.png`. I use the validation set to select a threshold before final testing.

## Review queue criteria

Cases can be flagged for review when:

- Dice is below the configured threshold
- predicted mask confidence is low
- the model predicts no lesion mask
- area error is unusually large

This is where the agent becomes useful for clinical translation: it routes uncertainty instead of hiding it.
