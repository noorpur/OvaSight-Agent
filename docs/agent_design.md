# Agent Design

OvaSight-Agent is a pipeline agent with six stages:

1. **Audit**: decide whether the dataset is usable.
2. **Prepare**: standardize images, masks, and splits.
3. **Train**: fit a segmentation model with early stopping.
4. **Tune**: select a probability threshold on validation data.
5. **Evaluate**: measure performance on held-out test data.
6. **Report**: generate a clinical-readiness report and review queue.

## Why call it an agent?

The agent makes workflow decisions that would otherwise be manual:

- It refuses to continue if image/mask pairs are missing.
- It filters near-empty masks during preprocessing.
- It selects a threshold from validation results.
- It identifies cases that need expert review.
- It writes a report explaining whether the results are technically ready for the next step.

## Human-in-the-loop behavior

The agent never presents segmentation as final truth. Instead, it produces `results/review_queue.csv`, which is the list I would hand to a clinician or domain expert first. This is deliberate: medical AI should make expert review easier, not silently bypass it.
