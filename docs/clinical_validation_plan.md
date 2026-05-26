# Clinical Validation Plan

This project is not clinically validated. The correct claim is narrower:

> If real-data test performance is strong and failure modes are understandable, the model may warrant the next stage of clinical validation.

## Proposed next step

1. Freeze preprocessing, model weights, threshold, and evaluation code.
2. Assemble a held-out external dataset from a different scanner/site if possible.
3. Ask qualified experts to review model overlays while blinded to model confidence.
4. Measure agreement between model masks and expert masks.
5. Track failure modes by lesion type, image quality, artifact level, and modality.
6. Compare the model against a simple baseline and a strong baseline.
7. Report results using medical AI reporting guidance such as CONSORT-AI/SPIRIT-AI for future clinical study design [5], [6].

## Minimum evidence I would want before clinician-facing validation

- Stable Dice and IoU across folds or external data
- Low empty-prediction rate
- Acceptable sensitivity at the selected threshold
- Clear qualitative overlays on best, median, and worst cases
- No hidden data leakage
- A documented review queue for uncertain cases

## What would block validation

- Poor lesion localization on small masses
- Strong performance only on a narrow image style
- Large gap between validation and test results
- Unexplained failure clusters
- Missing metadata needed for subgroup analysis
