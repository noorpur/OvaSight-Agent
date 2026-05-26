# Methodology

I designed OvaSight-Agent as an end-to-end research pipeline for ovarian tumor ultrasound segmentation. The emphasis is on disciplined experimentation rather than a single notebook result.

## 1. Data audit

The agent first recursively scans the raw MMOTU folder for image/mask pairs. It records dimensions, mask area, blank masks, corrupted files, duplicate-looking files through hashes, and pairing failures. I do this before training because medical imaging datasets often fail in unglamorous ways: inconsistent file names, empty masks, duplicated cases, and mixed image sizes.

## 2. Preprocessing

Images are converted to grayscale, resized to a configurable square resolution, and normalized to `[0, 1]`. Masks are resized with nearest-neighbor interpolation and binarized. The processed dataset is written separately from the raw dataset so I can reproduce exactly what was fed to the model.

## 3. Split strategy

The default split is train/validation/test with a fixed seed. Validation data is used for early stopping and threshold tuning. Test data is held until final evaluation.

## 4. Segmentation model

The default model is a compact U-Net-style encoder-decoder. U-Net remains a practical baseline for biomedical segmentation because it combines downsampled context with skip-connected high-resolution spatial detail [2]. I chose a smaller version for Apple Silicon compatibility and to keep the project runnable on a laptop.

## 5. Loss function

Training uses a Dice + BCE objective. BCE encourages pixelwise calibration, while Dice directly optimizes overlap, which matters when lesion pixels are a minority of the image.

## 6. Threshold tuning

Instead of assuming 0.5 is the best mask threshold, the agent sweeps thresholds on the validation set and selects the best threshold by Dice or IoU. This is important because segmentation maps are probability fields, and the operating point should be selected before touching the test set.

## 7. Evaluation

The test report includes Dice, IoU, sensitivity, specificity, precision, lesion-area error, empty-prediction rate, and a boundary-distance proxy. I intentionally report distribution summaries because a model can have a decent mean while failing badly on clinically important subgroups.

## 8. Agentic review queue

The agent flags cases for human review based on low Dice, low confidence, or empty predictions. This is the agent component: it does not merely train a model; it decides what needs expert attention and generates a clinical-readiness report.

## 9. Clinical-readiness decision

The final report does not claim clinical validation. It decides whether the technical outputs are strong enough to justify the next stage: blinded expert review and external validation.
