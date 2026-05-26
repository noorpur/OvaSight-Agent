# OvaSight-Agent Improved Experiment Analysis

I ran a second, stronger OvaSight-Agent experiment on the real MMOTU ovarian ultrasound dataset. This run used a larger input resolution, a wider U-Net backbone, lower dropout, and an extended training schedule with early stopping. I treat this experiment as the main result for the repository, with the earlier 40-epoch run serving as the baseline.

## Experimental setup

- Dataset: MMOTU ovarian tumor ultrasound data
- Paired image-mask cases found: **1469**
- Valid cases used: **1469**
- Blank masks flagged: **0**
- Split: **1028 train / 220 validation / 221 test**
- Input size: **320 × 320**
- Model: U-Net style segmentation model
- Base channels: **48**
- Dropout: **0.05**
- Batch size: **4**
- Maximum epochs: **80**
- Early stopping: triggered after epoch **71**
- Best validation checkpoint: epoch **63**
- Best validation Dice: **0.709**
- Tuned segmentation threshold: **0.55**

## Baseline vs improved run

| Measure | Baseline | Improved | Change |
|---|---:|---:|---:|
| Best validation Dice | 0.697 | 0.709 | +0.011 |
| Selected threshold | 0.40 | 0.55 | +0.15 |
| Test mean Dice | 0.737 | 0.747 | +0.010 |
| Test median Dice | 0.827 | 0.847 | +0.020 |
| Test mean IoU | 0.629 | 0.639 | +0.010 |
| Mean sensitivity | 0.783 | 0.775 | -0.008 |
| Mean specificity | 0.968 | 0.972 | +0.004 |
| Mean precision | 0.767 | 0.779 | +0.012 |
| Mean lesion-area relative error | 0.472 | 0.384 | -0.087 |
| Mean boundary-distance proxy | 11.599 | 14.533 | +2.934 |
| Empty prediction rate | 0.452% | 0.000% | -0.452% |


The improved run produced a modest but meaningful gain over the first run. Mean test Dice improved from **0.737** to **0.747**, while median Dice improved from **0.827** to **0.847**. The model also eliminated empty predictions on the held-out test set, reducing the empty prediction rate from **0.45%** to **0.00%**.

The biggest practical improvement is not just the Dice increase. It is the combination of higher median Dice, higher precision, higher specificity, lower area error, and no empty masks. That makes the agent more usable as a human-in-the-loop annotation assistant because fewer outputs are unusable at first glance.

## Held-out test performance

| Metric | Mean | Median | P10 | P90 |
|---|---:|---:|---:|---:|
| Dice | 0.747 | 0.847 | 0.372 | 0.950 |
| IoU | 0.639 | 0.735 | 0.229 | 0.904 |
| Sensitivity | 0.775 | 0.859 | 0.402 | 0.985 |
| Specificity | 0.972 | 0.984 | 0.940 | 0.997 |
| Precision | 0.779 | 0.870 | 0.392 | 0.983 |
| Lesion-area relative error | 0.384 | 0.188 | 0.031 | 0.821 |
| Boundary-distance proxy | 14.533 | 11.478 | 3.909 | 28.079 |

The final model achieved a **mean Dice of 0.747** and a **median Dice of 0.847** across **221 held-out test cases**. The difference between mean and median is important: the median result is strong, but the lower tail still contains difficult cases. In other words, the model performs well on many ovarian ultrasound images, but not reliably enough to be treated as an autonomous clinical segmentation system.

## Case-level distribution

- Test cases evaluated: **221**
- Cases with Dice > 0.80: **125 / 221**
- Cases with Dice > 0.90: **66 / 221**
- Cases with Dice < 0.50: **34 / 221**
- Cases with Dice < 0.20: **6 / 221**
- Review queue size: **40 cases**

This distribution supports the agent design. The purpose of OvaSight is not just to output a mask. It also ranks uncertain or low-quality cases so a human reviewer can focus on the scans most likely to need correction. The 40-case review queue is therefore a feature of the pipeline, not a failure mode.

## Figure interpretation

### Training curves

The training curve shows consistent learning. Training Dice continued rising through the run, while validation Dice improved until the best checkpoint around epoch **63**. Early stopping triggered at epoch **71**, after validation performance stopped improving. The train-validation gap is present but not extreme, which suggests the model is learning meaningful lesion structure rather than simply memorizing the training masks.

### Threshold sweep

The validation threshold sweep selected **0.55**, with validation Dice **0.709** and IoU **0.598**. Compared with the baseline threshold of 0.40, the improved model preferred a more conservative cutoff. This matches the higher precision and specificity in the final test results: the model became better at suppressing false-positive background regions.

### Qualitative overlays

The qualitative grid shows that several cases have strong overlap between the ground-truth mask and predicted mask, especially examples where the lesion boundary is visually distinct. Some cases still show boundary mismatch or partial over-segmentation, especially around complex ultrasound texture, shadowing, or ambiguous lesion margins. This is consistent with the boundary-distance proxy worsening compared with the baseline, even while region-level metrics improved.

## Clinical-readiness interpretation

This experiment is strong enough to justify a **technical research portfolio claim** and a **next-step expert review plan**. It is not enough to claim clinical validation or diagnostic readiness.

My interpretation is:

> OvaSight-Agent reached a technically promising level of segmentation performance on a held-out MMOTU test split, with mean Dice **0.747**, median Dice **0.847**, and no empty predictions. The model is best positioned as a human-in-the-loop annotation assistant that can accelerate mask generation and prioritize difficult cases for expert review. Before any clinical claim, the system would need external validation, blinded expert assessment, scanner/site robustness testing, and predefined acceptance criteria.

## What improved

1. **Higher validation performance**  
   Best validation Dice increased from **0.697** to **0.709**.

2. **Better held-out segmentation quality**  
   Test mean Dice rose to **0.747**, and median Dice rose to **0.847**.

3. **No empty predictions**  
   The improved run produced no empty masks on the test set.

4. **Better background suppression**  
   Specificity increased to **0.972**, and precision increased to **0.779**.

5. **Lower area error**  
   Mean lesion-area relative error dropped from **0.472** to **0.384**, which matters if the masks are later used for lesion measurement or downstream quantitative analysis.

## What still needs work

1. **Lower-tail performance**  
   The p10 Dice is **0.372**, which means the worst-performing 10% of cases are still not good enough for direct downstream use.

2. **Boundary accuracy**  
   The boundary-distance proxy increased compared with the baseline, suggesting that the improved model may produce better regional overlap but less precise contours in some cases.

3. **External validation**  
   The model has only been evaluated on the current split. It still needs testing on external datasets or at least a separate institutional distribution.

4. **Expert review**  
   The review queue should be inspected by a qualified clinician or imaging expert to determine whether the failure modes are clinically acceptable, correctable, or systematic.

## Final repo wording

I would describe this project as:

> A human-in-the-loop ovarian ultrasound segmentation agent trained on real MMOTU data. The agent audits data quality, trains and tunes a U-Net segmentation model, evaluates held-out performance, and generates a review queue for uncertain or low-quality cases. The improved experiment achieved mean Dice **0.747** and median Dice **0.847** on **221** held-out test cases, supporting further expert review but not clinical deployment.

## Recommended next experiment

If I continue this project, the next technical experiment should focus on boundary quality rather than only Dice. I would test:

- boundary-aware loss, such as Dice + BCE + boundary loss
- stronger encoder backbone, such as ResNet or EfficientNet U-Net
- test-time augmentation
- post-processing to remove small false-positive regions
- separate analysis of 2D ultrasound versus CEUS/3D-style examples
- calibration plots to measure whether predicted probabilities reflect uncertainty

The current result is good enough to publish in the GitHub repo as the main experiment. The next step should be careful analysis and documentation, not endless retraining.
