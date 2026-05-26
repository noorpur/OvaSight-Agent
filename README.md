# OvaSight-Agent: Human-in-the-Loop Ovarian Tumor Ultrasound Segmentation Agent

I built this project as a research-grade medical imaging pipeline for ovarian tumor ultrasound analysis. The goal is not to replace a clinician. The goal is to build an agent that can clean real data, train and tune segmentation models, evaluate whether outputs are technically strong enough to justify the next stage of clinical validation, and package uncertain cases for expert review.

The project is designed around the public **MMOTU ovarian tumor ultrasound dataset**, which contains 2D ultrasound and contrast-enhanced ultrasound images with pixel-wise lesion annotations and image-level labels [1]. The repository does **not** include raw patient data or synthetic demo results. It expects the real dataset to be downloaded separately and placed under `data/raw/`.

## What the agent does

The `OvaSight` agent runs a complete research pipeline:

1. **Data audit**: scans dataset folders, checks image/mask pairs, dimensions, blank masks, missing labels, corrupted files, and class imbalance.
2. **Data preparation**: standardizes images and masks, builds train/validation/test splits, and writes a manifest.
3. **Model training**: trains a lightweight U-Net style segmentation model on ovarian ultrasound images.
4. **Threshold tuning**: tunes the segmentation probability threshold on validation data instead of relying on a fixed 0.5 cutoff.
5. **Testing and metrics**: reports Dice, IoU, sensitivity, specificity, precision, Hausdorff-style boundary proxy, and lesion-area agreement.
6. **Quality-control agent**: flags low-confidence cases, empty predictions, high uncertainty, and mask-image mismatch risks.
7. **Clinical-readiness report**: generates figures and a report discussing whether the model is ready for a clinician-facing validation study.

## Why ovarian ultrasound?

Ovarian and adnexal masses are clinically important, visually heterogeneous, and often assessed through ultrasound. MMOTU is a good research dataset because it provides both pixel-level tumor segmentation masks and global category labels [1]. This lets the project do more than simple classification: it can produce lesion localization, quantitative metrics, and a review queue for ambiguous cases.

## Repository layout

```text
OvaSight-Agent/
├── configs/                  # YAML experiment configs
├── data/                     # raw/ and processed/ placeholders, no patient data included
├── docs/                     # methodology, data notes, metric definitions, validation plan
├── figures/                  # generated plots
├── models/                   # trained model checkpoints, ignored by git except .gitkeep
├── notebooks/                # optional exploration notebook placeholder
├── reports/                  # generated research reports
├── results/                  # generated metrics, predictions, review queues
├── scripts/                  # convenience shell commands
└── src/ovasight_agent/       # package source code
```

## Dataset setup

Download MMOTU from the official dataset source or a mirrored research platform. The original dataset/code release describes MMOTU as containing 1469 2D ultrasound images and 170 CEUS images with pixel-wise and global-wise annotations [1]. Place the extracted data here:

```text
data/raw/MMOTU/
```

Because public mirrors sometimes use slightly different folder names, the preparation script searches recursively for common image and mask naming patterns. The cleanest structure is:

```text
data/raw/MMOTU/
├── images/
│   ├── case_0001.png
│   └── ...
└── masks/
    ├── case_0001.png
    └── ...
```

If your downloaded dataset has different names, run the audit first. It will print what it found and what it could not pair.

## Quick start on an M2 MacBook

```bash
cd ~/Downloads
unzip OvaSight-Agent.zip
cd OvaSight-Agent

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the full agent after placing MMOTU under `data/raw/MMOTU/`:

```bash
python -m ovasight_agent.agent run --config configs/mmotu_unet_m2.yaml
```

Run one stage at a time:

```bash
python -m ovasight_agent.agent audit --config configs/mmotu_unet_m2.yaml
python -m ovasight_agent.agent prepare --config configs/mmotu_unet_m2.yaml
python -m ovasight_agent.agent train --config configs/mmotu_unet_m2.yaml
python -m ovasight_agent.agent tune --config configs/mmotu_unet_m2.yaml
python -m ovasight_agent.agent evaluate --config configs/mmotu_unet_m2.yaml
python -m ovasight_agent.agent report --config configs/mmotu_unet_m2.yaml
```

## Expected outputs

After running on real data, the agent writes:

```text
results/data_audit.json
results/splits.csv
results/training_history.csv
results/threshold_sweep.csv
results/test_metrics.json
results/review_queue.csv
figures/training_curves.png
figures/threshold_sweep.png
figures/qualitative_grid.png
reports/clinical_readiness_report.md
```

I intentionally do not include pre-filled metrics because I want the results in this repository to reflect the real dataset run, not synthetic evidence or decorative screenshots.

## Model

The default model is a compact U-Net inspired by the encoder-decoder biomedical segmentation architecture introduced by Ronneberger et al. [2]. I kept the architecture lightweight so it can run on Apple Silicon with PyTorch MPS acceleration, while still producing interpretable lesion masks.

## Clinical-readiness logic

The agent does not claim clinical validity. It asks a narrower research question:

> Are the segmentation outputs consistent enough, quantitatively and qualitatively, to justify a prospective review by qualified clinicians?

The readiness report checks:

- Dice and IoU distribution, not just means
- Failure modes in the worst cases
- Lesion area agreement
- Sensitivity/specificity tradeoff across thresholds
- Whether uncertain cases are routed to human review
- Whether the data split and preprocessing are documented

This follows the spirit of medical AI reporting guidance that emphasizes transparent dataset description, evaluation design, and limitations before clinical deployment [5].

## References

See [`docs/references.md`](docs/references.md).

## Improved Real-Data Experiment

I ran an improved OvaSight-Agent experiment on the real MMOTU ovarian ultrasound dataset using a larger image resolution and wider U-Net configuration.

### Configuration

| Setting | Value |
|---|---:|
| Input size | 320 × 320 |
| Base channels | 48 |
| Dropout | 0.05 |
| Batch size | 4 |
| Maximum epochs | 80 |
| Early stopping | epoch 71 |
| Best checkpoint | epoch 63 |
| Tuned threshold | 0.55 |

### Dataset split

| Split | Cases |
|---|---:|
| Train | 1028 |
| Validation | 220 |
| Test | 221 |

### Held-out test results

| Metric | Mean | Median | P10 | P90 |
|---|---:|---:|---:|---:|
| Dice | 0.747 | 0.847 | 0.372 | 0.950 |
| IoU | 0.639 | 0.735 | 0.229 | 0.904 |
| Sensitivity | 0.775 | 0.859 | 0.402 | 0.985 |
| Specificity | 0.972 | 0.984 | 0.940 | 0.997 |
| Precision | 0.779 | 0.870 | 0.392 | 0.983 |

The improved model achieved **mean Dice 0.747** and **median Dice 0.847** on **221 held-out test cases**, with an empty prediction rate of **0.00%**. Compared with the baseline experiment, the improved run increased validation Dice, mean test Dice, median test Dice, IoU, precision, and specificity, while lowering lesion-area error.

I interpret this result as technically promising for a **human-in-the-loop annotation assistant**. It is not a clinical validation result. The lower-tail Dice scores and boundary errors show that expert review is still necessary, which is why the agent also generates a review queue for difficult cases.

### Generated artifacts

- `figures/training_curves.png`
- `figures/threshold_sweep.png`
- `figures/qualitative_grid.png`
- `results/test_metrics.json`
- `results/case_metrics.csv`
- `results/review_queue.csv`
- `reports/clinical_readiness_report.md`
- `experiments/improved_80epoch_320px_real_run/`
