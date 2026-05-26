# M2 MacBook Runbook

## 1. Install Python

Use Python 3.10 or 3.11. If you use Homebrew:

```bash
brew install python@3.11
```

## 2. Create environment

```bash
cd ~/Downloads/OvaSight-Agent
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Add the dataset

Place the MMOTU dataset under:

```text
data/raw/MMOTU/
```

## 4. Run audit first

```bash
python -m ovasight_agent.agent audit --config configs/mmotu_unet_m2.yaml
```

Open the audit summary:

```bash
cat results/data_audit.json
```

## 5. Run the full pipeline

```bash
python -m ovasight_agent.agent run --config configs/mmotu_unet_m2.yaml
```

## 6. Open outputs

```bash
open figures/training_curves.png
open figures/threshold_sweep.png
open figures/qualitative_grid.png
open reports/clinical_readiness_report.md
```

## MPS notes

PyTorch should automatically use Apple Silicon MPS if available. If something behaves strangely, switch to CPU in `configs/mmotu_unet_m2.yaml`:

```yaml
training:
  device: cpu
```

CPU will be slower, but it is useful for debugging.
