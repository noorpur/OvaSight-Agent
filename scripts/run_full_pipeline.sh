#!/usr/bin/env bash
set -euo pipefail
python -m ovasight_agent.agent run --config configs/mmotu_unet_m2.yaml
