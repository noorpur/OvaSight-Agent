from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .data import audit_dataset, prepare_dataset
from .evaluate import evaluate_test_set, tune_threshold
from .report import generate_report
from .train import train_model
from .utils import ensure_dirs, load_config, seed_everything

console = Console()


def load_and_prepare(config_path: str):
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))
    ensure_dirs(cfg)
    return cfg


def run_audit(cfg):
    console.print(Panel.fit("OvaSight-Agent: dataset audit"))
    summary = audit_dataset(cfg)
    console.print(summary)
    return summary


def run_prepare(cfg):
    console.print(Panel.fit("OvaSight-Agent: data preparation"))
    splits = prepare_dataset(cfg)
    console.print(splits.split.value_counts().to_dict())
    return splits


def run_train(cfg):
    console.print(Panel.fit("OvaSight-Agent: model training"))
    result = train_model(cfg)
    console.print(result)
    return result


def run_tune(cfg):
    console.print(Panel.fit("OvaSight-Agent: threshold tuning"))
    result = tune_threshold(cfg)
    console.print(result)
    return result


def run_evaluate(cfg):
    console.print(Panel.fit("OvaSight-Agent: test-set evaluation"))
    result = evaluate_test_set(cfg)
    console.print(result)
    return result


def run_report(cfg):
    console.print(Panel.fit("OvaSight-Agent: clinical-readiness report"))
    path = generate_report(cfg)
    console.print(f"Report written to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="OvaSight-Agent pipeline")
    parser.add_argument("stage", choices=["audit", "prepare", "train", "tune", "evaluate", "report", "run"])
    parser.add_argument("--config", default="configs/mmotu_unet_m2.yaml")
    args = parser.parse_args()
    cfg = load_and_prepare(args.config)

    if args.stage in ["audit", "run"]:
        run_audit(cfg)
    if args.stage in ["prepare", "run"]:
        run_prepare(cfg)
    if args.stage in ["train", "run"]:
        run_train(cfg)
    if args.stage in ["tune", "run"]:
        run_tune(cfg)
    if args.stage in ["evaluate", "run"]:
        run_evaluate(cfg)
    if args.stage in ["report", "run"]:
        run_report(cfg)


if __name__ == "__main__":
    main()
