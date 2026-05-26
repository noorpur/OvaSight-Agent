from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import SegmentationDataset
from .metrics import area_error, boundary_distance_proxy, dice_score, iou_score, precision, sensitivity, specificity, summarize
from .model import UNetSmall
from .utils import select_device, write_json


def load_best_model(cfg: dict):
    device = select_device(cfg["training"].get("device", "auto"))
    model = UNetSmall(cfg["model"]["base_channels"], cfg["model"]["dropout"]).to(device)
    checkpoint = torch.load(Path(cfg["outputs"]["model_dir"]) / "ovasight_unet_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device


def collect_predictions(cfg: dict, split: str):
    splits = pd.read_csv(Path(cfg["outputs"]["results_dir"]) / "splits.csv")
    frame = splits[splits.split == split]
    model, device = load_best_model(cfg)
    loader = DataLoader(SegmentationDataset(frame, augment=False), batch_size=1, shuffle=False, num_workers=0)
    records = []
    with torch.no_grad():
        for image, mask, case_id in loader:
            logits = model(image.to(device))
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
            records.append({"case_id": case_id[0], "image": image.numpy()[0, 0], "mask": mask.numpy()[0, 0], "prob": prob})
    return records


def tune_threshold(cfg: dict) -> dict:
    records = collect_predictions(cfg, "val")
    rows = []
    for t in cfg["threshold_tuning"]["thresholds"]:
        dices, ious = [], []
        for r in records:
            pred = r["prob"] >= t
            true = r["mask"] > 0.5
            dices.append(dice_score(true, pred))
            ious.append(iou_score(true, pred))
        rows.append({"threshold": t, "dice": float(np.mean(dices)), "iou": float(np.mean(ious))})
    df = pd.DataFrame(rows)
    df.to_csv(Path(cfg["outputs"]["results_dir"]) / "threshold_sweep.csv", index=False)
    metric = cfg["threshold_tuning"]["primary_metric"]
    best = df.sort_values(metric, ascending=False).iloc[0].to_dict()
    write_json(Path(cfg["outputs"]["results_dir"]) / "best_threshold.json", best)
    plot_threshold_sweep(cfg, df)
    return best


def evaluate_test_set(cfg: dict) -> dict:
    threshold_path = Path(cfg["outputs"]["results_dir"]) / "best_threshold.json"
    threshold = 0.5
    if threshold_path.exists():
        import json
        threshold = float(json.loads(threshold_path.read_text())["threshold"])

    records = collect_predictions(cfg, "test")
    rows = []
    for r in records:
        pred = r["prob"] >= threshold
        true = r["mask"] > 0.5
        rows.append({
            "case_id": r["case_id"],
            "dice": dice_score(true, pred),
            "iou": iou_score(true, pred),
            "sensitivity": sensitivity(true, pred),
            "specificity": specificity(true, pred),
            "precision": precision(true, pred),
            "area_error": area_error(true, pred),
            "boundary_distance_proxy": boundary_distance_proxy(true, pred),
            "mean_probability_inside_pred": float(r["prob"][pred].mean()) if pred.any() else 0.0,
            "empty_prediction": bool(pred.sum() == 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(Path(cfg["outputs"]["results_dir"]) / "case_metrics.csv", index=False)
    summary = {m: summarize(df[m].tolist()) for m in ["dice", "iou", "sensitivity", "specificity", "precision", "area_error", "boundary_distance_proxy"]}
    summary["threshold"] = threshold
    summary["n_test_cases"] = int(len(df))
    summary["empty_prediction_rate"] = float(df.empty_prediction.mean()) if len(df) else None
    write_json(Path(cfg["outputs"]["results_dir"]) / "test_metrics.json", summary)
    write_review_queue(cfg, df)
    plot_training_curves(cfg)
    plot_qualitative_grid(cfg, records, threshold)
    return summary


def write_review_queue(cfg: dict, df: pd.DataFrame) -> None:
    low_dice = cfg["qc"]["low_dice_flag"]
    low_conf = cfg["qc"]["low_confidence_flag"]
    review = df.copy()
    review["review_reason"] = ""
    review.loc[review.dice < low_dice, "review_reason"] += "low_dice;"
    review.loc[review.mean_probability_inside_pred < low_conf, "review_reason"] += "low_confidence;"
    review.loc[review.empty_prediction, "review_reason"] += "empty_prediction;"
    review = review[review.review_reason != ""].sort_values(["dice", "mean_probability_inside_pred"]).head(cfg["qc"]["review_top_k"])
    review.to_csv(Path(cfg["outputs"]["results_dir"]) / "review_queue.csv", index=False)


def plot_training_curves(cfg: dict) -> None:
    path = Path(cfg["outputs"]["results_dir"]) / "training_history.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig_dir = Path(cfg["outputs"]["figures_dir"]); fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(df.epoch, df.train_dice, label="train Dice")
    plt.plot(df.epoch, df.val_dice, label="validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Training and validation Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "training_curves.png", dpi=180)
    plt.close()


def plot_threshold_sweep(cfg: dict, df: pd.DataFrame) -> None:
    fig_dir = Path(cfg["outputs"]["figures_dir"]); fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(df.threshold, df.dice, marker="o", label="Dice")
    plt.plot(df.threshold, df.iou, marker="o", label="IoU")
    plt.xlabel("Segmentation probability threshold")
    plt.ylabel("Validation score")
    plt.title("Threshold tuning on validation set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "threshold_sweep.png", dpi=180)
    plt.close()


def plot_qualitative_grid(cfg: dict, records: list[dict], threshold: float) -> None:
    fig_dir = Path(cfg["outputs"]["figures_dir"]); fig_dir.mkdir(parents=True, exist_ok=True)
    n = min(6, len(records))
    if n == 0:
        return
    plt.figure(figsize=(10, 3 * n))
    for i, r in enumerate(records[:n]):
        image, true, prob = r["image"], r["mask"] > 0.5, r["prob"]
        pred = prob >= threshold
        overlay = np.stack([image, image, image], axis=-1)
        overlay[..., 0] = np.maximum(overlay[..., 0], pred.astype(float))
        overlay[..., 1] = np.maximum(overlay[..., 1], true.astype(float) * 0.8)
        ax = plt.subplot(n, 3, i * 3 + 1); ax.imshow(image, cmap="gray"); ax.set_title(f"{r['case_id']} image"); ax.axis("off")
        ax = plt.subplot(n, 3, i * 3 + 2); ax.imshow(prob, cmap="gray"); ax.set_title("probability map"); ax.axis("off")
        ax = plt.subplot(n, 3, i * 3 + 3); ax.imshow(overlay); ax.set_title("overlay: GT green, pred red"); ax.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / "qualitative_grid.png", dpi=180)
    plt.close()
