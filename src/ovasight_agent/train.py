from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SegmentationDataset
from .metrics import dice_score
from .model import DiceBCELoss, UNetSmall
from .utils import select_device


def _epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    losses, dices = [], []
    for images, masks, _ in tqdm(loader, leave=False):
        images, masks = images.to(device), masks.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = loss_fn(logits, masks)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y = masks.detach().cpu().numpy()
        for i in range(len(probs)):
            dices.append(dice_score(y[i, 0] > 0.5, probs[i, 0] > 0.5))
        losses.append(float(loss.detach().cpu()))
    return {"loss": sum(losses) / max(len(losses), 1), "dice": sum(dices) / max(len(dices), 1)}


def train_model(cfg: dict) -> dict:
    splits = pd.read_csv(Path(cfg["outputs"]["results_dir"]) / "splits.csv")
    train_df = splits[splits.split == "train"]
    val_df = splits[splits.split == "val"]

    device = select_device(cfg["training"].get("device", "auto"))
    model = UNetSmall(cfg["model"]["base_channels"], cfg["model"]["dropout"]).to(device)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])

    train_loader = DataLoader(SegmentationDataset(train_df, augment=True), batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"])
    val_loader = DataLoader(SegmentationDataset(val_df, augment=False), batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"])

    best_dice = -1
    wait = 0
    history = []
    model_dir = Path(cfg["outputs"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "ovasight_unet_best.pt"

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        tr = _epoch(model, train_loader, loss_fn, device, optimizer)
        va = _epoch(model, val_loader, loss_fn, device)
        record = {"epoch": epoch, "train_loss": tr["loss"], "train_dice": tr["dice"], "val_loss": va["loss"], "val_dice": va["dice"], "device": str(device)}
        print(record)
        history.append(record)
        if va["dice"] > best_dice:
            best_dice = va["dice"]
            wait = 0
            torch.save({"model_state": model.state_dict(), "config": cfg, "best_val_dice": best_dice}, best_path)
        else:
            wait += 1
            if wait >= cfg["training"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(Path(cfg["outputs"]["results_dir"]) / "training_history.csv", index=False)
    return {"best_model": str(best_path), "best_val_dice": float(best_dice), "device": str(device)}
