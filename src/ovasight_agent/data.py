from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch


def _stem_key(path: Path) -> str:
    stem = path.stem.lower()
    for token in ["_mask", "-mask", " mask", "_label", "-label", "_seg", "-seg", "_gt", "-gt", "annotation"]:
        stem = stem.replace(token, "")
    stem = re.sub(r"[^a-z0-9]+", "", stem)
    return stem


def find_files(root: Path, extensions: Iterable[str]) -> list[Path]:
    extensions = {e.lower() for e in extensions}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in extensions])


def pair_images_and_masks(raw_root: Path, extensions: list[str], mask_keywords: list[str]) -> pd.DataFrame:
    all_files = find_files(raw_root, extensions)
    mask_keywords_lower = [k.lower() for k in mask_keywords]

    masks = []
    images = []
    for p in all_files:
        parts = [x.lower() for x in p.parts]
        name = p.name.lower()
        is_mask = any(k in name or k in parts for k in mask_keywords_lower)
        if is_mask:
            masks.append(p)
        else:
            images.append(p)

    mask_map = {_stem_key(p): p for p in masks}
    rows = []
    for img in images:
        key = _stem_key(img)
        mask = mask_map.get(key)
        if mask:
            rows.append({"case_id": key, "image_path": str(img), "mask_path": str(mask)})

    return pd.DataFrame(rows).drop_duplicates("case_id") if rows else pd.DataFrame(columns=["case_id", "image_path", "mask_path"])


def image_hash(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(data).hexdigest()[:12]


def audit_dataset(cfg: dict) -> dict:
    raw_root = Path(cfg["data"]["raw_root"])
    manifest = pair_images_and_masks(raw_root, cfg["data"]["image_extensions"], cfg["data"]["mask_keywords"])

    audit_rows = []
    for _, row in manifest.iterrows():
        image_path = Path(row.image_path)
        mask_path = Path(row.mask_path)
        record = {"case_id": row.case_id, "image_path": str(image_path), "mask_path": str(mask_path)}
        try:
            img = Image.open(image_path).convert("L")
            mask = Image.open(mask_path).convert("L")
            mask_arr = np.array(mask)
            record.update({
                "image_width": img.width,
                "image_height": img.height,
                "mask_width": mask.width,
                "mask_height": mask.height,
                "mask_area_px": int((mask_arr > 0).sum()),
                "blank_mask": bool((mask_arr > 0).sum() < cfg["data"]["min_mask_area_px"]),
                "image_hash": image_hash(image_path),
                "status": "ok",
            })
        except Exception as exc:
            record.update({"status": "error", "error": repr(exc)})
        audit_rows.append(record)

    df = pd.DataFrame(audit_rows)
    Path(cfg["outputs"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(cfg["outputs"]["results_dir"]) / "data_audit.csv", index=False)
    summary = {
        "raw_root": str(raw_root),
        "paired_cases": int(len(manifest)),
        "valid_cases": int((df.get("status", pd.Series(dtype=str)) == "ok").sum()) if len(df) else 0,
        "blank_masks": int(df.get("blank_mask", pd.Series(dtype=bool)).sum()) if len(df) else 0,
        "note": "No raw data is included in this repository. Place MMOTU under data/raw/MMOTU before running.",
    }
    with open(Path(cfg["outputs"]["results_dir"]) / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def preprocess_image(path: str | Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def preprocess_mask(path: str | Path, size: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.float32)
    return mask


def prepare_dataset(cfg: dict) -> pd.DataFrame:
    raw_root = Path(cfg["data"]["raw_root"])
    processed_root = Path(cfg["data"]["processed_root"])
    image_dir = processed_root / "images"
    mask_dir = processed_root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    manifest = pair_images_and_masks(raw_root, cfg["data"]["image_extensions"], cfg["data"]["mask_keywords"])
    if manifest.empty:
        raise RuntimeError("No image/mask pairs found. Check data/raw/MMOTU and mask naming patterns.")

    rows = []
    size = int(cfg["data"]["image_size"])
    for _, row in manifest.iterrows():
        img = preprocess_image(row.image_path, size)
        mask = preprocess_mask(row.mask_path, size)
        if mask.sum() < cfg["data"]["min_mask_area_px"]:
            continue
        out_img = image_dir / f"{row.case_id}.png"
        out_mask = mask_dir / f"{row.case_id}.png"
        cv2.imwrite(str(out_img), (img * 255).astype(np.uint8))
        cv2.imwrite(str(out_mask), (mask * 255).astype(np.uint8))
        rows.append({"case_id": row.case_id, "image_path": str(out_img), "mask_path": str(out_mask)})

    df = pd.DataFrame(rows)
    train_df, temp_df = train_test_split(df, test_size=cfg["data"]["val_fraction"] + cfg["data"]["test_fraction"], random_state=cfg["seed"], shuffle=True)
    val_relative = cfg["data"]["val_fraction"] / (cfg["data"]["val_fraction"] + cfg["data"]["test_fraction"])
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_relative, random_state=cfg["seed"], shuffle=True)
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    split_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    split_df.to_csv(Path(cfg["outputs"]["results_dir"]) / "splits.csv", index=False)
    return split_df


class SegmentationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = (cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy(); mask = np.fliplr(mask).copy()
            if np.random.rand() < 0.2:
                image = np.clip(image + np.random.normal(0, 0.02, image.shape), 0, 1)

        image = torch.from_numpy(image[None, ...]).float()
        mask = torch.from_numpy(mask[None, ...]).float()
        return image, mask, row.case_id
