#!/usr/bin/env python3
"""Local training runner for the paper's ResNet34-FCN landfill segmentation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


MEAN_4CH = torch.tensor([0.485, 0.456, 0.406, 0.5], dtype=torch.float32)
STD_4CH = torch.tensor([0.229, 0.224, 0.225, 0.5], dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the paper-aligned ResNet34-FCN model for landfill segmentation."
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing TIFF images.")
    parser.add_argument("--labels-csv", type=Path, required=True, help="CSV with columns like Idx, Image Index, IsLandfill, json index.")
    parser.add_argument("--json-dir", type=Path, required=True, help="Directory containing GeoJSON polygon files.")
    parser.add_argument("--output-dir", type=Path, default=Path("./runs/resnet34_fcn"), help="Directory to save checkpoints/logs/plots.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=3, help="CPU threads used by PyTorch on Mac/CPU.")
    parser.add_argument("--max-ram-gb", type=float, default=11.0, help="Soft process memory cap in GB (best effort).")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-imagenet-pretrain", action="store_true", help="Disable ImageNet initialization for ResNet34.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_runtime(torch_threads: int, max_ram_gb: float) -> None:
    os.environ["OMP_NUM_THREADS"] = str(max(1, torch_threads))
    torch.set_num_threads(max(1, torch_threads))
    torch.set_num_interop_threads(max(1, torch_threads))

    if max_ram_gb <= 0:
        return
    limit_bytes = int(max_ram_gb * (1024**3))
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_soft = min(limit_bytes, hard) if hard > 0 else limit_bytes
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
        print(f"Configured RAM soft limit: {max_ram_gb:.2f} GB")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: unable to enforce RAM cap ({exc}).")


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and np.isnan(value)


def _pick_json_path(row: pd.Series, json_dir: Path) -> Optional[Path]:
    candidates: List[str] = []
    if "json index" in row and row["json index"] is not None and not _is_nan(row["json index"]):
        raw = str(row["json index"]).strip()
        if raw:
            candidates.append(raw)
            if Path(raw).suffix == "":
                candidates.extend([f"{raw}.geojson", f"{raw}.json"])
    else:
        stem = Path(str(row["Image Index"])).stem
        candidates.extend([f"{stem}.geojson", f"{stem}.json"])

    for candidate in candidates:
        path = json_dir / candidate
        if path.exists():
            return path
    if candidates:
        return json_dir / candidates[0]
    return None


def _band_select_for_paper(raster_chw: np.ndarray) -> np.ndarray:
    channels = raster_chw.shape[0]
    if channels == 8:
        # WorldView-3: R,G,B,NIR1 (indices 4,2,1,6) to match original code
        image = np.dstack((raster_chw[4], raster_chw[2], raster_chw[1], raster_chw[6]))
    elif channels == 4:
        # GeoEye-1: R,G,B,NIR (indices 2,1,0,3)
        image = np.dstack((raster_chw[2], raster_chw[1], raster_chw[0], raster_chw[3]))
    else:
        raise ValueError(f"Unsupported channel count: {channels}. Expected 4 or 8.")
    return image.astype(np.uint8)


def _geojson_geometries(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return []
    payload_type = payload.get("type")
    if payload_type == "FeatureCollection":
        geoms = []
        for feat in payload.get("features", []):
            geom = feat.get("geometry", {}) if isinstance(feat, dict) else {}
            if geom and geom.get("type") in {"Polygon", "MultiPolygon"}:
                geoms.append(geom)
        return geoms
    if payload_type in {"Polygon", "MultiPolygon"}:
        return [payload]
    return []


def _rasterize_mask(
    is_landfill: int,
    json_path: Optional[Path],
    shape_hw: Tuple[int, int],
    transform: rasterio.Affine,
) -> np.ndarray:
    if int(is_landfill) == 0 or json_path is None or not json_path.exists():
        return np.zeros(shape_hw, dtype=np.uint8)
    geometries = _geojson_geometries(json_path)
    if not geometries:
        return np.zeros(shape_hw, dtype=np.uint8)
    return rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )


def _fit_patch(image_hwc: np.ndarray, mask_hw: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = image_hwc.shape
    crop_h = min(h, patch_size)
    crop_w = min(w, patch_size)
    image_hwc = image_hwc[:crop_h, :crop_w, :]
    mask_hw = mask_hw[:crop_h, :crop_w]
    if crop_h == patch_size and crop_w == patch_size:
        return image_hwc, mask_hw
    image_out = np.zeros((patch_size, patch_size, image_hwc.shape[2]), dtype=image_hwc.dtype)
    mask_out = np.zeros((patch_size, patch_size), dtype=mask_hw.dtype)
    image_out[:crop_h, :crop_w, :] = image_hwc
    mask_out[:crop_h, :crop_w] = mask_hw
    return image_out, mask_out


class LandfillDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        images_dir: Path,
        json_dir: Path,
        patch_size: int,
        train: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.images_dir = images_dir
        self.json_dir = json_dir
        self.patch_size = patch_size
        self.train = train

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.frame.iloc[idx]
        img_name = str(row["Image Index"])
        image_path = self.images_dir / img_name
        with rasterio.open(image_path) as src:
            raster = src.read()
            height = src.height
            width = src.width
            transform = src.transform

        image = _band_select_for_paper(raster)
        json_path = _pick_json_path(row, self.json_dir)
        mask = _rasterize_mask(int(row["IsLandfill"]), json_path, (height, width), transform)
        image, mask = _fit_patch(image, mask, self.patch_size)

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask.astype(np.int64))

        if self.train:
            if random.random() > 0.5:
                image_t = torch.flip(image_t, dims=(2,))
                mask_t = torch.flip(mask_t, dims=(1,))
            if random.random() > 0.5:
                image_t = torch.flip(image_t, dims=(1,))
                mask_t = torch.flip(mask_t, dims=(0,))

        image_t = (image_t - MEAN_4CH[:, None, None]) / STD_4CH[:, None, None]

        mask_onehot = torch.nn.functional.one_hot(mask_t, num_classes=2).permute(2, 0, 1).float()
        return {"image": image_t, "mask": mask_t, "mask_onehot": mask_onehot, "name": img_name}


class ResNet34FCN(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        encoder = self._load_resnet34(pretrained=pretrained)
        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        # Mirrors original decoder layout in landfillpipeline_resnet_fcn.py
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_d1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_d2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_d5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 2, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _load_resnet34(pretrained: bool) -> nn.Module:
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            encoder = models.resnet34(weights=weights)
        except AttributeError:
            encoder = models.resnet34(pretrained=pretrained)

        old_conv = encoder.conv1
        new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
        encoder.conv1 = new_conv
        return encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x1 = x
        x = self.maxpool(x)
        x2 = x
        x = self.layer1(x)
        x3 = x
        x = self.layer2(x)
        x4 = x
        x = self.layer3(x)
        x5 = x
        x = self.layer4(x)
        x6 = x

        x = self.act(self.deconv1(x6))
        x = self.bn_d1(x + x5)
        x = self.act(self.deconv2(x))
        x = self.bn_d2(x + x4)
        x = self.bn_d3(self.act(self.deconv3(x)))
        x = self.bn_d4(self.act(self.deconv4(x)))
        x = self.bn_d5(self.act(self.deconv5(x)))
        return self.classifier(x)


@dataclass
class EpochStats:
    loss: float
    pixel_acc: float
    landfill_iou: float


def _batch_iou_landfill(pred: torch.Tensor, target: torch.Tensor) -> Tuple[int, int]:
    pred_pos = pred == 1
    target_pos = target == 1
    intersection = int((pred_pos & target_pos).sum().item())
    union = int((pred_pos | target_pos).sum().item())
    return intersection, union


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
) -> EpochStats:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0
    i_intersection = 0
    i_union = 0

    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        mask_onehot = batch["mask_onehot"].to(device)

        if training:
            optimizer.zero_grad()

        logits = model(image)
        loss = criterion(logits, mask_onehot)

        if training:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += int((pred == mask).sum().item())
            total += mask.numel()
            inter, union = _batch_iou_landfill(pred, mask)
            i_intersection += inter
            i_union += union

        total_loss += float(loss.item()) * image.size(0)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    pixel_acc = correct / max(total, 1)
    landfill_iou = (i_intersection / i_union) if i_union > 0 else 0.0
    return EpochStats(loss=avg_loss, pixel_acc=pixel_acc, landfill_iou=landfill_iou)


def save_visual_samples(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: Path, max_images: int = 6) -> None:
    if len(loader.dataset) == 0:
        return
    model.eval()
    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1)

    n = min(max_images, images.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    mean_rgb = MEAN_4CH[:3, None, None].to(device)
    std_rgb = STD_4CH[:3, None, None].to(device)

    for i in range(n):
        rgb = (images[i, :3] * std_rgb + mean_rgb).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        gt = masks[i].cpu().numpy()
        pdm = preds[i].cpu().numpy()

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(rgb)
        axes[i, 1].imshow(gt, cmap="Greens", alpha=0.35)
        axes[i, 1].set_title("Ground Truth Overlay")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(rgb)
        axes[i, 2].imshow(pdm, cmap="Reds", alpha=0.35)
        axes[i, 2].set_title("Prediction Overlay")
        axes[i, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "sample_predictions.png", dpi=180)
    plt.close(fig)


def validate_inputs(frame: pd.DataFrame, images_dir: Path, json_dir: Path) -> None:
    required_cols = {"Idx", "Image Index", "IsLandfill"}
    missing = required_cols - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_runtime(args.torch_threads, args.max_ram_gb)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.labels_csv)
    validate_inputs(frame, args.images_dir, args.json_dir)
    frame["IsLandfill"] = frame["IsLandfill"].astype(int)

    train_df, val_df = train_test_split(
        frame,
        test_size=args.val_split,
        random_state=args.seed,
        shuffle=True,
        stratify=frame["IsLandfill"],
    )

    train_ds = LandfillDataset(
        frame=train_df,
        images_dir=args.images_dir,
        json_dir=args.json_dir,
        patch_size=args.patch_size,
        train=True,
    )
    val_ds = LandfillDataset(
        frame=val_df,
        images_dir=args.images_dir,
        json_dir=args.json_dir,
        patch_size=args.patch_size,
        train=False,
    )

    loader_kwargs: Dict[str, object] = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34FCN(pretrained=not args.no_imagenet_pretrain).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    ckpt_path = args.output_dir / "best_resnet34_fcn.pth"
    metrics_path = args.output_dir / "metrics.csv"
    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []

    print(f"Device: {device}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, device, criterion, optimizer=optimizer)
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, device, criterion, optimizer=None)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_stats.loss,
            "train_pixel_acc": train_stats.pixel_acc,
            "train_landfill_iou": train_stats.landfill_iou,
            "val_loss": val_stats.loss,
            "val_pixel_acc": val_stats.pixel_acc,
            "val_landfill_iou": val_stats.landfill_iou,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_stats.loss:.4f} val_loss={val_stats.loss:.4f} | "
            f"train_acc={train_stats.pixel_acc:.4f} val_acc={val_stats.pixel_acc:.4f} | "
            f"train_iou={train_stats.landfill_iou:.4f} val_iou={val_stats.landfill_iou:.4f}"
        )

        if val_stats.loss < best_val_loss:
            best_val_loss = val_stats.loss
            torch.save(model.state_dict(), ckpt_path)

    pd.DataFrame(history).to_csv(metrics_path, index=False)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    save_visual_samples(model, val_loader, device, args.output_dir, max_images=min(6, args.batch_size))

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved visual sample: {args.output_dir / 'sample_predictions.png'}")


if __name__ == "__main__":
    main()
