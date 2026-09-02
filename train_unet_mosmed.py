import os
import sys
import argparse
import json
import math
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF


# ============================================================
# Paths
# ============================================================

ROOT = "/data/leiw/Summer"

MOSMED_ROOT = os.path.join(
    ROOT,
    "datasets",
    "MosMedDataPlus"
)

SPLIT_DIR = os.path.join(
    MOSMED_ROOT,
    "splits"
)

RESULT_DIR = os.path.join(
    ROOT,
    "results",
    "unet_mosmed"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# Import existing U-Net
# ============================================================

LVIT_CODE = os.path.join(ROOT, "code", "LViT")
sys.path.insert(0, LVIT_CODE)

from nets.UNet import UNet


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=666):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Build file index
# ============================================================

def build_mosmed_index():
    """
    Build the fixed MosMedData+ same-filename frame/mask mapping.
    """
    frame_dir = os.path.join(MOSMED_ROOT, "frames")
    mask_dir = os.path.join(MOSMED_ROOT, "masks")
    valid_extensions = (".png", ".jpg", ".jpeg")
    image_index = {
        filename: os.path.join(frame_dir, filename)
        for filename in os.listdir(frame_dir)
        if filename.lower().endswith(valid_extensions)
    }
    mask_index = {
        filename: os.path.join(mask_dir, filename)
        for filename in os.listdir(mask_dir)
        if filename.lower().endswith(valid_extensions)
    }

    return image_index, mask_index


print("Scanning MosMedData+ dataset...")

IMAGE_INDEX, MASK_INDEX = build_mosmed_index()

print("Images found:", len(IMAGE_INDEX))
print("Masks found:", len(MASK_INDEX))


# ============================================================
# Dataset
# ============================================================

class MosMedDataset(Dataset):

    def __init__(
        self,
        csv_path,
        image_size=224,
        augment=False
    ):

        self.df = pd.read_csv(csv_path)

        if "filename" not in self.df.columns:
            raise ValueError(
                f"'filename' column not found in {csv_path}"
            )

        self.filenames = self.df["filename"].astype(str).tolist()

        self.image_size = image_size
        self.augment = augment

        missing_images = []
        missing_masks = []

        for filename in self.filenames:

            if filename not in IMAGE_INDEX:
                missing_images.append(filename)

            if filename not in MASK_INDEX:
                missing_masks.append(filename)

        if missing_images:
            raise FileNotFoundError(
                f"Missing {len(missing_images)} images. "
                f"Examples: {missing_images[:5]}"
            )

        if missing_masks:
            raise FileNotFoundError(
                f"Missing {len(missing_masks)} masks. "
                f"Examples: {missing_masks[:5]}"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]

        image_path = IMAGE_INDEX[filename]
        mask_path = MASK_INDEX[filename]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(
            image,
            [self.image_size, self.image_size]
        )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.NEAREST
        )

        # simple augmentation for training
        if self.augment:

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # force binary mask
        mask = (mask > 0.5).float()

        return image, mask, filename


# ============================================================
# Loss
# ============================================================

def dice_loss(pred, target, smooth=1e-6):

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)

    dice = (
        (2 * intersection + smooth)
        /
        (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    )

    return 1 - dice.mean()


class DiceBCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):

        bce = self.bce(pred, target)
        dice = dice_loss(pred, target)

        return 0.5 * bce + 0.5 * dice


# ============================================================
# Dice metric
# ============================================================

def dice_scores(pred, target, threshold=0.5, smooth=1e-6):

    pred = (pred > threshold).float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)

    dice = (
        (2 * intersection + smooth)
        /
        (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    )

    return dice


def iou_scores(pred, target, threshold=0.5, smooth=1e-6):

    pred = (pred > threshold).float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou


def surface_points(mask):
    """Return coordinates of the one-pixel, 8-connected foreground surface."""

    mask = mask.bool()
    neighbors = F.conv2d(
        mask[None, None].float(),
        torch.ones((1, 1, 3, 3), device=mask.device),
        padding=1
    )[0, 0]
    surface = mask & (neighbors < 9)

    return torch.nonzero(surface, as_tuple=False).float()


def directed_surface_distances(source, target, chunk_size=2048):
    """Nearest Euclidean distance from each source point to the target surface."""

    distances = []

    for start in range(0, len(source), chunk_size):
        distances.append(
            torch.cdist(source[start:start + chunk_size], target).min(dim=1).values
        )

    return torch.cat(distances)


def surface_distance_scores(pred, target, threshold=0.5, chunk_size=2048):
    """Return per-image HD95 and ASSD in pixels after resizing."""

    pred = (pred > threshold)
    target = target.bool()
    hd95_values = []
    assd_values = []

    for pred_mask, target_mask in zip(pred[:, 0], target[:, 0]):
        pred_empty = not pred_mask.any().item()
        target_empty = not target_mask.any().item()

        if pred_empty and target_empty:
            hd95_values.append(0.0)
            assd_values.append(0.0)
            continue

        if pred_empty or target_empty:
            height, width = pred_mask.shape
            maximum_distance = math.hypot(height - 1, width - 1)
            hd95_values.append(maximum_distance)
            assd_values.append(maximum_distance)
            continue

        pred_surface = surface_points(pred_mask)
        target_surface = surface_points(target_mask)
        distances = torch.cat([
            directed_surface_distances(pred_surface, target_surface, chunk_size),
            directed_surface_distances(target_surface, pred_surface, chunk_size)
        ])

        hd95_values.append(torch.quantile(distances, 0.95).item())
        assd_values.append(distances.mean().item())

    return hd95_values, assd_values


def model_complexity(model, image_size):
    """Count parameters and convolutional FLOPs for one RGB input image."""

    macs = 0
    handles = []

    def count_convolution(module, inputs, output):
        nonlocal macs

        kernel_area = math.prod(module.kernel_size)

        if isinstance(module, nn.ConvTranspose2d):
            operations = (
                inputs[0].numel()
                * (module.out_channels // module.groups)
                * kernel_area
            )
        else:
            operations = (
                output.numel()
                * (module.in_channels // module.groups)
                * kernel_area
            )

        if module.bias is not None:
            operations += output.numel()

        macs += operations

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(module.register_forward_hook(count_convolution))

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    was_training = model.training
    model.eval()

    with torch.no_grad():
        model(torch.zeros(1, 3, image_size, image_size, device=next(model.parameters()).device))

    for handle in handles:
        handle.remove()

    model.train(was_training)

    # One multiply-accumulate is reported as two floating-point operations.
    return parameter_count, int(2 * macs)


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0
    total_dice = 0

    for images, masks, _ in loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        preds = model(images)

        loss = criterion(preds, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        total_dice += dice_scores(
            preds.detach(),
            masks
        ).sum().item()

    return (
        total_loss / len(loader),
        total_dice / len(loader.dataset)
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    threshold=0.5,
    include_surface_distances=False,
    surface_chunk_size=2048
):

    model.eval()

    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_hd95 = 0
    total_assd = 0

    for images, masks, _ in loader:

        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        loss = criterion(preds, masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        total_dice += dice_scores(
            preds,
            masks,
            threshold=threshold
        ).sum().item()

        total_iou += iou_scores(
            preds,
            masks,
            threshold=threshold
        ).sum().item()

        if include_surface_distances:
            hd95_values, assd_values = surface_distance_scores(
                preds,
                masks,
                threshold=threshold,
                chunk_size=surface_chunk_size
            )
            total_hd95 += sum(hd95_values)
            total_assd += sum(assd_values)

    sample_count = len(loader.dataset)
    metrics = {
        "loss": total_loss / sample_count,
        "dice": total_dice / sample_count,
        "miou": total_iou / sample_count,
    }

    if include_surface_distances:
        metrics["hd95_pixels"] = total_hd95 / sample_count
        metrics["assd_pixels"] = total_assd / sample_count

    return metrics


# ============================================================
# Main
# ============================================================

def main(args):

    if args.image_size != 224:
        raise ValueError("Use image_size=224 to preserve the QaTa U-Net setup")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    train_csv = os.path.join(
        SPLIT_DIR,
        "train.csv"
    )

    val_csv = os.path.join(
        SPLIT_DIR,
        "val.csv"
    )

    test_csv = os.path.join(
        SPLIT_DIR,
        "test.csv"
    )

    train_dataset = MosMedDataset(
        train_csv,
        image_size=args.image_size,
        augment=True
    )

    val_dataset = MosMedDataset(
        val_csv,
        image_size=args.image_size,
        augment=False
    )

    test_dataset = MosMedDataset(
        test_csv,
        image_size=args.image_size,
        augment=False
    )

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Val samples:",
        len(val_dataset)
    )

    print(
        "Test samples:",
        len(test_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = UNet(
        n_channels=3,
        n_classes=1
    ).to(device)

    parameter_count, flops = model_complexity(
        model,
        args.image_size
    )

    print("Model parameters:", parameter_count)
    print(
        f"FLOPs per {args.image_size}x{args.image_size} image: {flops} "
        "(convolutional multiply-add = 2 FLOPs)"
    )

    criterion = DiceBCELoss()

    if args.preflight_only:
        images, masks, filenames = next(iter(train_loader))
        images = images.to(device)
        masks = masks.to(device)
        model.train()
        predictions = model(images)
        loss = criterion(predictions, masks)
        loss.backward()
        model.zero_grad(set_to_none=True)
        print("Preflight filenames:", list(filenames))
        print("Preflight image shape:", list(images.shape))
        print("Preflight mask shape:", list(masks.shape))
        print("Preflight output shape:", list(predictions.shape))
        print("Preflight loss:", float(loss.detach()))
        print("Preflight forward/backward: PASS")
        return

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    best_val_dice = -1

    best_path = os.path.join(
        RESULT_DIR,
        "best_unet_mosmed_smoke.pth" if args.epochs == 1
        else "best_unet_mosmed.pth"
    )

    for epoch in range(
        1,
        args.epochs + 1
    ):

        train_loss, train_dice = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=args.threshold
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val mIoU: {val_metrics['miou']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:

            best_val_dice = val_metrics["dice"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_metrics["dice"],
                    "val_miou": val_metrics["miou"],
                    "image_size": args.image_size,
                    "threshold": args.threshold,
                },
                best_path
            )

            print(
                f"Saved new best model: "
                f"Val Dice = {val_metrics['dice']:.4f}"
            )

    if args.epochs == 1:
        smoke_metrics = {
            "epochs_completed": 1,
            "training_loss": train_loss,
            "validation_dice": val_metrics["dice"],
            "validation_miou": val_metrics["miou"],
            "model_parameters": parameter_count,
            "flops_per_image": flops,
            "flops_convention": "convolutional multiply-add = 2 FLOPs",
            "image_size": args.image_size,
            "threshold": args.threshold,
            "test_evaluated": False,
            "checkpoint": best_path,
        }
        metrics_path = os.path.join(RESULT_DIR, "smoke_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            json.dump(smoke_metrics, metrics_file, indent=2)
            metrics_file.write("\n")
        print("Smoke test finished without test-set evaluation.")
        print("Checkpoint:", best_path)
        print("Smoke metrics:", metrics_path)
        return

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print()
    print("Evaluating the validation-selected best checkpoint...")

    final_val_metrics = evaluate(
        model,
        val_loader,
        criterion,
        device,
        threshold=args.threshold,
        include_surface_distances=True,
        surface_chunk_size=args.surface_chunk_size
    )
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        threshold=args.threshold,
        include_surface_distances=True,
        surface_chunk_size=args.surface_chunk_size
    )

    final_metrics = {
        "best_epoch": checkpoint["epoch"],
        "validation_dice": final_val_metrics["dice"],
        "validation_miou": final_val_metrics["miou"],
        "validation_hd95_pixels": final_val_metrics["hd95_pixels"],
        "validation_assd_pixels": final_val_metrics["assd_pixels"],
        "test_dice": test_metrics["dice"],
        "test_miou": test_metrics["miou"],
        "test_hd95_pixels": test_metrics["hd95_pixels"],
        "test_assd_pixels": test_metrics["assd_pixels"],
        "model_parameters": parameter_count,
        "flops_per_image": flops,
        "flops_convention": "convolutional multiply-add = 2 FLOPs",
        "image_size": args.image_size,
        "threshold": args.threshold,
        "checkpoint": best_path,
    }
    metrics_path = os.path.join(RESULT_DIR, "final_metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(final_metrics, metrics_file, indent=2)
        metrics_file.write("\n")

    print()
    print("Training finished.")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Validation Dice: {final_val_metrics['dice']:.6f}")
    print(f"Validation mIoU: {final_val_metrics['miou']:.6f}")
    print(f"Validation HD95 (pixels): {final_val_metrics['hd95_pixels']:.6f}")
    print(f"Validation ASSD (pixels): {final_val_metrics['assd_pixels']:.6f}")
    print(f"Test Dice: {test_metrics['dice']:.6f}")
    print(f"Test mIoU: {test_metrics['miou']:.6f}")
    print(f"Test HD95 (pixels): {test_metrics['hd95_pixels']:.6f}")
    print(f"Test ASSD (pixels): {test_metrics['assd_pixels']:.6f}")
    print(f"Model parameters: {parameter_count}")
    print(f"FLOPs per image: {flops}")
    print("Best checkpoint:", best_path)
    print("Metrics JSON:", metrics_path)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=666
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--surface_chunk_size",
        type=int,
        default=2048
    )

    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Run one training-batch forward/backward check without training"
    )

    args = parser.parse_args()

    main(args)
