import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import train_unet_mosmed as pipeline


ROOT = Path("/data/leiw/Summer")
BUSI_ROOT = ROOT / "datasets" / "Dataset_BUSI_with_GT_Clean"
SPLIT_DIR = BUSI_ROOT / "splits"
RESULT_DIR = ROOT / "results" / "unet_busi"
IMAGE_DIR = BUSI_ROOT / "images"
MASK_DIR = BUSI_ROOT / "labels"


class BUSIDataset(Dataset):
    """Fixed-split BUSI adapter with same-filename image/mask pairing."""

    def __init__(self, csv_path, image_size=224, augment=False):
        frame = pd.read_csv(csv_path)
        if list(frame.columns) != ["filename"]:
            raise ValueError(f"Expected only a 'filename' column in {csv_path}")
        self.filenames = frame["filename"].astype(str).tolist()
        if len(self.filenames) != len(set(self.filenames)):
            raise ValueError(f"Duplicate filenames in {csv_path}")
        self.image_size = image_size
        self.augment = augment

        missing_images = [name for name in self.filenames if not (IMAGE_DIR / name).is_file()]
        missing_masks = [name for name in self.filenames if not (MASK_DIR / name).is_file()]
        if missing_images:
            raise FileNotFoundError(f"Missing images: {missing_images[:5]}")
        if missing_masks:
            raise FileNotFoundError(f"Missing masks: {missing_masks[:5]}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        with Image.open(IMAGE_DIR / filename) as source:
            image = source.convert("RGB")
        with Image.open(MASK_DIR / filename) as source:
            # BUSI labels are palette images whose foreground value is 1.
            mask = Image.fromarray((np.asarray(source) > 0).astype(np.uint8) * 255, mode="L")

        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        if self.augment and pipeline.random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return TF.to_tensor(image), (TF.to_tensor(mask) > 0.5).float(), filename


def split_datasets(image_size):
    datasets = {
        "train": BUSIDataset(SPLIT_DIR / "train.csv", image_size, augment=True),
        "val": BUSIDataset(SPLIT_DIR / "val.csv", image_size, augment=False),
        "test": BUSIDataset(SPLIT_DIR / "test.csv", image_size, augment=False),
    }
    names = {key: set(value.filenames) for key, value in datasets.items()}
    if names["train"] & names["val"] or names["train"] & names["test"] or names["val"] & names["test"]:
        raise ValueError("BUSI fixed splits are not disjoint")
    return datasets


def main(args):
    if args.image_size != 224:
        raise ValueError("Use image_size=224 to preserve the completed U-Net setup")
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")

    pipeline.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    datasets = split_datasets(args.image_size)
    for split, dataset in datasets.items():
        print(f"{split.capitalize()} samples:", len(dataset))

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    model = pipeline.UNet(n_channels=3, n_classes=1).to(device)
    parameter_count, flops = pipeline.model_complexity(model, args.image_size)
    print("Model parameters:", parameter_count)
    print("FLOPs per image:", flops, "(convolutional multiply-add = 2 FLOPs)")
    criterion = pipeline.DiceBCELoss()

    if args.preflight_only:
        images, masks, filenames = next(iter(loaders["train"]))
        images, masks = images.to(device), masks.to(device)
        predictions = model(images)
        loss = criterion(predictions, masks)
        loss.backward()
        model.zero_grad(set_to_none=True)
        print("Preflight filenames:", list(filenames))
        print("Preflight image shape:", list(images.shape))
        print("Preflight mask shape:", list(masks.shape))
        print("Preflight mask values:", sorted(torch.unique(masks).tolist()))
        print("Preflight output shape:", list(predictions.shape))
        print("Preflight loss:", float(loss.detach()))
        print("Preflight forward/backward: PASS")
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_dice = -1.0
    best_path = RESULT_DIR / ("best_unet_busi_smoke.pth" if args.epochs == 1 else "best_unet_busi.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = pipeline.train_one_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        val_metrics = pipeline.evaluate(
            model, loaders["val"], criterion, device, threshold=args.threshold
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | Val mIoU: {val_metrics['miou']:.4f}"
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
                best_path,
            )

    if args.epochs == 1:
        metrics = {
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
            "checkpoint": str(best_path),
        }
        metrics_path = RESULT_DIR / "smoke_metrics.json"
    else:
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_val = pipeline.evaluate(
            model, loaders["val"], criterion, device, args.threshold, True, args.surface_chunk_size
        )
        test = pipeline.evaluate(
            model, loaders["test"], criterion, device, args.threshold, True, args.surface_chunk_size
        )
        metrics = {
            "epochs_completed": args.epochs,
            "best_epoch": checkpoint["epoch"],
            "validation_dice": final_val["dice"],
            "validation_miou": final_val["miou"],
            "validation_hd95_pixels": final_val["hd95_pixels"],
            "validation_assd_pixels": final_val["assd_pixels"],
            "test_dice": test["dice"],
            "test_miou": test["miou"],
            "test_hd95_pixels": test["hd95_pixels"],
            "test_assd_pixels": test["assd_pixels"],
            "model_parameters": parameter_count,
            "flops_per_image": flops,
            "flops_convention": "convolutional multiply-add = 2 FLOPs",
            "image_size": args.image_size,
            "threshold": args.threshold,
            "checkpoint": str(best_path),
        }
        metrics_path = RESULT_DIR / "final_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)
        output.write("\n")
    print("Checkpoint:", best_path)
    print("Metrics JSON:", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--surface_chunk_size", type=int, default=2048)
    parser.add_argument("--preflight_only", action="store_true")
    main(parser.parse_args())
