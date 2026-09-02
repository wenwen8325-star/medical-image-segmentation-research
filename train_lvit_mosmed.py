import argparse
import hashlib
import json
import math
import os
import random
import sys
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from transformers import AutoModel, AutoTokenizer


ROOT = Path("/data/leiw/Summer")
MOSMED_ROOT = ROOT / "datasets" / "MosMedDataPlus"
SPLIT_DIR = MOSMED_ROOT / "splits"
RESULT_DIR = ROOT / "results" / "lvit_mosmed"
LVIT_CODE = ROOT / "code" / "LViT"
BERT_CACHE = ROOT / ".cache" / "huggingface"
BERT_HUB_CACHE = BERT_CACHE / "hub"
BERT_MODEL = "google-bert/bert-base-uncased"

TEXT_DIR = MOSMED_ROOT / "text_annotations"
TEXT_FILES = (
    TEXT_DIR / "Train_text_MosMedData.xlsx",
    TEXT_DIR / "Val_text_MosMedData.xlsx",
    TEXT_DIR / "Test_text_MosMedData+.xlsx",
)

sys.path.insert(0, str(LVIT_CODE))
from nets.LViT import LViT  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_index(reference):
    index = 0
    for character in reference:
        if not character.isalpha():
            break
        index = index * 26 + ord(character.upper()) - ord("A") + 1
    return index - 1


def read_xlsx_rows(path):
    """Read the first XLSX sheet without adding an openpyxl dependency."""
    spreadsheet_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relationship_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    namespace = {"s": spreadsheet_ns}
    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.iter(f"{{{spreadsheet_ns}}}t"))
                for item in shared_root.findall("s:si", namespace)
            ]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {node.attrib["Id"]: node.attrib["Target"] for node in relationships}
        sheet = next(iter(workbook.find("s:sheets", namespace)))
        target = targets[sheet.attrib[f"{{{relationship_ns}}}id"]].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        sheet_root = ET.fromstring(archive.read(target))
        rows = []
        for row in sheet_root.findall(".//s:sheetData/s:row", namespace):
            values = {}
            for cell in row.findall("s:c", namespace):
                index = _column_index(cell.attrib.get("r", "A1"))
                cell_type = cell.attrib.get("t")
                value_node = cell.find("s:v", namespace)
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.findall(".//s:t", namespace)
                    )
                elif value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text)]
                else:
                    value = value_node.text or ""
                values[index] = value
            rows.append([values.get(0, ""), values.get(1, "")])
    if not rows or [value.strip() for value in rows[0]] != ["Image", "text"]:
        raise ValueError(f"Expected XLSX columns ['Image', 'text'] in {path}")
    return rows[1:]


def load_descriptions():
    """Combine official workbooks globally; fixed CSVs alone assign splits."""
    descriptions = {}
    for path in TEXT_FILES:
        seen_in_file = set()
        for raw_filename, raw_description in read_xlsx_rows(path):
            filename, description = raw_filename.strip(), raw_description.strip()
            if not filename or not description:
                raise ValueError(f"Blank filename or text in {path}")
            if filename in seen_in_file:
                raise ValueError(f"Duplicate text filename in {path}: {filename}")
            seen_in_file.add(filename)
            if filename in descriptions and descriptions[filename] != description:
                raise ValueError(f"Conflicting descriptions for {filename}")
            descriptions[filename] = description
    return descriptions


def read_split(split):
    frame = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    if "filename" not in frame.columns:
        raise ValueError(f"Missing filename column in {split}.csv")
    filenames = frame["filename"].astype(str).tolist()
    if len(filenames) != len(set(filenames)):
        raise ValueError(f"Duplicate filenames in {split}.csv")
    return filenames


def verify_text_coverage(descriptions, split_names):
    coverage = {}
    for split, filenames in split_names.items():
        missing = [name for name in filenames if name not in descriptions]
        coverage[split] = {
            "samples": len(filenames),
            "with_text": len(filenames) - len(missing),
            "missing": len(missing),
        }
        if missing:
            raise KeyError(f"Missing {len(missing)} {split} descriptions: {missing[:5]}")
    return coverage


def build_file_index():
    extensions = {".png", ".jpg", ".jpeg"}
    images = {path.name: path for path in (MOSMED_ROOT / "frames").iterdir()
              if path.is_file() and path.suffix.lower() in extensions}
    masks = {path.name: path for path in (MOSMED_ROOT / "masks").iterdir()
             if path.is_file() and path.suffix.lower() in extensions}
    return images, masks


def build_text_embeddings(descriptions, device, batch_size):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RESULT_DIR / "bert_text_embeddings.pt"
    source_hashes = {str(path): sha256(path) for path in TEXT_FILES}

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            cached.get("model") == BERT_MODEL
            and cached.get("source_hashes") == source_hashes
            and cached.get("sequence_length") == 10
        ):
            print("Loaded cached BERT text embeddings:", cache_path)
            return cached["embeddings"], cache_path

    unique_descriptions = sorted(set(descriptions.values()))
    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL,
        cache_dir=BERT_HUB_CACHE,
        local_files_only=True,
    )
    bert = AutoModel.from_pretrained(
        BERT_MODEL,
        cache_dir=BERT_HUB_CACHE,
        local_files_only=True,
    ).to(device)
    bert.eval()
    embeddings = {}

    print("Encoding unique descriptions:", len(unique_descriptions))
    with torch.no_grad():
        for start in range(0, len(unique_descriptions), batch_size):
            batch = unique_descriptions[start:start + batch_size]
            tokens = tokenizer(
                batch,
                add_special_tokens=False,
                truncation=True,
                padding="max_length",
                max_length=10,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = bert(**tokens).last_hidden_state
            output = output * tokens["attention_mask"].unsqueeze(-1)

            for description, embedding in zip(batch, output):
                embeddings[description] = embedding.detach().cpu()

    del bert
    if device.type == "cuda":
        torch.cuda.empty_cache()

    torch.save(
        {
            "model": BERT_MODEL,
            "sequence_length": 10,
            "hidden_size": 768,
            "source_hashes": source_hashes,
            "embeddings": embeddings,
        },
        cache_path,
    )
    print("Saved BERT text embeddings:", cache_path)
    return embeddings, cache_path


class MosMedLViTDataset(Dataset):
    def __init__(
        self,
        filenames,
        descriptions,
        text_embeddings,
        image_index,
        mask_index,
        image_size=224,
        augment=False,
    ):
        self.filenames = filenames
        self.descriptions = descriptions
        self.text_embeddings = text_embeddings
        self.image_index = image_index
        self.mask_index = mask_index
        self.image_size = image_size
        self.augment = augment

        missing_images = [name for name in filenames if name not in image_index]
        missing_masks = [name for name in filenames if name not in mask_index]
        if missing_images:
            raise FileNotFoundError(f"Missing images: {missing_images[:5]}")
        if missing_masks:
            raise FileNotFoundError(f"Missing masks: {missing_masks[:5]}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        image = Image.open(self.image_index[filename]).convert("RGB")
        mask = Image.open(self.mask_index[filename]).convert("L")
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        # Match the original LViT augmentation probabilities.
        if self.augment:
            if random.random() > 0.5:
                angle = 90 * random.randint(0, 3)
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
                if random.randint(0, 1) == 0:
                    image, mask = TF.hflip(image), TF.hflip(mask)
                else:
                    image, mask = TF.vflip(image), TF.vflip(mask)
            elif random.random() > 0.5:
                angle = random.randint(-20, 19)
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        image = TF.to_tensor(image)[[2, 1, 0]]  # Original loader uses OpenCV BGR.
        mask = (TF.to_tensor(mask) > 0.5).float()
        text = self.text_embeddings[self.descriptions[filename]].clone().float()

        return image, mask, text, filename


class WeightedBCE(nn.Module):
    def forward(self, prediction, target):
        prediction = prediction.reshape(-1)
        target = target.reshape(-1)
        loss = F.binary_cross_entropy(prediction, target, reduction="none")
        positive = (target > 0.5).float()
        negative = (target < 0.5).float()
        positive_count = positive.sum() + 1e-12
        negative_count = negative.sum() + 1e-12
        return (
            0.5 * positive * loss / positive_count
            + 0.5 * negative * loss / negative_count
        ).sum()


class WeightedDiceLoss(nn.Module):
    def forward(self, prediction, target, smooth=1e-5):
        batch_size = prediction.size(0)
        prediction = prediction.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)
        intersection = (0.5 * prediction * 0.5 * target).sum(dim=1)
        union = (0.5 * prediction).square().sum(dim=1) + (0.5 * target).square().sum(dim=1)
        return (1 - (2 * intersection + smooth) / (union + smooth)).mean()


class WeightedDiceBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = WeightedBCE()
        self.dice = WeightedDiceLoss()

    def forward(self, prediction, target):
        return 0.5 * self.dice(prediction, target) + 0.5 * self.bce(prediction, target)


def segmentation_scores(prediction, target, threshold=0.5, smooth=1e-6):
    prediction = (prediction > threshold).float().flatten(1)
    target = target.float().flatten(1)
    intersection = (prediction * target).sum(dim=1)
    dice = (2 * intersection + smooth) / (prediction.sum(dim=1) + target.sum(dim=1) + smooth)
    union = prediction.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou


def surface_points(mask):
    mask = mask.bool()
    neighbors = F.conv2d(mask[None, None].float(), torch.ones((1, 1, 3, 3), device=mask.device), padding=1)[0, 0]
    return torch.nonzero(mask & (neighbors < 9), as_tuple=False).float()


def directed_surface_distances(source, target, chunk_size=2048):
    return torch.cat([
        torch.cdist(source[start:start + chunk_size], target).min(dim=1).values
        for start in range(0, len(source), chunk_size)
    ])


def surface_distance_scores(prediction, target, threshold=0.5, chunk_size=2048):
    """Match the completed U-Net experiment's HD95/ASSD pixel convention."""
    prediction, target = prediction > threshold, target.bool()
    hd95_values, assd_values = [], []
    for predicted_mask, target_mask in zip(prediction[:, 0], target[:, 0]):
        predicted_empty, target_empty = not predicted_mask.any().item(), not target_mask.any().item()
        if predicted_empty and target_empty:
            hd95_values.append(0.0); assd_values.append(0.0)
            continue
        if predicted_empty or target_empty:
            maximum_distance = math.hypot(predicted_mask.shape[0] - 1, predicted_mask.shape[1] - 1)
            hd95_values.append(maximum_distance); assd_values.append(maximum_distance)
            continue
        predicted_surface, target_surface = surface_points(predicted_mask), surface_points(target_mask)
        distances = torch.cat([
            directed_surface_distances(predicted_surface, target_surface, chunk_size),
            directed_surface_distances(target_surface, predicted_surface, chunk_size),
        ])
        hd95_values.append(torch.quantile(distances, 0.95).item())
        assd_values.append(distances.mean().item())
    return hd95_values, assd_values


def model_complexity(model, image_size, text_length=10, text_width=768):
    """Count parameters and multiply-add FLOPs for one image/text pair."""
    macs, handles = 0, []
    def count_convolution(module, inputs, output):
        nonlocal macs
        kernel_area = math.prod(module.kernel_size)
        if isinstance(module, nn.ConvTranspose2d):
            operations = inputs[0].numel() * (module.out_channels // module.groups) * kernel_area
        else:
            operations = output.numel() * (module.in_channels // module.groups) * kernel_area
        macs += operations + (output.numel() if module.bias is not None else 0)
    def count_linear(module, inputs, output):
        nonlocal macs
        macs += output.numel() * module.in_features + (output.numel() if module.bias is not None else 0)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(module.register_forward_hook(count_convolution))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(count_linear))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    was_training, device = model.training, next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, image_size, image_size, device=device), torch.zeros(1, text_length, text_width, device=device))
    for handle in handles:
        handle.remove()
    model.train(was_training)
    return parameter_count, int(2 * macs)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, masks, text, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images, text)
        loss = criterion(predictions, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold, include_surface_distances=False, surface_chunk_size=2048):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_hd95 = 0.0
    total_assd = 0.0

    for images, masks, text, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
        predictions = model(images, text)
        batch_size = images.size(0)
        total_loss += criterion(predictions, masks).item() * batch_size
        dice, iou = segmentation_scores(predictions, masks, threshold)
        total_dice += dice.sum().item()
        total_iou += iou.sum().item()
        if include_surface_distances:
            hd95_values, assd_values = surface_distance_scores(predictions, masks, threshold, surface_chunk_size)
            total_hd95 += sum(hd95_values)
            total_assd += sum(assd_values)

    samples = len(loader.dataset)
    metrics = {
        "loss": total_loss / samples,
        "dice": total_dice / samples,
        "miou": total_iou / samples,
    }
    if include_surface_distances:
        metrics["hd95_pixels"] = total_hd95 / samples
        metrics["assd_pixels"] = total_assd / samples
    return metrics


def preflight(model, loader, criterion, device):
    images, masks, text, _ = next(iter(loader))
    print("Preflight image shape:", list(images.shape[1:]))
    print("Preflight mask shape:", list(masks.shape[1:]))
    print("Preflight text shape:", list(text.shape[1:]))
    assert list(images.shape[1:]) == [3, 224, 224]
    assert list(masks.shape[1:]) == [1, 224, 224]
    assert list(text.shape[1:]) == [10, 768]

    model.eval()
    images, masks, text = images.to(device), masks.to(device), text.to(device)
    predictions = model(images, text)
    loss = criterion(predictions, masks)
    loss.backward()
    model.zero_grad(set_to_none=True)
    print("Preflight forward/backward: PASS")
    print("Preflight output shape:", list(predictions.shape[1:]))

    return {
        "image": list(images.shape[1:]),
        "mask": list(masks.shape[1:]),
        "text": list(text.shape[1:]),
        "output": list(predictions.shape[1:]),
    }


def main(args):
    if args.image_size != 224:
        raise ValueError("The unchanged LViT architecture is configured for image_size=224")
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.smoke_test and args.epochs != 1:
        raise ValueError("--smoke_test requires --epochs 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    split_names = {split: read_split(split) for split in ("train", "val", "test")}
    descriptions = load_descriptions()
    coverage = verify_text_coverage(descriptions, split_names)
    for split, values in coverage.items():
        print(f"Text coverage {split}: {values['with_text']}/{values['samples']}")

    text_embeddings, embedding_cache_path = build_text_embeddings(
        descriptions,
        device,
        args.bert_batch_size,
    )
    image_index, mask_index = build_file_index()

    train_dataset = MosMedLViTDataset(
        split_names["train"], descriptions, text_embeddings, image_index, mask_index,
        image_size=args.image_size, augment=True,
    )
    val_dataset = MosMedLViTDataset(
        split_names["val"], descriptions, text_embeddings, image_index, mask_index,
        image_size=args.image_size, augment=False,
    )
    test_dataset = MosMedLViTDataset(
        split_names["test"], descriptions, text_embeddings, image_index, mask_index,
        image_size=args.image_size, augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    config = SimpleNamespace(base_channel=64)
    model = LViT(config, n_channels=3, n_classes=1, img_size=224).to(device)
    parameter_count, flops = model_complexity(model, args.image_size)
    print("Model parameters:", parameter_count)
    print("FLOPs per image/text pair:", flops, "(multiply-add = 2 FLOPs)")
    criterion = WeightedDiceBCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        eta_min=1e-4,
    )

    shapes = preflight(model, train_loader, criterion, device)
    if args.preflight_only:
        print("Preflight-only validation finished; no epoch or test evaluation was run.")
        return

    set_seed(args.seed)
    best_dice = -math.inf
    best_val_metrics = None
    best_path = RESULT_DIR / (
        "best_lvit_mosmed_smoke.pth" if args.smoke_test else "best_lvit_mosmed.pth"
    )
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, args.threshold)
        scheduler.step()
        print(
            f"Epoch {epoch:03d}/{args.epochs} | Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | Val Dice: {val_metrics['dice']:.6f} | "
            f"Val mIoU: {val_metrics['miou']:.6f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_val_metrics = dict(val_metrics)
            best_epoch = epoch
            epochs_without_improvement = 0
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_dice": val_metrics["dice"],
                    "val_miou": val_metrics["miou"],
                    "image_size": args.image_size,
                    "threshold": args.threshold,
                    "bert_model": BERT_MODEL,
                },
                best_path,
            )
            print("Saved new best checkpoint:", best_path)
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement > args.early_stopping_patience:
            print("Early stopping triggered.")
            break

    if args.smoke_test:
        smoke_metrics = {
            "epochs_completed": epoch,
            "training_loss": train_loss,
            "validation_dice": val_metrics["dice"],
            "validation_miou": val_metrics["miou"],
            "test_evaluated": False,
            "text_coverage": coverage,
            "tensor_shapes": shapes,
            "checkpoint": str(best_path),
        }
        metrics_path = RESULT_DIR / "smoke_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as output:
            json.dump(smoke_metrics, output, indent=2)
            output.write("\n")
        print("Smoke test finished without test-set evaluation.")
        print("Checkpoint:", best_path)
        print("Metrics JSON:", metrics_path)
        return

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Evaluating the validation-selected best checkpoint on validation and test sets...")
    final_val_metrics = evaluate(model, val_loader, criterion, device, args.threshold, True, args.surface_chunk_size)
    test_metrics = evaluate(model, test_loader, criterion, device, args.threshold, True, args.surface_chunk_size)

    training_metrics = {
        "epochs_requested": args.epochs,
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "training_loss_last_epoch": train_loss,
        "validation_dice": final_val_metrics["dice"],
        "validation_miou": final_val_metrics["miou"],
        "validation_hd95_pixels": final_val_metrics["hd95_pixels"],
        "validation_assd_pixels": final_val_metrics["assd_pixels"],
        "test_dice": test_metrics["dice"],
        "test_miou": test_metrics["miou"],
        "test_hd95_pixels": test_metrics["hd95_pixels"],
        "test_assd_pixels": test_metrics["assd_pixels"],
        "model_parameters": parameter_count,
        "flops_per_image_text_pair": flops,
        "flops_convention": "multiply-add = 2 FLOPs; convolution, transposed convolution, and linear layers",
        "text_coverage": coverage,
        "tensor_shapes": shapes,
        "image_size": args.image_size,
        "threshold": args.threshold,
        "bert_model": BERT_MODEL,
        "text_embedding_cache": str(embedding_cache_path),
        "checkpoint": str(best_path),
    }
    metrics_path = RESULT_DIR / "final_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as output:
        json.dump(training_metrics, output, indent=2)
        output.write("\n")

    print("Training finished.")
    print("Best epoch:", best_epoch)
    print("Best validation Dice:", best_dice)
    print("Checkpoint:", best_path)
    print("Metrics JSON:", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bert_batch_size", type=int, default=64)
    parser.add_argument("--early_stopping_patience", type=int, default=50)
    parser.add_argument("--surface_chunk_size", type=int, default=2048)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    main(parser.parse_args())
