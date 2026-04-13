from __future__ import annotations

"""FastViT-based semantic segmentation training script for RGB, HSV, and H2SV inputs.

This file is intentionally written in a readable top-to-bottom flow:
1. Configuration
2. General utilities
3. Data loading and preprocessing
4. Model building
5. Training and evaluation
6. Visualization, reporting, and Weights & Biases logging
7. Main execution flow

The goal is to make it easy to:
- understand what is happening at each stage,
- change experiment settings without hunting around the file,
- compare RGB, HSV, and H2SV fairly using the same architecture and training setup.
"""

# ==============================================================================
# CONFIGURATION
# ==============================================================================

import argparse
import copy
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


Color = Tuple[int, int, int]


@dataclass
class ExperimentConfig:
    """Main experiment settings kept together so common changes happen in one place."""

    data_dir: Path
    output_dir: Path
    input_mode: str = "rgb"
    backbone_name: str = "fastvit_t8"
    use_pretrained_backbone: bool = False
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 10
    image_size: Tuple[int, int] = (224, 224)
    num_workers: int = 0
    seed: int = 42
    save_num_predictions: int = 10

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.input_mode


def parse_args() -> argparse.Namespace:
    """Expose the key experiment values through the command line."""

    parser = argparse.ArgumentParser(
        description="Train and evaluate FastViT-based semantic segmentation models for RGB, HSV, H2SV, or all."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs_fastvit")
    parser.add_argument("--input-mode", choices=["rgb", "hsv", "h2sv", "all"], default="all")
    parser.add_argument("--backbone-name", default="fastvit_t8")
    parser.add_argument("--use-pretrained-backbone", action="store_true", help="Load pretrained FastViT weights when available.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--image-size", nargs=2, type=int, default=[224, 224], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-num-predictions", type=int, default=10)
    parser.add_argument("--use-wandb", action="store_true", help="Log training, evaluation, and prediction samples to Weights & Biases.")
    parser.add_argument("--wandb-project", default="fastvit_segmentation")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


# ==============================================================================
# GENERAL UTILITIES
# ==============================================================================


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Keep runs as repeatable as possible across Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    """Prefer Apple MPS first on Mac, then CUDA, otherwise fall back to CPU."""

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_device_info(device: torch.device) -> None:
    """Make the hardware choice explicit in the logs."""

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    elif device.type == "mps":
        print("Apple Metal Performance Shaders (MPS) is enabled.")


def count_model_parameters(model: nn.Module) -> int:
    """Count trainable parameters for reporting and comparison."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================


def load_rgb(path: Path) -> np.ndarray:
    """Load an image or mask from disk as an RGB numpy array."""

    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize images with bilinear interpolation to keep them visually smooth."""

    return np.array(Image.fromarray(image).resize(size, resample=Image.Resampling.BILINEAR), dtype=np.uint8)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize masks with nearest-neighbor interpolation so class ids stay discrete."""

    return np.array(Image.fromarray(mask.astype(np.uint8)).resize(size, resample=Image.Resampling.NEAREST), dtype=np.int64)


def horizontal_flip(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same left-right flip to the image and mask."""

    return np.ascontiguousarray(image[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])


def rgb_uint8_to_float01(image: np.ndarray) -> np.ndarray:
    """Convert 0-255 RGB values into 0-1 float values."""

    return image.astype(np.float32) / 255.0


# ------------------------------------------------------------------------------
# Color space conversion
# ------------------------------------------------------------------------------


def rgb_to_hsv_channels(rgb_float: np.ndarray) -> np.ndarray:
    """Convert normalized RGB channels into HSV channels."""

    return rgb_to_hsv(rgb_float).astype(np.float32)


def rgb_to_h2sv_channels(rgb_float: np.ndarray) -> np.ndarray:
    """Convert RGB into four-channel H2SV: (H1, H2, S, V)."""

    hsv = rgb_to_hsv_channels(rgb_float)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    angle = 2.0 * np.pi * hue
    h1 = np.cos(angle).astype(np.float32)
    h2 = np.sin(angle).astype(np.float32)
    return np.stack([h1, h2, saturation, value], axis=-1).astype(np.float32)


def infer_class_name_from_color(color: Color, index: int) -> str:
    """Map known label colors to friendly class names."""

    if color == (0, 0, 0):
        return "background"
    if color == (0, 255, 0):
        return "crop"
    if color == (255, 0, 0):
        return "weed"
    return f"class_{index}"


def find_unique_colors(mask_paths: Sequence[Path]) -> List[Color]:
    """Scan every mask and collect all unique RGB label colors."""

    unique_colors = set()
    for mask_path in mask_paths:
        mask = load_rgb(mask_path)
        colors = np.unique(mask.reshape(-1, 3), axis=0)
        unique_colors.update(tuple(int(value) for value in color) for color in colors)
    return sorted(unique_colors)


def build_class_mapping(mask_paths: Sequence[Path]) -> Dict[Color, int]:
    """Build a color-to-class-index lookup shared by all splits."""

    return {color: index for index, color in enumerate(find_unique_colors(mask_paths))}


def save_class_mapping(mapping: Dict[Color, int], json_path: Path) -> List[str]:
    """Save the discovered class mapping for reproducibility."""

    ensure_dir(json_path.parent)
    ordered_items = sorted(mapping.items(), key=lambda item: item[1])
    class_names = [infer_class_name_from_color(color, index) for color, index in ordered_items]
    payload = {
        "num_classes": len(mapping),
        "classes": [
            {"index": index, "name": infer_class_name_from_color(color, index), "color_rgb": list(color)}
            for color, index in ordered_items
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return class_names


def validate_mapping_consistency(split_to_masks: Dict[str, Sequence[Path]], mapping: Dict[Color, int]) -> None:
    """Check that every split uses the same mask colors."""

    known_colors = set(mapping.keys())
    for split_name, mask_paths in split_to_masks.items():
        split_colors = set(find_unique_colors(mask_paths))
        missing_colors = split_colors - known_colors
        if missing_colors:
            raise ValueError(f"Split '{split_name}' contains unknown mask colors: {sorted(missing_colors)}")


def mask_rgb_to_index(mask_rgb: np.ndarray, mapping: Dict[Color, int]) -> np.ndarray:
    """Convert a color mask into integer class ids."""

    mask = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    unmatched = np.ones(mask_rgb.shape[:2], dtype=bool)
    for color, class_index in mapping.items():
        matches = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        mask[matches] = class_index
        unmatched &= ~matches
    if unmatched.any():
        row, col = np.argwhere(unmatched)[0]
        raise ValueError(f"Unknown color {mask_rgb[row, col].tolist()} found in segmentation mask.")
    return mask


# ------------------------------------------------------------------------------
# Dataset discovery and dataset class
# ------------------------------------------------------------------------------


@dataclass
class SampleRecord:
    """Keep the image path, mask path, and stem together for one sample."""

    image_path: Path
    mask_path: Path
    stem: str


def discover_split_samples(split_dir: Path) -> List[SampleRecord]:
    """Pair each RGB image with its matching segmentation mask inside one split."""

    image_dir = split_dir / "rgb"
    mask_dir = split_dir / "colorCleaned"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

    image_paths = {path.stem: path for path in image_dir.glob("*.png")}
    mask_paths = {path.stem: path for path in mask_dir.glob("*.png")}
    missing_masks = sorted(set(image_paths) - set(mask_paths))
    missing_images = sorted(set(mask_paths) - set(image_paths))
    if missing_masks:
        raise FileNotFoundError(f"Missing masks for stems: {missing_masks}")
    if missing_images:
        raise FileNotFoundError(f"Missing RGB images for stems: {missing_images}")

    return [SampleRecord(image_paths[stem], mask_paths[stem], stem) for stem in sorted(image_paths)]


class SegmentationDataset(Dataset):
    """Load images, convert them to the chosen input representation, and return tensors."""

    def __init__(
        self,
        samples: Sequence[SampleRecord],
        color_to_class: Dict[Color, int],
        input_mode: str,
        image_size: tuple[int, int],
        train: bool,
    ) -> None:
        self.samples = list(samples)
        self.color_to_class = color_to_class
        self.input_mode = input_mode
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def _convert_input(self, image_rgb: np.ndarray) -> np.ndarray:
        """Convert the RGB image into RGB, HSV, or H2SV according to the selected mode."""

        rgb_float = rgb_uint8_to_float01(image_rgb)
        if self.input_mode == "rgb":
            return rgb_float.astype(np.float32)
        if self.input_mode == "hsv":
            return rgb_to_hsv_channels(rgb_float)
        if self.input_mode == "h2sv":
            return rgb_to_h2sv_channels(rgb_float)
        raise ValueError(f"Unsupported input mode: {self.input_mode}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image_rgb = load_rgb(sample.image_path)
        mask_rgb = load_rgb(sample.mask_path)
        mask = mask_rgb_to_index(mask_rgb, self.color_to_class)

        image_rgb = resize_image(image_rgb, self.image_size)
        mask = resize_mask(mask, self.image_size)

        if self.train and torch.rand(1).item() < 0.5:
            image_rgb, mask = horizontal_flip(image_rgb, mask)

        image = np.transpose(self._convert_input(image_rgb), (2, 0, 1))
        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).long(),
            "stem": sample.stem,
        }


def build_datasets(
    data_dir: Path,
    input_mode: str,
    image_size: tuple[int, int],
    color_mapping: Dict[Color, int],
) -> tuple[Dict[str, List[SampleRecord]], Dict[str, SegmentationDataset]]:
    """Build train, validation, and test datasets for one input representation."""

    split_samples = {
        "Train": discover_split_samples(data_dir / "Train"),
        "Validate": discover_split_samples(data_dir / "Validate"),
        "Test": discover_split_samples(data_dir / "Test"),
    }
    datasets = {
        "Train": SegmentationDataset(split_samples["Train"], color_mapping, input_mode, image_size, train=True),
        "Validate": SegmentationDataset(split_samples["Validate"], color_mapping, input_mode, image_size, train=False),
        "Test": SegmentationDataset(split_samples["Test"], color_mapping, input_mode, image_size, train=False),
    }
    return split_samples, datasets


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Create a PyTorch DataLoader with sensible settings for the current hardware."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ==============================================================================
# MODEL BUILDING
# ==============================================================================


# ------------------------------------------------------------------------------
# Decoder building blocks
# ------------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """A small convolution block used repeatedly inside the decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample the current feature map, merge it with a skip feature, and refine the result."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.refine = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)


# ------------------------------------------------------------------------------
# FastViT segmentation model
# ------------------------------------------------------------------------------


class FastViTSegmentationModel(nn.Module):
    """FastViT backbone with a U-Net-like multi-scale decoder for semantic segmentation."""

    def __init__(
        self,
        input_mode: str,
        num_classes: int,
        image_size: tuple[int, int],
        backbone_name: str,
        pretrained: bool,
    ) -> None:
        super().__init__()

        in_channels = 4 if input_mode == "h2sv" else 3
        backbone_kwargs = {
            "features_only": True,
            "in_chans": in_channels,
        }

        try:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, **backbone_kwargs)
        except Exception as exc:
            if pretrained:
                print(f"Could not load pretrained weights for {backbone_name}. Falling back to random initialization: {exc}")
                self.backbone = timm.create_model(backbone_name, pretrained=False, **backbone_kwargs)
            else:
                raise

        encoder_channels = list(self.backbone.feature_info.channels())
        self.encoder_channels = encoder_channels

        # The decoder mirrors a U-Net-style structure:
        # deepest feature -> upsample + merge with stage 3 -> merge with stage 2 -> merge with stage 1.
        self.center = ConvBlock(encoder_channels[3], 256)
        self.decode3 = DecoderBlock(256, encoder_channels[2], 192)
        self.decode2 = DecoderBlock(192, encoder_channels[1], 128)
        self.decode1 = DecoderBlock(128, encoder_channels[0], 96)
        self.final_refine = ConvBlock(96, 64)
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.backbone(x)
        stage1, stage2, stage3, stage4 = features

        x = self.center(stage4)
        x = self.decode3(x, stage3)
        x = self.decode2(x, stage2)
        x = self.decode1(x, stage1)
        x = F.interpolate(x, scale_factor=4.0, mode="bilinear", align_corners=False)
        x = self.final_refine(x)
        logits = self.segmentation_head(x)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


def build_segmentation_model(config: ExperimentConfig, num_classes: int) -> nn.Module:
    """Build the selected FastViT-based segmentation model."""

    return FastViTSegmentationModel(
        input_mode=config.input_mode,
        num_classes=num_classes,
        image_size=config.image_size,
        backbone_name=config.backbone_name,
        pretrained=config.use_pretrained_backbone,
    )


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================


def compute_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Accumulate a confusion matrix so all metrics come from the same counts."""

    predictions = predictions.view(-1).to(torch.int64)
    targets = targets.view(-1).to(torch.int64)
    valid = (targets >= 0) & (targets < num_classes)
    indices = targets[valid] * num_classes + predictions[valid]
    confmat = torch.bincount(indices, minlength=num_classes * num_classes)
    return confmat.reshape(num_classes, num_classes)


def metrics_from_confusion(confmat: np.ndarray, class_names: Sequence[str]) -> Dict[str, object]:
    """Compute standard segmentation metrics from the confusion matrix."""

    tp = np.diag(confmat).astype(np.float64)
    fp = confmat.sum(axis=0) - tp
    fn = confmat.sum(axis=1) - tp
    total = confmat.sum()

    per_class_iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    per_class_precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    per_class_recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    per_class_f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    pixel_accuracy = float(tp.sum() / total) if total > 0 else 0.0

    return {
        "miou": float(per_class_iou.mean()),
        "per_class_iou": {name: float(value) for name, value in zip(class_names, per_class_iou)},
        "f1": float(per_class_f1.mean()),
        "precision": float(per_class_precision.mean()),
        "recall": float(per_class_recall.mean()),
        "pixel_accuracy": pixel_accuracy,
    }


def move_batch_to_device(batch: Dict[str, torch.Tensor | str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Move image and mask tensors onto the selected device."""

    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    return images, masks


def create_autocast_context(device: torch.device):
    """Use autocast only where it is supported and helpful."""

    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return torch.autocast(device_type=device.type, enabled=False)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return the average training loss."""

    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in dataloader:
        images, masks = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with create_autocast_context(device):
            logits = model(images)
            loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
    class_colors: Sequence[Color],
    input_mode: str,
    sample_output_dir: Path | None,
    max_prediction_examples: int,
) -> Dict[str, object]:
    """Evaluate on validation or test data and optionally save prediction examples."""

    model.eval()
    num_classes = len(class_names)
    total_loss = 0.0
    total_items = 0
    total_time = 0.0
    total_images = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    saved_predictions = 0

    for batch in dataloader:
        images, masks = move_batch_to_device(batch, device)

        start_time = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_time = time.perf_counter() - start_time

        loss = criterion(logits, masks)
        predictions = torch.argmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)
        total_time += batch_time
        total_images += images.size(0)
        confusion += compute_confusion_matrix(predictions, masks, num_classes).to(device)

        if sample_output_dir is not None and saved_predictions < max_prediction_examples:
            remaining = max_prediction_examples - saved_predictions
            save_prediction_examples(
                output_dir=sample_output_dir,
                batch_images=images.cpu(),
                batch_targets=masks.cpu(),
                batch_predictions=predictions.cpu(),
                class_colors=class_colors,
                input_mode=input_mode,
                start_index=saved_predictions,
                max_items=min(remaining, images.size(0)),
            )
            saved_predictions += min(remaining, images.size(0))

    confusion_np = confusion.cpu().numpy()
    metrics = metrics_from_confusion(confusion_np, class_names)
    metrics.update(
        {
            "loss": total_loss / max(total_items, 1),
            "confusion_matrix": confusion_np,
            "inference_time_per_image": total_time / max(total_images, 1),
            "images_per_second": total_images / max(total_time, 1e-12),
        }
    )
    return metrics


# ==============================================================================
# VISUALIZATION, REPORTING, AND W&B LOGGING
# ==============================================================================


def tensor_to_rgb_image(tensor: torch.Tensor, input_mode: str) -> np.ndarray:
    """Convert model inputs back into something readable for saved examples."""

    array = tensor.detach().cpu().numpy()
    if input_mode == "h2sv":
        visual = np.stack(
            [
                np.clip((array[0] + 1.0) / 2.0, 0.0, 1.0),
                np.clip((array[1] + 1.0) / 2.0, 0.0, 1.0),
                np.clip(array[3], 0.0, 1.0),
            ],
            axis=-1,
        )
        return (visual * 255).astype(np.uint8)
    visual = np.transpose(np.clip(array, 0.0, 1.0), (1, 2, 0))
    return (visual * 255).astype(np.uint8)


def decode_class_mask(mask: np.ndarray, class_colors: Sequence[Color]) -> np.ndarray:
    """Convert class ids back into a color mask for visualization."""

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for index, color in enumerate(class_colors):
        color_mask[mask == index] = np.array(color, dtype=np.uint8)
    return color_mask


def save_prediction_examples(
    output_dir: Path,
    batch_images: torch.Tensor,
    batch_targets: torch.Tensor,
    batch_predictions: torch.Tensor,
    class_colors: Sequence[Color],
    input_mode: str,
    start_index: int,
    max_items: int,
) -> None:
    """Save input, ground truth, and prediction panels for qualitative review."""

    ensure_dir(output_dir)
    max_items = min(max_items, batch_images.shape[0])
    for index in range(max_items):
        image = tensor_to_rgb_image(batch_images[index], input_mode)
        target = decode_class_mask(batch_targets[index].cpu().numpy(), class_colors)
        prediction = decode_class_mask(batch_predictions[index].cpu().numpy(), class_colors)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image)
        axes[0].set_title("Input")
        axes[1].imshow(target)
        axes[1].set_title("Target")
        axes[2].imshow(prediction)
        axes[2].set_title("Prediction")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"sample_{start_index + index:03d}.png", dpi=200)
        plt.close(fig)


def save_metrics_csv(rows: List[Dict[str, object]], path: Path) -> None:
    """Write per-epoch metrics to CSV for later review."""

    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_confusion_matrix_csv(confmat: np.ndarray, class_names: Sequence[str], path: Path) -> None:
    """Save the final confusion matrix as a CSV table."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, confmat.tolist()):
            writer.writerow([class_name, *row])


def save_confusion_matrix_plot(confmat: np.ndarray, class_names: Sequence[str], path: Path) -> None:
    """Create a simple confusion matrix heatmap."""

    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confmat, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix")
    for row in range(confmat.shape[0]):
        for col in range(confmat.shape[1]):
            ax.text(col, row, int(confmat[row, col]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training_curves(history_rows: List[Dict[str, object]], output_dir: Path, input_mode: str) -> None:
    """Plot the core training and validation curves."""

    ensure_dir(output_dir)
    if not history_rows:
        return

    epochs = [int(row["epoch"]) for row in history_rows]
    plots = [
        ("train_loss", "Train Loss", "train_loss_vs_epoch.png"),
        ("val_loss", "Validation Loss", "val_loss_vs_epoch.png"),
        ("val_miou", "Validation mIoU", "val_miou_vs_epoch.png"),
        ("val_f1", "Validation F1", "val_f1_vs_epoch.png"),
    ]
    for key, label, filename in plots:
        values = [float(row[key]) for row in history_rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, values, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{input_mode.upper()} {label}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Save a nested dictionary as JSON."""

    ensure_dir(path.parent)

    def _to_jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: _to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(item) for item in value]
        return value

    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def save_summary_report(
    path: Path,
    config: ExperimentConfig,
    dataset_sizes: Dict[str, int],
    class_names: Sequence[str],
    best_epoch: int,
    best_val_metrics: Dict[str, object],
    test_metrics: Dict[str, object],
    parameter_count: int,
) -> None:
    """Write a concise text summary of the completed run."""

    ensure_dir(path.parent)
    lines = [
        "FastViT Semantic Segmentation Summary",
        f"Input mode: {config.input_mode}",
        f"Backbone: {config.backbone_name}",
        f"Use pretrained backbone: {config.use_pretrained_backbone}",
        f"Image size: {config.image_size}",
        f"Best epoch: {best_epoch}",
        f"Parameter count: {parameter_count}",
        f"Dataset sizes: {dataset_sizes}",
        f"Classes: {list(class_names)}",
        "",
        "Best validation metrics:",
        json.dumps({k: v for k, v in best_val_metrics.items() if k != 'confusion_matrix'}, indent=2),
        "",
        "Test metrics:",
        json.dumps({k: v for k, v in test_metrics.items() if k != 'confusion_matrix'}, indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def create_wandb_run(config: ExperimentConfig, args: argparse.Namespace, dataset_sizes: Dict[str, int]):
    """Start a W&B run when requested."""

    if not args.use_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed in the current environment.")

    if args.wandb_run_name:
        run_name = f"{args.wandb_run_name}_{config.input_mode}"
    else:
        run_name = f"{config.input_mode}_{config.backbone_name}"

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        config={
            "data_dir": str(config.data_dir),
            "output_dir": str(config.output_dir),
            "input_mode": config.input_mode,
            "backbone_name": config.backbone_name,
            "use_pretrained_backbone": config.use_pretrained_backbone,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "max_epochs": config.max_epochs,
            "image_width": config.image_size[0],
            "image_height": config.image_size[1],
            "num_workers": config.num_workers,
            "seed": config.seed,
            "save_num_predictions": config.save_num_predictions,
            "dataset_sizes": dataset_sizes,
        },
        reinit=True,
    )


def log_prediction_images_to_wandb(wandb_run, prediction_dir: Path, split_name: str, max_items: int) -> None:
    """Log saved prediction panels to W&B."""

    if wandb_run is None or wandb is None:
        return
    image_paths = sorted(prediction_dir.glob("sample_*.png"))[:max_items]
    if not image_paths:
        return
    wandb_run.log(
        {
            f"{split_name}/predictions": [
                wandb.Image(str(image_path), caption=image_path.name)
                for image_path in image_paths
            ]
        }
    )


# ==============================================================================
# MAIN TRAINING FLOW
# ==============================================================================


def train_single_experiment(
    config: ExperimentConfig,
    datasets: Dict[str, SegmentationDataset],
    class_names: Sequence[str],
    class_colors: Sequence[Color],
    device: torch.device,
    wandb_run,
) -> Dict[str, object]:
    """Train one FastViT model and evaluate the best checkpoint."""

    run_dir = ensure_dir(config.run_dir)
    ensure_dir(run_dir / "models")
    ensure_dir(run_dir / "plots")
    ensure_dir(run_dir / "predictions" / "val")
    ensure_dir(run_dir / "predictions" / "test")

    train_loader = create_dataloader(datasets["Train"], config.batch_size, True, config.num_workers)
    val_loader = create_dataloader(datasets["Validate"], config.batch_size, False, config.num_workers)
    test_loader = create_dataloader(datasets["Test"], config.batch_size, False, config.num_workers)

    model = build_segmentation_model(config, num_classes=len(class_names)).to(device)
    parameter_count = count_model_parameters(model)
    print(f"[{config.input_mode}] Model parameters: {parameter_count:,}")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history_rows: List[Dict[str, object]] = []
    best_state = None
    best_epoch = -1
    best_val_miou = -1.0
    total_start = time.perf_counter()
    epoch_times: List[float] = []
    peak_memory_mb = 0.0

    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        if device.type == "cuda":
            peak_memory_mb = max(peak_memory_mb, torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            torch.cuda.reset_peak_memory_stats(device)

        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            class_colors=class_colors,
            input_mode=config.input_mode,
            sample_output_dir=(run_dir / "predictions" / "val") if epoch == config.max_epochs else None,
            max_prediction_examples=config.save_num_predictions,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_miou": val_metrics["miou"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        for class_name, class_iou in val_metrics["per_class_iou"].items():
            row[f"val_iou_{class_name}"] = class_iou
        history_rows.append(row)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_metrics["loss"],
                    "val/miou": val_metrics["miou"],
                    "val/f1": val_metrics["f1"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                    "val/pixel_accuracy": val_metrics["pixel_accuracy"],
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "timing/epoch_time_sec": epoch_time,
                    **{f"val/per_class_iou/{key}": value for key, value in val_metrics["per_class_iou"].items()},
                }
            )

        print(
            f"[{config.input_mode}] Epoch {epoch:02d}/{config.max_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_mIoU={val_metrics['miou']:.4f} time={epoch_time:.2f}s"
        )

        if val_metrics["miou"] > best_val_miou:
            best_val_miou = float(val_metrics["miou"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, run_dir / "models" / "best_model.pt")

    total_time = time.perf_counter() - total_start
    torch.save(model.state_dict(), run_dir / "models" / "final_model.pt")

    if best_state is None:
        raise RuntimeError("No best model state was captured during training.")

    model.load_state_dict(best_state)
    best_val_metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        class_colors=class_colors,
        input_mode=config.input_mode,
        sample_output_dir=None,
        max_prediction_examples=config.save_num_predictions,
    )
    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        class_colors=class_colors,
        input_mode=config.input_mode,
        sample_output_dir=run_dir / "predictions" / "test",
        max_prediction_examples=config.save_num_predictions,
    )

    if wandb_run is not None:
        wandb_run.log(
            {
                "best_epoch": best_epoch,
                "test/loss": test_metrics["loss"],
                "test/miou": test_metrics["miou"],
                "test/f1": test_metrics["f1"],
                "test/precision": test_metrics["precision"],
                "test/recall": test_metrics["recall"],
                "test/pixel_accuracy": test_metrics["pixel_accuracy"],
                "timing/total_time_sec": total_time,
                "timing/epoch_time_sec_mean": float(np.mean(epoch_times)) if epoch_times else 0.0,
                "timing/inference_time_per_image": test_metrics["inference_time_per_image"],
                "timing/images_per_second": test_metrics["images_per_second"],
                **{f"test/per_class_iou/{key}": value for key, value in test_metrics["per_class_iou"].items()},
            }
        )
        log_prediction_images_to_wandb(wandb_run, run_dir / "predictions" / "val", "val", config.save_num_predictions)
        log_prediction_images_to_wandb(wandb_run, run_dir / "predictions" / "test", "test", config.save_num_predictions)

    return {
        "run_dir": run_dir,
        "history_rows": history_rows,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "parameter_count": parameter_count,
        "total_time_sec": total_time,
        "epoch_time_sec_mean": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "peak_gpu_memory_mb": peak_memory_mb,
    }


def run_single_mode(
    args: argparse.Namespace,
    input_mode: str,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Run the full segmentation pipeline for one input mode."""

    set_seed(args.seed)
    print_device_info(device)

    split_samples = {
        "Train": discover_split_samples(data_dir / "Train"),
        "Validate": discover_split_samples(data_dir / "Validate"),
        "Test": discover_split_samples(data_dir / "Test"),
    }
    mask_paths = {split: [sample.mask_path for sample in samples] for split, samples in split_samples.items()}
    color_mapping = build_class_mapping(mask_paths["Train"] + mask_paths["Validate"] + mask_paths["Test"])
    class_names = save_class_mapping(color_mapping, output_dir / "class_mapping.json")
    validate_mapping_consistency(mask_paths, color_mapping)
    class_colors = [color for color, _ in sorted(color_mapping.items(), key=lambda item: item[1])]
    dataset_sizes = {split: len(samples) for split, samples in split_samples.items()}

    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Running mode: {input_mode.upper()}")

    config = ExperimentConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        input_mode=input_mode,
        backbone_name=args.backbone_name,
        use_pretrained_backbone=args.use_pretrained_backbone,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        image_size=(args.image_size[0], args.image_size[1]),
        num_workers=args.num_workers,
        seed=args.seed,
        save_num_predictions=args.save_num_predictions,
    )

    wandb_run = create_wandb_run(config, args, dataset_sizes)
    _, datasets = build_datasets(data_dir, input_mode, config.image_size, color_mapping)
    result = train_single_experiment(
        config=config,
        datasets=datasets,
        class_names=class_names,
        class_colors=class_colors,
        device=device,
        wandb_run=wandb_run,
    )

    run_dir = result["run_dir"]
    save_metrics_csv(result["history_rows"], run_dir / "metrics_per_epoch.csv")
    save_confusion_matrix_csv(result["test_metrics"]["confusion_matrix"], class_names, run_dir / "test_confusion_matrix.csv")
    save_confusion_matrix_plot(result["test_metrics"]["confusion_matrix"], class_names, run_dir / "test_confusion_matrix.png")
    plot_training_curves(result["history_rows"], run_dir / "plots", input_mode)
    save_summary_report(
        path=run_dir / "summary_report.txt",
        config=config,
        dataset_sizes=dataset_sizes,
        class_names=class_names,
        best_epoch=result["best_epoch"],
        best_val_metrics=result["best_val_metrics"],
        test_metrics=result["test_metrics"],
        parameter_count=result["parameter_count"],
    )
    save_json(
        run_dir / "results_summary.json",
        {
            "config": asdict(config),
            "dataset_sizes": dataset_sizes,
            "class_names": class_names,
            "best_epoch": result["best_epoch"],
            "parameter_count": result["parameter_count"],
            "best_val_metrics": result["best_val_metrics"],
            "test_metrics": result["test_metrics"],
            "epoch_time_sec_mean": result["epoch_time_sec_mean"],
            "total_time_sec": result["total_time_sec"],
            "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
        },
    )

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "best_epoch": result["best_epoch"],
                "parameter_count": result["parameter_count"],
                "final_test_miou": result["test_metrics"]["miou"],
            }
        )
        wandb_run.finish()


def main() -> None:
    """Parse arguments and run one or more input modes."""

    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = ensure_dir(args.output_dir.resolve())
    device = select_device()

    modes = ["rgb", "hsv", "h2sv"] if args.input_mode == "all" else [args.input_mode]
    for input_mode in modes:
        run_single_mode(args=args, input_mode=input_mode, data_dir=data_dir, output_dir=output_dir, device=device)


if __name__ == "__main__":
    main()
