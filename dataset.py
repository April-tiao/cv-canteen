from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterator

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset, Sampler

from transforms import get_transform


LABEL_MAP = {
    "food": torch.tensor([1.0, 0.0], dtype=torch.float32),
    "chart": torch.tensor([0.0, 1.0], dtype=torch.float32),
    "other": torch.tensor([0.0, 0.0], dtype=torch.float32),
}
CLASS_NAMES = ("food", "chart", "other")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
DEFAULT_MAX_IMAGE_PIXELS = 50_000_000


class DirectoryImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 224,
        validate_images: bool = False,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        aug_strength: str = "default",
    ) -> None:
        self.root = Path(root) / split
        self.split = split
        self.image_size = image_size
        self.validate_images = validate_images
        self.max_image_pixels = max_image_pixels
        self.aug_strength = aug_strength
        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

    def _collect_samples(self) -> list[dict]:
        samples: list[dict] = []
        for label in CLASS_NAMES:
            label_dir = self.root / label
            if label_dir.exists():
                samples.extend(self._collect_from_dir(label_dir, label, source="real"))

            if label == "chart":
                synthetic_dir = self.root / "chart_synthetic"
                if synthetic_dir.exists():
                    samples.extend(self._collect_from_dir(synthetic_dir, label, source="synthetic"))
        return samples

    def _collect_from_dir(self, path: Path, label: str, source: str) -> list[dict]:
        rows = []
        for image_path in path.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                if self.validate_images and not is_valid_image(image_path):
                    continue
                rows.append({"path": image_path, "label": label, "source": source})
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = None
        for attempt in range(10):
            try:
                image = Image.open(sample["path"])
                width, height = image.size
                if width * height > self.max_image_pixels:
                    raise OSError(f"image too large: {width}x{height}")
                image = ImageOps.exif_transpose(image).convert("RGB")
                break
            except (OSError, UnidentifiedImageError):
                log_bad_image(sample["path"])
                index = (index + attempt + 1) % len(self.samples)
                sample = self.samples[index]
        if image is None:
            raise RuntimeError(f"Failed to read image after retries near: {sample['path']}")
        transform = get_transform(self.split, sample["label"], self.image_size, self.aug_strength)
        return {
            "image": transform(image),
            "target": LABEL_MAP[sample["label"]].clone(),
            "label": sample["label"],
            "path": str(sample["path"]),
            "source": sample["source"],
        }

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in CLASS_NAMES}
        for sample in self.samples:
            counts[sample["label"]] += 1
        return counts


def log_bad_image(path: Path) -> None:
    try:
        with open("bad_images_runtime.log", "a", encoding="utf-8") as file:
            file.write(f"{path}\n")
    except OSError:
        pass


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True
    except (OSError, UnidentifiedImageError):
        return False


class BalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: DirectoryImageDataset,
        class_counts: dict[str, int],
        chart_real_fraction: float = 0.30,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.class_counts = class_counts
        self.chart_real_fraction = chart_real_fraction
        self.seed = seed
        self.epoch = 0
        self.indices_by_key = self._group_indices()
        if batches_per_epoch is None:
            batch_size = sum(class_counts.values())
            batches_per_epoch = math.ceil(len(dataset) / max(1, batch_size))
        self.batches_per_epoch = batches_per_epoch

    def _group_indices(self) -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(self.dataset.samples):
            label = sample["label"]
            grouped[label].append(index)
            if label == "chart":
                grouped[f"chart_{sample['source']}"].append(index)
        return grouped

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            for label, count in self.class_counts.items():
                if count <= 0:
                    continue
                if label == "chart" and self.indices_by_key.get("chart_synthetic") and self.indices_by_key.get("chart_real"):
                    real_count = round(count * self.chart_real_fraction)
                    synthetic_count = count - real_count
                    batch.extend(rng.choices(self.indices_by_key["chart_real"], k=real_count))
                    batch.extend(rng.choices(self.indices_by_key["chart_synthetic"], k=synthetic_count))
                else:
                    indices = self.indices_by_key.get(label, [])
                    if indices:
                        batch.extend(rng.choices(indices, k=count))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch
