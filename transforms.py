from __future__ import annotations

import io
import random
from typing import Callable

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class KeepRatioResizePad:
    def __init__(self, size: int = 224, fill: tuple[int, int, int] = (255, 255, 255)) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        scale = self.size / max(width, height)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(new_size, Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - new_size[0]) // 2
        top = (self.size - new_size[1]) // 2
        canvas.paste(image, (left, top))
        return canvas


class RandomJpegCompression:
    def __init__(self, p: float = 0.25, quality_range: tuple[int, int] = (55, 95)) -> None:
        self.p = p
        self.quality_range = quality_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(*self.quality_range))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomGaussianNoise:
    def __init__(self, p: float = 0.15, std: float = 0.015) -> None:
        self.p = p
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


class RandomBackgroundPad:
    def __init__(self, size: int = 224) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        fill_value = random.randint(235, 255)
        return KeepRatioResizePad(self.size, (fill_value, fill_value, fill_value))(image)


def _common_tail() -> list[Callable]:
    return [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def get_transform(split: str, label: str | None = None, image_size: int = 224, aug_strength: str = "default") -> Callable:
    if split != "train":
        return transforms.Compose([KeepRatioResizePad(image_size), *_common_tail()])

    if label == "food":
        if aug_strength == "light":
            return transforms.Compose(
                [
                    KeepRatioResizePad(image_size),
                    transforms.RandomHorizontalFlip(p=0.35),
                    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
                    RandomJpegCompression(p=0.12, quality_range=(70, 96)),
                    *_common_tail(),
                ]
            )
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.75, 1.0),
                    ratio=(0.8, 1.25),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8, fill=(255, 255, 255)),
                transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.03),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.12),
                RandomJpegCompression(p=0.25),
                transforms.ToTensor(),
                RandomGaussianNoise(p=0.15),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    if label == "chart":
        return transforms.Compose(
            [
                RandomBackgroundPad(image_size),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.94, 1.04), fill=(255, 255, 255)),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.10),
                RandomJpegCompression(p=0.30, quality_range=(60, 96)),
                *_common_tail(),
            ]
        )

    return transforms.Compose(
        [
            KeepRatioResizePad(image_size),
            ImageOps.exif_transpose,
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            RandomJpegCompression(p=0.20),
            *_common_tail(),
        ]
    )
