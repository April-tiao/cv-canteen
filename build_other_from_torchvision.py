from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps
from torchvision import datasets


FOOD_LIKE_CLASS_NAMES = {
    "apple",
    "apples",
    "baby_food",
    "bowl",
    "can",
    "cup",
    "food_container",
    "mushroom",
    "mushrooms",
    "orange",
    "oranges",
    "pear",
    "pears",
    "plate",
    "sweet_pepper",
    "sweet peppers",
}


def sha1_image(image: Image.Image) -> str:
    image = image.convert("RGB")
    return hashlib.sha1(image.tobytes()).hexdigest()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def save_image(image: Image.Image, output_dir: Path, prefix: str, index: int, min_size: int) -> Path | None:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if min(width, height) < min_size:
        scale = min_size / min(width, height)
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    digest = sha1_image(image)[:12]
    output_path = output_dir / f"{prefix}_{index:05d}_{digest}.jpg"
    image.save(output_path, format="JPEG", quality=92, optimize=True)
    return output_path


def get_label_name(dataset, label: int) -> str:
    classes = getattr(dataset, "classes", None)
    if classes and isinstance(label, int) and label < len(classes):
        return str(classes[label])
    categories = getattr(dataset, "_categories", None)
    if categories and isinstance(label, int) and label < len(categories):
        return str(categories[label])
    return str(label)


def iter_dataset_images(dataset, max_items: int | None, seed: int) -> Iterable[tuple[Image.Image, str]]:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    if max_items is not None:
        indices = indices[:max_items]
    for index in indices:
        item = dataset[index]
        image, label = item[0], item[1]
        label_name = get_label_name(dataset, int(label) if isinstance(label, int) else 0)
        if label_name.lower() in FOOD_LIKE_CLASS_NAMES:
            continue
        yield image, label_name


def load_dataset(name: str, root: Path, split: str):
    name = name.lower()
    if name == "cifar10":
        train = split != "test"
        return datasets.CIFAR10(root=root, train=train, download=True)
    if name == "cifar100":
        train = split != "test"
        return datasets.CIFAR100(root=root, train=train, download=True)
    if name == "stl10":
        stl_split = "train" if split != "test" else "test"
        return datasets.STL10(root=root, split=stl_split, download=True)
    if name == "oxford_pets":
        target_types = "category"
        pet_split = "trainval" if split != "test" else "test"
        return datasets.OxfordIIITPet(root=root, split=pet_split, target_types=target_types, download=True)
    if name == "flowers102":
        flower_split = "train" if split == "train" else "val"
        return datasets.Flowers102(root=root, split=flower_split, download=True)
    if name == "dtd":
        dtd_split = "train" if split == "train" else "val"
        return datasets.DTD(root=root, split=dtd_split, download=True)
    if name == "eurosat":
        return datasets.EuroSAT(root=root, download=True)
    raise ValueError(f"Unsupported dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build diverse other images from TorchVision public datasets.")
    parser.add_argument("--output-dir", default="images/网上其他")
    parser.add_argument("--cache-dir", default="public_dataset_cache")
    parser.add_argument("--target-count", type=int, default=1500)
    parser.add_argument("--min-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets",
        default="cifar10,cifar100,eurosat",
        help=(
            "Comma-separated: cifar10,cifar100,eurosat,flowers102,dtd,oxford_pets,stl10. "
            "Default avoids large downloads such as STL10."
        ),
    )
    parser.add_argument("--split", default="train", choices=["train", "test"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    saved = 0
    seen_hashes: set[str] = set()
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]

    for dataset_name in dataset_names:
        if saved >= args.target_count:
            break
        try:
            dataset = load_dataset(dataset_name, cache_dir, args.split)
        except Exception as exc:
            print(f"load_failed dataset={dataset_name}: {exc}")
            continue

        for image, label_name in iter_dataset_images(dataset, max_items=None, seed=args.seed):
            if saved >= args.target_count:
                break
            digest = sha1_image(image)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            prefix = f"{safe_name(dataset_name)}_{safe_name(label_name)}"
            output_path = save_image(image, output_dir, prefix, saved, args.min_size)
            if output_path is None:
                continue
            manifest.append(
                {
                    "dataset": dataset_name,
                    "label": label_name,
                    "target": str(output_path),
                }
            )
            saved += 1
            if saved % 100 == 0:
                print(f"saved {saved}/{args.target_count}")

    manifest_path = output_dir / "torchvision_other_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"saved={saved} output_dir={output_dir}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
