from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import datasets


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Avoid obvious food/food-container classes when using generic public datasets.
FOOD_LIKE = {
    "apple",
    "baby_food",
    "bowl",
    "can",
    "cup",
    "mushroom",
    "orange",
    "pear",
    "plate",
    "sweet_pepper",
}


@dataclass(frozen=True)
class SourceSpec:
    group: str
    name: str
    target: int
    kind: str


DEFAULT_SOURCES = [
    # Natural objects. CIFAR is low-res but useful as broad negative coverage.
    SourceSpec("natural_object", "cifar100", 1600, "torchvision"),
    SourceSpec("natural_object", "cifar10", 500, "torchvision"),
    SourceSpec("natural_object", "oxford_pets", 600, "torchvision"),
    SourceSpec("natural_object", "flowers102", 800, "torchvision"),
    SourceSpec("natural_object", "caltech101", 1000, "torchvision"),
    SourceSpec("natural_object", "caltech256", 1000, "torchvision"),
    SourceSpec("natural_object", "hf_caltech101", 1000, "huggingface"),
    SourceSpec("natural_object", "hf_caltech256", 1000, "huggingface"),
    SourceSpec("natural_object", "svhn", 1000, "torchvision"),
    SourceSpec("natural_object", "gtsrb", 1000, "torchvision"),
    # Scenes/textures, closer to real background and non-object images.
    SourceSpec("scene_texture", "eurosat", 1200, "torchvision"),
    SourceSpec("scene_texture", "dtd", 1300, "torchvision"),
    SourceSpec("scene_texture", "sun397", 1000, "torchvision"),
    SourceSpec("scene_texture", "hf_sun397", 1000, "huggingface"),
    # Documents. Hugging Face sources are optional; if `datasets` is missing or the
    # server cannot access HF, the script prints the failure and continues.
    SourceSpec("document", "aharley/rvl_cdip", 1200, "huggingface"),
    SourceSpec("document", "chainyo/rvl-cdip", 1000, "huggingface"),
    SourceSpec("document", "psyche/publaynet", 800, "huggingface"),
    SourceSpec("document", "docling-project/DocLayNet", 500, "huggingface"),
    # These are best supplied from local collected hard negatives.
    SourceSpec("food_related", "local_food_related", 1500, "local"),
    SourceSpec("chart_related", "local_chart_related", 1500, "local"),
]


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))[:96]


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def image_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    return hashlib.sha1(rgb.tobytes()).hexdigest()


def normalize_image(image: Image.Image, min_size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if min(width, height) < min_size:
        scale = min_size / min(width, height)
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    return image


def save_image(image: Image.Image, out_dir: Path, prefix: str, index: int, min_size: int) -> Path:
    image = normalize_image(image, min_size)
    digest = image_digest(image)[:12]
    path = out_dir / f"{safe_name(prefix)}_{index:06d}_{digest}.jpg"
    image.save(path, format="JPEG", quality=92, optimize=True)
    return path


def get_torchvision_dataset(name: str, cache_dir: Path, split: str):
    name = name.lower()
    if name == "cifar10":
        return datasets.CIFAR10(root=cache_dir, train=(split != "test"), download=True)
    if name == "cifar100":
        return datasets.CIFAR100(root=cache_dir, train=(split != "test"), download=True)
    if name == "oxford_pets":
        return datasets.OxfordIIITPet(
            root=cache_dir,
            split=("trainval" if split != "test" else "test"),
            target_types="category",
            download=True,
        )
    if name == "flowers102":
        return datasets.Flowers102(root=cache_dir, split=("train" if split != "test" else "test"), download=True)
    if name == "dtd":
        return datasets.DTD(root=cache_dir, split=("train" if split != "test" else "test"), download=True)
    if name == "eurosat":
        return datasets.EuroSAT(root=cache_dir, download=True)
    if name == "caltech101":
        return datasets.Caltech101(root=cache_dir, target_type="category", download=True)
    if name == "caltech256":
        return datasets.Caltech256(root=cache_dir, download=True)
    if name == "sun397":
        return datasets.SUN397(root=cache_dir, download=True)
    if name == "svhn":
        return datasets.SVHN(root=cache_dir, split=("train" if split != "test" else "test"), download=True)
    if name == "gtsrb":
        return datasets.GTSRB(root=cache_dir, split=("train" if split != "test" else "test"), download=True)
    raise ValueError(f"Unsupported TorchVision dataset: {name}")


def label_name(dataset: Any, label: int) -> str:
    classes = getattr(dataset, "classes", None)
    if classes and 0 <= label < len(classes):
        return safe_name(classes[label]).lower()
    categories = getattr(dataset, "_categories", None)
    if categories and 0 <= label < len(categories):
        return safe_name(categories[label]).lower()
    return str(label)


def collect_torchvision_by_label(name: str, cache_dir: Path, split: str, seed: int) -> dict[str, list[Image.Image]]:
    dataset = get_torchvision_dataset(name, cache_dir, split)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    by_label: dict[str, list[Image.Image]] = defaultdict(list)
    for index in indices:
        item = dataset[index]
        image = item[0]
        label = int(item[1])
        label = label_name(dataset, label)
        if label in FOOD_LIKE:
            continue
        by_label[label].append(image)
    return by_label


def hf_image_from_row(row: dict[str, Any]) -> Image.Image | None:
    for key in ("image", "page_image", "png", "jpg"):
        value = row.get(key)
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict) and value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"]))
    return None


def hf_label_from_row(row: dict[str, Any], fallback: str) -> str:
    filename = row.get("filename")
    if isinstance(filename, str) and "/" in filename:
        return safe_name(filename.split("/", 1)[0]).lower()
    for key in ("label", "labels", "category", "doc_category", "class", "type"):
        if key in row and row[key] is not None:
            return safe_name(row[key]).lower()
    return fallback


def hf_dataset_name(name: str) -> str:
    aliases = {
        "hf_caltech101": "flwrlabs/caltech101",
        "hf_caltech256": "ilee0022/Caltech-256",
        "hf_sun397": "tanganke/sun397",
    }
    return aliases.get(name, name)


def collect_huggingface_by_label(name: str, split: str, target: int, seed: int, max_scan: int) -> dict[str, list[Image.Image]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install Hugging Face datasets first: pip install datasets") from exc

    # Streaming keeps huge document datasets from downloading everything.
    dataset = load_dataset(hf_dataset_name(name), split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=2000)
    by_label: dict[str, list[Image.Image]] = defaultdict(list)
    max_per_label = max(25, target // 12)
    scanned = 0
    for row in dataset:
        scanned += 1
        if scanned > max_scan:
            break
        image = hf_image_from_row(row)
        if image is None:
            continue
        label = hf_label_from_row(row, fallback=safe_name(name.split("/")[-1]).lower())
        if len(by_label[label]) >= max_per_label:
            continue
        by_label[label].append(image)
        if sum(len(v) for v in by_label.values()) >= target * 2:
            break
    return by_label


def collect_local_by_label(roots: list[Path], seed: int) -> dict[str, list[Image.Image]]:
    rng = random.Random(seed)
    by_label_paths: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            print(f"local root missing: {root}", flush=True)
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not is_valid_image(path):
                continue
            rel = path.relative_to(root)
            label = rel.parts[0] if len(rel.parts) > 1 else root.name
            by_label_paths[safe_name(label).lower()].append(path)

    by_label: dict[str, list[Image.Image]] = defaultdict(list)
    for label, paths in by_label_paths.items():
        rng.shuffle(paths)
        for path in paths:
            try:
                by_label[label].append(Image.open(path).copy())
            except (OSError, UnidentifiedImageError):
                continue
    return by_label


def balanced_emit(
    by_label: dict[str, list[Image.Image]],
    source: SourceSpec,
    target: int,
    out_dir: Path,
    min_size: int,
    start_index: int,
    seen: set[str],
) -> tuple[int, list[dict]]:
    labels = [label for label, images in by_label.items() if images]
    labels.sort()
    if not labels or target <= 0:
        return 0, []

    # Hard cap prevents one class from dominating when source labels are uneven.
    max_per_label = max(20, target // max(1, len(labels)) + 5)
    per_label_saved = {label: 0 for label in labels}
    cursors = {label: 0 for label in labels}
    saved = 0
    manifest: list[dict] = []

    while saved < target:
        progressed = False
        for label in labels:
            if saved >= target:
                break
            if per_label_saved[label] >= max_per_label:
                continue
            images = by_label[label]
            while cursors[label] < len(images):
                image = normalize_image(images[cursors[label]], min_size)
                cursors[label] += 1
                digest = image_digest(image)
                if digest in seen:
                    continue
                seen.add(digest)
                path = save_image(image, out_dir, f"{source.group}_{safe_name(source.name)}_{label}", start_index + saved, min_size)
                manifest.append(
                    {
                        "group": source.group,
                        "source": source.name,
                        "label": label,
                        "target": str(path),
                    }
                )
                per_label_saved[label] += 1
                saved += 1
                progressed = True
                break
        if not progressed:
            break
    return saved, manifest


def parse_source_overrides(text: str | None) -> dict[str, int]:
    if not text:
        return {}
    result = {}
    for part in text.split(","):
        if not part.strip():
            continue
        name, value = part.split("=", 1)
        result[name.strip()] = int(value)
    return result


def load_existing_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def existing_seen_from_manifest(rows: list[dict]) -> set[str]:
    seen = set()
    for row in rows:
        target = row.get("target")
        if not target:
            continue
        path = Path(target)
        if path.exists():
            try:
                with Image.open(path) as image:
                    seen.add(image_digest(normalize_image(image, 1)))
            except (OSError, UnidentifiedImageError):
                continue
    return seen


def remaining_sources_for_append(sources: list[SourceSpec], existing_rows: list[dict]) -> list[SourceSpec]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in existing_rows:
        counts[(row.get("group", ""), row.get("source", ""))] += 1
    remaining = []
    for source in sources:
        already = counts[(source.group, source.name)]
        left = max(0, source.target - already)
        if left > 0:
            remaining.append(SourceSpec(source.group, source.name, left, source.kind))
    return remaining


def scale_sources_to_total(sources: list[SourceSpec], target_total: int | None) -> list[SourceSpec]:
    if target_total is None:
        return sources
    current = sum(source.target for source in sources)
    if current <= 0 or current == target_total:
        return sources
    scaled = []
    running = 0
    for index, source in enumerate(sources):
        if index == len(sources) - 1:
            target = target_total - running
        else:
            target = round(source.target * target_total / current)
            running += target
        if target > 0:
            scaled.append(SourceSpec(source.group, source.name, target, source.kind))
    return scaled


def build_sources(overrides: dict[str, int]) -> list[SourceSpec]:
    if not overrides:
        return DEFAULT_SOURCES
    by_name = {source.name: source for source in DEFAULT_SOURCES}
    by_group = defaultdict(list)
    for source in DEFAULT_SOURCES:
        by_group[source.group].append(source)

    sources: list[SourceSpec] = []
    for key, target in overrides.items():
        if key in by_name:
            source = by_name[key]
            sources.append(SourceSpec(source.group, source.name, target, source.kind))
            continue
        if key in by_group:
            group_sources = by_group[key]
            base = target // len(group_sources)
            remainder = target % len(group_sources)
            for index, source in enumerate(group_sources):
                sources.append(SourceSpec(source.group, source.name, base + (1 if index < remainder else 0), source.kind))
            continue
        raise ValueError(f"Unknown source or group in --source-counts: {key}")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a category-balanced other dataset from public datasets.")
    parser.add_argument("--output-dir", default="images/网上其他")
    parser.add_argument("--cache-dir", default="public_dataset_cache")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-size", type=int, default=224)
    parser.add_argument("--hf-max-scan", type=int, default=30000)
    parser.add_argument("--target-total", type=int, default=None, help="Scale requested source quotas to this total.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append", action="store_true", help="Keep existing output and fill missing quotas from manifest.")
    parser.add_argument(
        "--source-counts",
        default=None,
        help=(
            "Either group quotas, e.g. natural_object=3500,scene_texture=2500,document=2500,"
            "food_related=750,chart_related=750; or source quotas, e.g. cifar100=1200,eurosat=800."
        ),
    )
    parser.add_argument("--food-related-root", action="append", default=[])
    parser.add_argument("--chart-related-root", action="append", default=[])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    if args.overwrite and args.append:
        raise ValueError("--overwrite and --append cannot be used together")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "balanced_other_manifest.json"
    sources = scale_sources_to_total(build_sources(parse_source_overrides(args.source_counts)), args.target_total)
    existing_manifest = load_existing_manifest(manifest_path) if args.append else []
    if args.append:
        sources = remaining_sources_for_append(sources, existing_manifest)
        print(f"append mode: existing={len(existing_manifest)} remaining_target={sum(s.target for s in sources)}", flush=True)
    seen: set[str] = existing_seen_from_manifest(existing_manifest) if args.append else set()
    manifest: list[dict] = list(existing_manifest)
    total_saved = len(existing_manifest)

    for source in sources:
        if source.target <= 0:
            continue
        print(f"source start group={source.group} name={source.name} target={source.target}", flush=True)
        try:
            if source.kind == "torchvision":
                by_label = collect_torchvision_by_label(source.name, cache_dir, args.split, args.seed)
            elif source.kind == "huggingface":
                by_label = collect_huggingface_by_label(source.name, args.split, source.target, args.seed, args.hf_max_scan)
            elif source.kind == "local":
                roots = [Path(p) for p in (args.food_related_root if source.group == "food_related" else args.chart_related_root)]
                by_label = collect_local_by_label(roots, args.seed)
            else:
                raise ValueError(source.kind)
        except Exception as exc:
            print(f"source failed group={source.group} name={source.name}: {exc}", flush=True)
            continue

        saved, rows = balanced_emit(by_label, source, source.target, out_dir, args.min_size, total_saved, seen)
        total_saved += saved
        manifest.extend(rows)
        print(f"source done group={source.group} name={source.name} saved={saved} total={total_saved}", flush=True)

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    summary = defaultdict(int)
    for row in manifest:
        summary[f"{row['group']}::{row['source']}"] += 1
    print(json.dumps(dict(sorted(summary.items())), ensure_ascii=False, indent=2), flush=True)
    print(f"saved_total={total_saved}", flush=True)
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
