from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


CHART_DIR_NAMES = {
    "数据可视化图表_百度图片搜索",
    "条形图_百度图片搜索",
    "统计图表_百度图片搜索",
    "折线图_百度图片搜索",
    "柱状图_百度图片搜索",
    "合成图表",
}

OTHER_DIR_NAMES = {
    "other",
    "其他",
    "网上其他",
    "web_other",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
CLASS_NAMES = ("food", "chart", "other")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def collect_images(images_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in images_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(images_root)
        top_dir = rel.parts[0] if rel.parts else ""
        if top_dir in CHART_DIR_NAMES:
            label = "chart"
        elif top_dir in OTHER_DIR_NAMES:
            label = "other"
        else:
            label = "food"
        rows.append({"path": path, "label": label, "top_dir": top_dir})
    return rows


def collect_external_images(root: Path, label: str, top_dir: str) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        raise FileNotFoundError(root)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append({"path": path, "label": label, "top_dir": top_dir})
    return rows


def stable_output_name(path: Path, images_root: Path) -> str:
    try:
        rel = path.relative_to(images_root).as_posix()
    except ValueError:
        rel = path.resolve().as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)
    return f"{digest}_{stem}{path.suffix.lower()}"


def split_by_class(rows: list[dict], val_ratio: float, seed: int) -> dict[str, dict[str, list[dict]]]:
    rng = random.Random(seed)
    result = {split: {name: [] for name in CLASS_NAMES} for split in ("train", "val")}
    for label in CLASS_NAMES:
        items = [row for row in rows if row["label"] == label]
        rng.shuffle(items)
        val_count = max(1, round(len(items) * val_ratio)) if items else 0
        result["val"][label] = items[:val_count]
        result["train"][label] = items[val_count:]
    return result


def prepare_output(output_root: Path, overwrite: bool) -> None:
    output_root = output_root.resolve()
    cwd = Path.cwd().resolve()
    if output_root.exists() and overwrite:
        if not is_relative_to(output_root, cwd):
            raise ValueError(f"Refuse to delete output outside workspace: {output_root}")
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        for label in CLASS_NAMES:
            (output_root / split / label).mkdir(parents=True, exist_ok=True)


def materialize_file(src: Path, dst: Path, link: str) -> str:
    if dst.exists():
        try:
            if dst.samefile(src):
                return "exists_same_file"
        except OSError:
            pass
        stem = dst.stem
        suffix = dst.suffix
        parent = dst.parent
        counter = 1
        while dst.exists():
            dst = parent / f"{stem}_{counter}{suffix}"
            counter += 1
    if link == "hardlink":
        try:
            dst.hardlink_to(src)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"
    if link == "symlink":
        try:
            dst.symlink_to(src)
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"
    shutil.copy2(src, dst)
    return "copy"


def copy_split(split_rows: dict[str, dict[str, list[dict]]], images_root: Path, output_root: Path, link: str) -> dict:
    manifest: list[dict] = []
    counts = {split: {label: 0 for label in CLASS_NAMES} for split in ("train", "val")}
    for split, by_label in split_rows.items():
        for label, rows in by_label.items():
            for row in rows:
                src = row["path"]
                dst = output_root / split / label / stable_output_name(src, images_root)
                mode = materialize_file(src, dst, link)
                counts[split][label] += 1
                manifest.append(
                    {
                        "source": str(src),
                        "target": str(dst),
                        "split": split,
                        "label": label,
                        "top_dir": row["top_dir"],
                        "mode": mode,
                    }
                )
    return {"counts": counts, "manifest": manifest}


def dedupe_rows(rows: list[dict]) -> list[dict]:
    deduped = []
    seen: set[str] = set()
    for row in rows:
        key = str(row["path"].resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild food/chart dataset from images directory.")
    parser.add_argument("--images-root", default="images")
    parser.add_argument("--output-root", default="dataset")
    parser.add_argument("--extra-food-root", action="append", default=[])
    parser.add_argument("--extra-chart-root", action="append", default=[])
    parser.add_argument("--extra-other-root", action="append", default=[])
    parser.add_argument("--max-food", type=int, default=None)
    parser.add_argument("--max-chart", type=int, default=None)
    parser.add_argument("--max-other", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--link", choices=["copy", "hardlink", "symlink"], default="copy")
    args = parser.parse_args()

    images_root = Path(args.images_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not images_root.exists():
        raise FileNotFoundError(images_root)
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    rows = collect_images(images_root)
    for root in args.extra_food_root:
        rows.extend(collect_external_images(Path(root).resolve(), "food", Path(root).name))
    for root in args.extra_chart_root:
        rows.extend(collect_external_images(Path(root).resolve(), "chart", Path(root).name))
    for root in args.extra_other_root:
        rows.extend(collect_external_images(Path(root).resolve(), "other", Path(root).name))
    rows = dedupe_rows(rows)

    rng = random.Random(args.seed)
    limited_rows = []
    limits = {"food": args.max_food, "chart": args.max_chart, "other": args.max_other}
    for label in CLASS_NAMES:
        items = [row for row in rows if row["label"] == label]
        rng.shuffle(items)
        limit = limits[label]
        limited_rows.extend(items[:limit] if limit else items)
    rows = limited_rows

    split_rows = split_by_class(rows, args.val_ratio, args.seed)
    prepare_output(output_root, args.overwrite)
    result = copy_split(split_rows, images_root, output_root, args.link)

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(result["manifest"], file, ensure_ascii=False, indent=2)

    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
