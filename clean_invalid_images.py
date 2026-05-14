from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Find or quarantine invalid image files.")
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--quarantine-dir", default="invalid_images")
    parser.add_argument("--move", action="store_true", help="Move invalid images to quarantine-dir.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    quarantine_dir = Path(args.quarantine_dir).resolve()
    bad: list[Path] = []
    total = 0
    for path in iter_images(root):
        total += 1
        if not is_valid_image(path):
            bad.append(path)
            print(f"invalid: {path}", flush=True)
            if args.move:
                rel = path.relative_to(root)
                target = quarantine_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target))

    print(f"checked={total} invalid={len(bad)}", flush=True)
    if args.move:
        print(f"quarantine_dir={quarantine_dir}", flush=True)


if __name__ == "__main__":
    main()
