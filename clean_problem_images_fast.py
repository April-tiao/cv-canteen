from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def inspect_image_header(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError) as exc:
        return False, f"unreadable:{exc}"
    pixels = width * height
    return True, f"{width}x{height}:{pixels}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast header-only cleanup for unreadable or huge images.")
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--quarantine-dir", default="invalid_images/fast")
    parser.add_argument("--max-pixels", type=int, default=50_000_000)
    parser.add_argument("--move", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    quarantine_dir = Path(args.quarantine_dir).resolve()
    total = 0
    bad = 0
    huge = 0
    for path in iter_images(root):
        total += 1
        ok, info = inspect_image_header(path)
        should_move = False
        reason = ""
        if not ok:
            bad += 1
            should_move = True
            reason = info
        else:
            pixels = int(info.rsplit(":", 1)[-1])
            if pixels > args.max_pixels:
                huge += 1
                should_move = True
                reason = f"huge:{info}"

        if should_move:
            print(f"problem: {path} {reason}", flush=True)
            if args.move:
                rel = path.relative_to(root)
                target = quarantine_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target))

        if total % 1000 == 0:
            print(f"checked={total} bad={bad} huge={huge}", flush=True)

    print(f"done checked={total} bad={bad} huge={huge}", flush=True)
    if args.move:
        print(f"quarantine_dir={quarantine_dir}", flush=True)


if __name__ == "__main__":
    main()
