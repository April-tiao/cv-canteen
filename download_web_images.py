from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, TypeVar

from PIL import Image, ImageOps


QUERY_PRESETS = {
    "food": [
        "prepared food",
        "restaurant dish",
        "Chinese food",
        "rice dish",
        "noodle dish",
        "soup food",
        "fruit plate",
        "dessert food",
        "snack food",
        "takeaway food",
        "home cooked food",
        "bowl of food",
        "plate of food",
        "vegetable dish",
        "meat dish",
        "drink beverage",
    ],
    "other": [
        "landscape photograph",
        "street scene",
        "building exterior",
        "office interior",
        "furniture photograph",
        "vehicle photograph",
        "electronic device",
        "clothing product",
        "plant photograph",
        "sports scene",
        "tool photograph",
        "toy photograph",
        "empty plate",
        "empty bowl",
        "kitchen utensil",
        "restaurant interior",
        "supermarket shelf",
        "menu board",
        "document page",
        "form screenshot",
        "table document",
        "map diagram",
        "QR code",
        "barcode",
    ],
}

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "canteen-food-chart-classifier/1.0 (dataset preparation)"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
T = TypeVar("T")


def retry_call(fn: Callable[[], T], retries: int, sleep: float, label: str) -> T:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            wait = sleep * attempt
            print(f"{label} failed attempt {attempt}/{retries}: {exc}; retry in {wait:.1f}s")
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def request_json(url: str, timeout: int = 60) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def commons_search(query: str, limit: int, offset: int = 0, timeout: int = 60) -> list[str]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": min(50, limit),
        "gsroffset": offset,
        "prop": "imageinfo",
        "iiprop": "url|mime",
    }
    data = request_json(f"{API_URL}?{urllib.parse.urlencode(params)}", timeout=timeout)
    pages = data.get("query", {}).get("pages", {})
    urls = []
    for page in pages.values():
        info = page.get("imageinfo", [{}])[0]
        image_url = info.get("url")
        mime = info.get("mime", "")
        if not image_url or not mime.startswith("image/"):
            continue
        suffix = Path(urllib.parse.urlparse(image_url).path).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            urls.append(image_url)
    return urls


def download_bytes(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def count_images(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"})


def existing_hashes(path: Path) -> set[str]:
    hashes = set()
    for item in path.rglob("*"):
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            hashes.add(hashlib.sha1(item.read_bytes()).hexdigest())
    return hashes


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_image(content: bytes, output_dir: Path, prefix: str, min_size: int, index: int) -> tuple[Path | None, str | None]:
    digest = hashlib.sha1(content).hexdigest()
    try:
        image = Image.open(io.BytesIO(content))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        return None, f"open_failed:{exc}"

    width, height = image.size
    if min(width, height) < min_size:
        return None, f"too_small:{width}x{height}"

    output_path = output_dir / f"{prefix}_{index:05d}_{digest[:12]}.jpg"
    image.save(output_path, format="JPEG", quality=92, optimize=True)
    return output_path, None


def resolve_queries(preset: str, query_file: str | None) -> list[str]:
    queries = list(QUERY_PRESETS[preset])
    if query_file:
        with open(query_file, "r", encoding="utf-8") as file:
            custom = [line.strip() for line in file if line.strip() and not line.strip().startswith("#")]
        if custom:
            queries = custom
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public images from Wikimedia Commons.")
    parser.add_argument("--preset", choices=sorted(QUERY_PRESETS), default="food")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-total", type=int, default=1000, help="Target count including --existing-count and downloaded files.")
    parser.add_argument("--existing-count", type=int, default=0, help="Already available class images outside output dir.")
    parser.add_argument("--max-download", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=224)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--query-limit", type=int, default=200)
    parser.add_argument("--query-file", default=None, help="Optional UTF-8 text file, one Wikimedia search query per line.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or ("images/网上食物" if args.preset == "food" else "images/网上其他"))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "download_manifest.json"
    manifest = load_manifest(manifest_path)
    seen_urls = {row["url"] for row in manifest if "url" in row}
    hashes = existing_hashes(output_dir)

    downloaded = count_images(output_dir)
    need = max(0, args.target_total - args.existing_count - downloaded)
    if args.max_download is not None:
        need = min(need, args.max_download)
    print(f"preset={args.preset} existing_count={args.existing_count} downloaded={downloaded} need_download={need}")
    if need <= 0:
        return

    queries = resolve_queries(args.preset, args.query_file)
    saved = 0
    checked = 0
    for query in queries:
        for offset in range(0, args.query_limit, 50):
            if saved >= need:
                break
            try:
                urls = retry_call(
                    lambda: commons_search(
                        query,
                        limit=min(50, args.query_limit - offset),
                        offset=offset,
                        timeout=args.timeout,
                    ),
                    retries=args.retries,
                    sleep=args.retry_sleep,
                    label=f"search query={query!r} offset={offset}",
                )
            except Exception as exc:
                print(f"search_failed query={query!r} offset={offset}: {exc}")
                continue

            for url in urls:
                if saved >= need:
                    break
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                checked += 1
                try:
                    content = retry_call(
                        lambda: download_bytes(url, timeout=args.timeout),
                        retries=args.retries,
                        sleep=args.retry_sleep,
                        label=f"download url={url}",
                    )
                except Exception as exc:
                    print(f"download_failed {url}: {exc}")
                    continue

                digest = hashlib.sha1(content).hexdigest()
                if digest in hashes:
                    continue

                output_path, error = save_image(content, output_dir, args.preset, args.min_size, downloaded + saved)
                manifest.append(
                    {
                        "url": url,
                        "query": query,
                        "saved_path": str(output_path) if output_path else None,
                        "error": error,
                    }
                )
                if output_path:
                    hashes.add(digest)
                    saved += 1
                    print(f"saved {saved}/{need}: {output_path}")
                time.sleep(args.sleep)

            with open(manifest_path, "w", encoding="utf-8") as file:
                json.dump(manifest, file, ensure_ascii=False, indent=2)
        if saved >= need:
            break

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"checked={checked} saved={saved} output_dir={output_dir}")


if __name__ == "__main__":
    main()
