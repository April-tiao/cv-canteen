from __future__ import annotations

import argparse
import io
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


CHART_TYPES = ("bar", "line", "pie", "scatter", "area", "histogram", "heatmap")


def color(alpha: int = 255, light: bool = False) -> tuple[int, int, int, int]:
    low, high = (90, 245) if light else (20, 215)
    return (random.randint(low, high), random.randint(low, high), random.randint(low, high), alpha)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def plot_box(width: int, height: int) -> tuple[int, int, int, int]:
    left = random.randint(max(10, width // 25), max(22, width // 8))
    top = random.randint(max(10, height // 25), max(22, height // 8))
    right = width - random.randint(max(10, width // 30), max(18, width // 9))
    bottom = height - random.randint(max(10, height // 30), max(20, height // 8))
    return left, top, right, bottom


def maybe_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], with_grid: bool) -> None:
    left, top, right, bottom = box
    if random.random() < 0.22:
        return
    axis_color = (80, 80, 80, 255)
    draw.line((left, bottom, right, bottom), fill=axis_color, width=random.randint(1, 2))
    draw.line((left, top, left, bottom), fill=axis_color, width=random.randint(1, 2))
    if with_grid:
        grid_color = (190, 190, 190, random.randint(70, 140))
        for i in range(1, random.randint(4, 8)):
            y = top + (bottom - top) * i / 8
            draw.line((left, y, right, y), fill=grid_color, width=1)
        for i in range(1, random.randint(3, 7)):
            x = left + (right - left) * i / 7
            draw.line((x, top, x, bottom), fill=grid_color, width=1)


def maybe_text(draw: ImageDraw.ImageDraw, width: int, height: int, box: tuple[int, int, int, int], with_text: bool) -> None:
    if not with_text:
        return
    left, top, right, bottom = box
    text_color = (45, 45, 45, 255)
    draw.text((left, max(2, top - random.randint(16, 28))), random.choice(["Trend", "Sales", "Count", "Ratio", "Daily", "Q1"]), fill=text_color, font=font(random.randint(10, 15)))
    if random.random() < 0.55:
        draw.text((left, min(height - 14, bottom + 4)), random.choice(["date", "group", "type", "x"]), fill=text_color, font=font(random.randint(8, 11)))
    if random.random() < 0.55:
        draw.text((max(2, left - 24), top), random.choice(["value", "rate", "score"]), fill=text_color, font=font(random.randint(8, 11)))
    if random.random() < 0.25:
        lx, ly = right - random.randint(50, 90), top + random.randint(4, 26)
        for i, name in enumerate(["A", "B", "C"][: random.randint(1, 3)]):
            draw.rectangle((lx, ly + i * 13, lx + 9, ly + 9 + i * 13), fill=color())
            draw.text((lx + 13, ly - 1 + i * 13), name, fill=text_color, font=font(8))


def scale_points(values: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    left, top, right, bottom = box
    values = np.asarray(values, dtype=np.float32)
    values = (values - values.min()) / max(1e-6, values.max() - values.min())
    points = []
    for i, value in enumerate(values):
        x = left + (right - left) * i / max(1, len(values) - 1)
        y = bottom - (bottom - top) * float(value)
        points.append((x, y))
    return points


def draw_bar(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    values = np.random.gamma(2.0, 1.0, random.randint(5, 20))
    values = values / values.max()
    gap = random.uniform(0.15, 0.45)
    slot = (right - left) / len(values)
    for i, value in enumerate(values):
        x0 = left + i * slot + slot * gap / 2
        x1 = left + (i + 1) * slot - slot * gap / 2
        y0 = bottom - (bottom - top) * float(value)
        draw.rectangle((x0, y0, x1, bottom), fill=color(alpha=random.randint(165, 240)))


def draw_line(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], area: bool = False) -> None:
    for _ in range(random.randint(1, 3) if not area else 1):
        values = np.cumsum(np.random.randn(random.randint(8, 26))) + random.uniform(2, 10)
        points = scale_points(values, box)
        if area:
            polygon = [points[0], *points, points[-1], (points[-1][0], box[3]), (points[0][0], box[3])]
            draw.polygon(polygon, fill=color(alpha=random.randint(70, 125)))
        draw.line(points, fill=color(), width=random.randint(1, 4), joint="curve")
        if random.random() < 0.35:
            r = random.randint(2, 4)
            for x, y in points:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color())


def draw_scatter(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    for _ in range(random.randint(30, 130)):
        x = random.uniform(left, right)
        y = random.uniform(top, bottom)
        r = random.uniform(1.5, 5.5)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color(alpha=random.randint(120, 225)))


def draw_histogram(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    values = np.random.normal(size=random.randint(160, 420))
    bins = np.histogram(values, bins=random.randint(8, 26))[0].astype(np.float32)
    bins /= bins.max()
    left, top, right, bottom = box
    slot = (right - left) / len(bins)
    fill = color(alpha=random.randint(170, 230))
    for i, value in enumerate(bins):
        draw.rectangle((left + i * slot, bottom - (bottom - top) * float(value), left + (i + 0.85) * slot, bottom), fill=fill)


def draw_pie(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    size = min(right - left, bottom - top)
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    pie_box = (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)
    values = np.random.gamma(2.0, 1.0, random.randint(3, 8))
    start = random.uniform(0, 360)
    for value in values:
        extent = float(value / values.sum() * 360)
        draw.pieslice(pie_box, start=start, end=start + extent, fill=color(alpha=random.randint(170, 235)), outline=(255, 255, 255, 255))
        start += extent


def draw_heatmap(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    rows = random.randint(6, 18)
    cols = random.randint(6, 18)
    matrix = np.random.randn(rows, cols)
    matrix = (matrix - matrix.min()) / max(1e-6, matrix.max() - matrix.min())
    left, top, right, bottom = box
    for r in range(rows):
        for c in range(cols):
            value = float(matrix[r, c])
            fill = (int(35 + 190 * value), int(70 + 120 * (1 - value)), int(120 + 100 * (1 - value)), 255)
            x0 = left + (right - left) * c / cols
            x1 = left + (right - left) * (c + 1) / cols
            y0 = top + (bottom - top) * r / rows
            y1 = top + (bottom - top) * (r + 1) / rows
            draw.rectangle((x0, y0, x1, y1), fill=fill)


def save_with_artifacts(image: Image.Image, output_path: Path) -> None:
    if random.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    if random.random() < 0.65:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=random.randint(58, 95), optimize=True)
        output_path = output_path.with_suffix(".jpg")
        output_path.write_bytes(buffer.getvalue())
    else:
        image.convert("RGB").save(output_path.with_suffix(".png"), optimize=True)


def draw_chart(chart_type: str, output_path: Path, min_size: int, max_size: int) -> None:
    width = random.randint(min_size, max_size)
    height = random.randint(min_size, max_size)
    background = random.choice([(255, 255, 255, 255), (247, 247, 247, 255), (250, 252, 255, 255), (255, 252, 246, 255)])
    image = Image.new("RGBA", (width, height), background)
    draw = ImageDraw.Draw(image, "RGBA")
    box = plot_box(width, height)
    with_text = random.random() < 0.30
    if chart_type != "pie":
        maybe_axes(draw, box, with_grid=random.random() < 0.45)

    if chart_type == "bar":
        draw_bar(draw, box)
    elif chart_type == "line":
        draw_line(draw, box)
    elif chart_type == "pie":
        draw_pie(draw, box)
    elif chart_type == "scatter":
        draw_scatter(draw, box)
    elif chart_type == "area":
        draw_line(draw, box, area=True)
    elif chart_type == "histogram":
        draw_histogram(draw, box)
    elif chart_type == "heatmap":
        draw_heatmap(draw, box)

    maybe_text(draw, width, height, box, with_text)
    save_with_artifacts(image, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dataset/train/chart_synthetic")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size-min", type=int, default=224)
    parser.add_argument("--image-size-max", type=int, default=640)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.count):
        draw_chart(random.choice(CHART_TYPES), output_dir / f"synthetic_{index:06d}.png", args.image_size_min, args.image_size_max)
        if (index + 1) % 100 == 0:
            print(f"generated {index + 1}/{args.count}")
    print(f"done: {output_dir}")


if __name__ == "__main__":
    main()
