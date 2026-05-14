from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from metrics import Thresholds, logits_to_predictions
from model import load_checkpoint
from transforms import get_transform


def load_image_tensor(path: str | None, image_size: int, device: torch.device) -> torch.Tensor:
    if path:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
    else:
        image = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    tensor = get_transform("test", image_size=image_size)(image).unsqueeze(0)
    return tensor.to(device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(torch.quantile(torch.tensor(values), q).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--image", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--food-threshold", type=float, default=0.75)
    parser.add_argument("--chart-threshold", type=float, default=0.85)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device=device).to(device).eval()
    image = load_image_tensor(args.image, args.image_size, device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(image)
        synchronize(device)

        latencies_ms: list[float] = []
        last_logits = None
        for _ in range(args.iters):
            start = time.perf_counter()
            last_logits = model(image)
            synchronize(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    thresholds = Thresholds(food=args.food_threshold, chart=args.chart_threshold)
    pred, scores = logits_to_predictions(last_logits.detach().cpu(), thresholds)
    labels = ("food", "chart", "other")
    print(f"checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"device: {device}")
    print(f"batch_size: 1")
    print(f"prediction: {labels[int(pred[0])]}")
    print(f"food_score: {scores[0, 0]:.6f}")
    print(f"chart_score: {scores[0, 1]:.6f}")
    print(f"mean_ms: {statistics.mean(latencies_ms):.3f}")
    print(f"p50_ms: {percentile(latencies_ms, 0.50):.3f}")
    print(f"p95_ms: {percentile(latencies_ms, 0.95):.3f}")
    print(f"p99_ms: {percentile(latencies_ms, 0.99):.3f}")


if __name__ == "__main__":
    main()
