from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DirectoryImageDataset
from metrics import CLASS_NAMES, Thresholds, logits_to_predictions, targets_to_class
from model import load_checkpoint


def collate_fn(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "label": [item["label"] for item in batch],
        "path": [item["path"] for item in batch],
        "source": [item["source"] for item in batch],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export misclassified validation/test images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--food-threshold", type=float, default=0.75)
    parser.add_argument("--chart-threshold", type=float, default=0.85)
    parser.add_argument("--true-label", default=None, choices=[None, *CLASS_NAMES])
    parser.add_argument("--pred-label", default=None, choices=[None, *CLASS_NAMES])
    parser.add_argument("--output-csv", default="misclassified.csv")
    parser.add_argument("--copy-dir", default=None, help="Optional directory to copy exported images into.")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = DirectoryImageDataset(args.dataset_root, args.split, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    model = load_checkpoint(args.checkpoint, device=device).to(device).eval()
    thresholds = Thresholds(food=args.food_threshold, chart=args.chart_threshold)
    rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"]
            logits = model(images).detach().cpu()
            preds, scores = logits_to_predictions(logits, thresholds)
            truths = targets_to_class(targets)
            for index, path in enumerate(batch["path"]):
                truth = CLASS_NAMES[int(truths[index])]
                pred = CLASS_NAMES[int(preds[index])]
                if truth == pred:
                    continue
                if args.true_label and truth != args.true_label:
                    continue
                if args.pred_label and pred != args.pred_label:
                    continue
                rows.append(
                    {
                        "path": path,
                        "true": truth,
                        "pred": pred,
                        "food_score": float(scores[index, 0]),
                        "chart_score": float(scores[index, 1]),
                        "source": batch["source"][index],
                    }
                )

    rows.sort(key=lambda row: (row["true"], row["pred"], -row["food_score"], -row["chart_score"]))
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "true", "pred", "food_score", "chart_score", "source"])
        writer.writeheader()
        writer.writerows(rows)

    if args.copy_dir:
        copy_dir = Path(args.copy_dir)
        copy_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in enumerate(rows):
            src = Path(row["path"])
            dst = copy_dir / f"{idx:05d}_{row['true']}_as_{row['pred']}_{src.name}"
            if src.exists():
                shutil.copy2(src, dst)

    print(f"exported={len(rows)} csv={output_csv}")
    if args.copy_dir:
        print(f"copy_dir={args.copy_dir}")


if __name__ == "__main__":
    main()
