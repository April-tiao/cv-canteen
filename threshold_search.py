from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DirectoryImageDataset
from metrics import Thresholds, compute_metrics
from model import load_checkpoint


def collate_fn(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search food/chart sigmoid thresholds on a split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--food-thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--chart-thresholds", default="0.75,0.80,0.85,0.90")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-other-to-food", type=float, default=0.02)
    parser.add_argument("--max-other-to-chart", type=float, default=0.02)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    logits_list = []
    targets_list = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits_list.append(model(images).detach().cpu())
            targets_list.append(batch["target"])
    logits = torch.cat(logits_list)
    targets = torch.cat(targets_list)

    food_thresholds = [float(x) for x in args.food_thresholds.split(",") if x.strip()]
    chart_thresholds = [float(x) for x in args.chart_thresholds.split(",") if x.strip()]
    rows = []
    for t_food in food_thresholds:
        for t_chart in chart_thresholds:
            metrics = compute_metrics(logits, targets, Thresholds(food=t_food, chart=t_chart))
            row = {
                "t_food": t_food,
                "t_chart": t_chart,
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            }
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["other_to_food_fpr"] <= args.max_other_to_food,
            row["other_to_chart_fpr"] <= args.max_other_to_chart,
            row["food_recall"],
            row["macro_f1"],
        ),
        reverse=True,
    )

    csv_path = output_dir / "threshold_search.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_path = output_dir / "best_thresholds.json"
    with open(best_path, "w", encoding="utf-8") as file:
        json.dump(rows[0], file, ensure_ascii=False, indent=2)

    print(f"searched={len(rows)}")
    print(f"csv={csv_path}")
    print(f"best={best_path}")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
