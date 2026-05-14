from __future__ import annotations

import argparse
from datetime import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from dataset import BalancedBatchSampler, DirectoryImageDataset
from metrics import Thresholds, compute_metrics, format_metrics
from model import create_model, load_checkpoint, set_backbone_trainable


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        if yaml is not None:
            return yaml.safe_load(file)
        raise RuntimeError("PyYAML is required to read config.yaml. Install pyyaml or pass a JSON config.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "label": [item["label"] for item in batch],
        "path": [item["path"] for item in batch],
        "source": [item["source"] for item in batch],
    }


def make_eval_loader(config: dict, split: str) -> DataLoader | None:
    dataset_cfg = config["dataset"]
    split_root = Path(dataset_cfg["root"]) / split
    if not split_root.exists():
        return None
    dataset = DirectoryImageDataset(
        dataset_cfg["root"],
        split,
        dataset_cfg["image_size"],
        max_image_pixels=dataset_cfg.get("max_image_pixels", 50_000_000),
        aug_strength=dataset_cfg.get("aug_strength", "default"),
    )
    return DataLoader(
        dataset,
        batch_size=dataset_cfg.get("eval_batch_size", 64),
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )


def make_loaders(config: dict) -> tuple[DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    train_set = DirectoryImageDataset(
        dataset_cfg["root"],
        "train",
        dataset_cfg["image_size"],
        max_image_pixels=dataset_cfg.get("max_image_pixels", 50_000_000),
        aug_strength=dataset_cfg.get("aug_strength", "default"),
    )
    val_set = DirectoryImageDataset(
        dataset_cfg["root"],
        "val",
        dataset_cfg["image_size"],
        max_image_pixels=dataset_cfg.get("max_image_pixels", 50_000_000),
        aug_strength=dataset_cfg.get("aug_strength", "default"),
    )
    sampler = BalancedBatchSampler(
        train_set,
        dataset_cfg["batch_class_counts"],
        chart_real_fraction=dataset_cfg.get("chart_real_fraction", 0.30),
        seed=config.get("seed", 42),
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        num_workers=dataset_cfg.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=dataset_cfg.get("eval_batch_size", 64),
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    print(f"train counts: {train_set.class_counts()}")
    print(f"val counts: {val_set.class_counts()}")
    return train_loader, val_loader


def make_optimizer(model: nn.Module, config: dict, backbone_trainable: bool) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    if backbone_trainable:
        backbone_params = [p for name, p in model.named_parameters() if not name.startswith("classifier.") and p.requires_grad]
        head_params = [p for name, p in model.named_parameters() if name.startswith("classifier.") and p.requires_grad]
        groups = [
            {"params": backbone_params, "lr": train_cfg["lr_backbone"]},
            {"params": head_params, "lr": train_cfg["lr_head"]},
        ]
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": train_cfg["lr_head"]}]
    return torch.optim.AdamW(groups, weight_decay=train_cfg["weight_decay"])


def make_criterion(config: dict, device: torch.device) -> nn.Module:
    loss_cfg = config["train"].get("loss", {})
    pos_weight = loss_cfg.get("pos_weight")
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=weight)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = True,
    grad_clip_norm: float | None = None,
    log_interval: int = 0,
    epoch_name: str = "",
) -> tuple[float, torch.Tensor, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and training and device.type == "cuda")
    losses: list[float] = []
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        start_time = time.time()
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            losses.append(float(loss.detach().cpu()))
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
            if log_interval and batch_index % log_interval == 0:
                elapsed = max(1e-6, time.time() - start_time)
                print(
                    f"{epoch_name} batch {batch_index}/{len(loader)} "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"{batch_index / elapsed:.2f} batch/s",
                    flush=True,
                )
    return float(np.mean(losses)), torch.cat(all_logits), torch.cat(all_targets)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, config: dict, val_metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "weights": config["model"].get("weights"),
            "val_metrics": val_metrics,
        },
        path,
    )


def make_run_dir(base_dir: str | Path, run_name: str | None = None) -> Path:
    base_dir = Path(base_dir)
    name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / name
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = torch.device(args.device)
    train_loader, val_loader = make_loaders(config)
    thresholds = Thresholds(food=config["thresholds"]["food"], chart=config["thresholds"]["chart"])

    model = create_model(weights=config["model"].get("weights"), freeze_backbone=True).to(device)
    criterion = make_criterion(config, device)
    optimizer = make_optimizer(model, config, backbone_trainable=False)
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    early_stopping_patience = config["train"].get("early_stopping_patience")
    checkpoint_dir = make_run_dir(config["model"].get("checkpoint_dir", "checkpoints"), args.run_name)
    print(f"checkpoint_dir: {checkpoint_dir}")
    with open(checkpoint_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    for epoch in range(1, config["train"]["epochs"] + 1):
        should_unfreeze = (
            config["model"].get("unfreeze_backbone", True)
            and epoch == config["model"].get("freeze_backbone_epochs", 3) + 1
        )
        if should_unfreeze:
            set_backbone_trainable(model, True)
            optimizer = make_optimizer(model, config, backbone_trainable=True)
            print("unfroze ConvNeXt backbone")

        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            amp=config["train"].get("amp", True),
            grad_clip_norm=config["train"].get("grad_clip_norm"),
            log_interval=config["train"].get("log_interval", 0),
            epoch_name=f"train epoch {epoch}",
        )
        val_loss, val_logits, val_targets = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            amp=config["train"].get("amp", True),
            log_interval=0,
            epoch_name=f"val epoch {epoch}",
        )
        val_metrics = compute_metrics(val_logits, val_targets, thresholds)
        print(f"\nepoch {epoch}/{config['train']['epochs']} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        print(format_metrics(val_metrics))

        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, config, val_metrics)
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, config, val_metrics)
            with open(checkpoint_dir / "best_metrics.json", "w", encoding="utf-8") as file:
                json.dump(val_metrics, file, ensure_ascii=False, indent=2)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"early stopping: no macro_f1 improvement for {early_stopping_patience} epochs")
                break

    test_loader = make_eval_loader(config, "test")
    if test_loader is not None:
        best_model = load_checkpoint(str(checkpoint_dir / "best.pt"), device=device).to(device)
        test_loss, test_logits, test_targets = run_epoch(
            best_model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            amp=config["train"].get("amp", True),
        )
        test_metrics = compute_metrics(test_logits, test_targets, thresholds)
        print(f"\ntest_loss={test_loss:.4f}")
        print(format_metrics(test_metrics))
        with open(checkpoint_dir / "test_metrics.json", "w", encoding="utf-8") as file:
            json.dump(test_metrics, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
