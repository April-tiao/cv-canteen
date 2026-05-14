from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


def _resolve_weights(name: Optional[str]) -> Optional[ConvNeXt_Tiny_Weights]:
    if name is None or str(name).lower() in {"none", "null", ""}:
        return None
    if name == "DEFAULT":
        return ConvNeXt_Tiny_Weights.DEFAULT
    return ConvNeXt_Tiny_Weights[name]


def create_model(weights: Optional[str] = "IMAGENET1K_V1", freeze_backbone: bool = True) -> nn.Module:
    model = convnext_tiny(weights=_resolve_weights(weights))
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    if freeze_backbone:
        set_backbone_trainable(model, False)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("classifier."):
            param.requires_grad = trainable


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=device)
    weights = checkpoint.get("weights", None)
    model = create_model(weights=weights, freeze_backbone=False)
    model.load_state_dict(checkpoint["model"])
    return model
