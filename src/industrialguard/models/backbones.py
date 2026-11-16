"""Torchvision feature extractors shared by PaDiM and PatchCore."""

from __future__ import annotations

from collections import OrderedDict
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn


class ResNetFeatureExtractor(nn.Module):
    """Return selected intermediate feature maps from a pretrained ResNet."""

    SUPPORTED: ClassVar[set[str]] = {
        "resnet18",
        "resnet34",
        "resnet50",
        "wide_resnet50_2",
    }

    def __init__(self, name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if name not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported ResNet backbone {name!r}; choose from {sorted(self.SUPPORTED)}"
            )
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError(
                "Install the 'models' extra to use pretrained torchvision backbones"
            ) from exc
        constructor = getattr(models, name)
        weights = "DEFAULT" if pretrained else None
        self.network = constructor(weights=weights)
        self.network.fc = nn.Identity()
        self.network.avgpool = nn.Identity()
        self.name = name
        self.pretrained = pretrained
        self.out_channels = {
            "resnet18": (64, 128, 256),
            "resnet34": (64, 128, 256),
            "resnet50": (256, 512, 1024),
            "wide_resnet50_2": (256, 512, 1024),
        }[name]

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        network = self.network
        x = network.relu(network.bn1(network.conv1(images)))
        x = network.maxpool(x)
        layer1 = network.layer1(x)
        layer2 = network.layer2(layer1)
        layer3 = network.layer3(layer2)
        return OrderedDict(layer1=layer1, layer2=layer2, layer3=layer3)


def build_backbone(
    name: str = "resnet18", pretrained: bool = True
) -> ResNetFeatureExtractor:
    """Build a lazy torchvision backbone; pretrained weights download only on request."""
    return ResNetFeatureExtractor(name=name, pretrained=pretrained)


def aggregate_features(features: dict[str, torch.Tensor]) -> torch.Tensor:
    """Resize deeper maps to the finest map and concatenate channels."""
    if not features:
        raise ValueError("Backbone returned no feature maps")
    target = next(iter(features.values())).shape[-2:]
    aligned = [
        feature
        if feature.shape[-2:] == target
        else F.interpolate(feature, size=target, mode="bilinear", align_corners=False)
        for feature in features.values()
    ]
    return torch.cat(aligned, dim=1)
