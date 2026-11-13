"""Five transfer-learning classifiers, swappable via config, compared head-to-head
in the channel/condition/model study (scripts/run_experiment_grid.py).

Backbones: ResNet50, ResNet18, VGG16, DenseNet121, MobileNetV2 -- all pretrained
on ImageNet. For every backbone the same recipe is followed: keep the pretrained
convolutional feature extractor, freeze everything except a small, late slice of
it, and replace the *entire* original classifier with one small binary head
trained from scratch on top of the pooled features. As in the single-model
version of this project, "replace final FC layer" is interpreted as replacing
the whole original classifier, not just its last linear layer, so every
backbone is handled identically in shape.

Two families of "which layers stay trainable" logic are used:
  - Name-addressable backbones (ResNet18/50, DenseNet121): `model.freeze_until`
    in the config names one of the feature extractor's own named children
    (e.g. "layer4" for ResNet, "denseblock4" for DenseNet) and everything from
    that child onward is unfrozen.
  - Index-only backbones (VGG16, MobileNetV2): these expose their features as a
    flat, unnamed `nn.Sequential`, so there is no block name to put in a config
    file. The last convolutional block is always unfrozen regardless of
    `freeze_until`, matching the documented VGG16 behaviour already in this
    project -- noted here again since it means the config option isn't fully
    backbone-agnostic for these two.
"""
from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

_VGG16_LAST_BLOCK_START = 24      # index of the first Conv2d in VGG16's 5th conv block
_MOBILENET_LAST_BLOCK_START = 14  # index of the first InvertedResidual kept trainable

SUPPORTED_BACKBONES = ("resnet50", "resnet18", "vgg16", "densenet121", "mobilenet_v2")


def _freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def _unfreeze_from_named_child(features: nn.Module, freeze_until: str, backbone_name: str) -> None:
    names = list(features._modules.keys())
    if freeze_until not in names:
        raise ValueError(f"freeze_until={freeze_until!r} not in {backbone_name} blocks {names}")
    for name in names[names.index(freeze_until):]:
        for p in features._modules[name].parameters():
            p.requires_grad = True


def _make_head(in_features: int, hidden_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


class EEGDepressionClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        freeze_until: str,
        dropout: float,
        hidden_dim: int,
    ):
        super().__init__()
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone_name!r} (expected one of {SUPPORTED_BACKBONES})")
        self.backbone_name = backbone_name
        self._relu_before_pool = False  # DenseNet's own forward() applies ReLU before pooling

        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet50(weights=weights)
            in_features = net.fc.in_features
            self.features = nn.Sequential(OrderedDict(list(net.named_children())[:-2]))
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            _freeze_all(self.features)
            _unfreeze_from_named_child(self.features, freeze_until, backbone_name)

        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
            in_features = net.fc.in_features
            self.features = nn.Sequential(OrderedDict(list(net.named_children())[:-2]))
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            _freeze_all(self.features)
            _unfreeze_from_named_child(self.features, freeze_until, backbone_name)

        elif backbone_name == "vgg16":
            weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.vgg16(weights=weights)
            in_features = 512 * 7 * 7
            self.features = net.features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            _freeze_all(self.features)
            for layer in self.features[_VGG16_LAST_BLOCK_START:]:
                for p in layer.parameters():
                    p.requires_grad = True

        elif backbone_name == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.densenet121(weights=weights)
            in_features = net.classifier.in_features
            self.features = net.features  # named children: conv0,norm0,relu0,pool0,denseblock1..4,transition1..3,norm5
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self._relu_before_pool = True  # torchvision's own DenseNet.forward() does this
            _freeze_all(self.features)
            _unfreeze_from_named_child(self.features, freeze_until, backbone_name)

        elif backbone_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.mobilenet_v2(weights=weights)
            in_features = net.classifier[1].in_features
            self.features = net.features  # flat Sequential of InvertedResidual blocks, index-only like VGG16
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            _freeze_all(self.features)
            for layer in self.features[_MOBILENET_LAST_BLOCK_START:]:
                for p in layer.parameters():
                    p.requires_grad = True

        self.head = _make_head(in_features, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        if self._relu_before_pool:
            feats = F.relu(feats, inplace=True)
        pooled = self.avgpool(feats)
        flat = torch.flatten(pooled, 1)
        return self.head(flat).squeeze(1)  # raw logits, shape (batch,)


# Each backbone's sensible default for `model.freeze_until` when a config doesn't
# override it (ignored by vgg16/mobilenet_v2, which always unfreeze their fixed
# last block -- see module docstring).
DEFAULT_FREEZE_UNTIL = {
    "resnet50": "layer4",
    "resnet18": "layer4",
    "densenet121": "denseblock4",
    "vgg16": "",
    "mobilenet_v2": "",
}


def build_model(cfg: SimpleNamespace) -> EEGDepressionClassifier:
    backbone_name = cfg.model.backbone
    freeze_until = getattr(cfg.model, "freeze_until", None) or DEFAULT_FREEZE_UNTIL.get(backbone_name, "")
    return EEGDepressionClassifier(
        backbone_name=backbone_name,
        pretrained=cfg.model.pretrained,
        freeze_until=freeze_until,
        dropout=cfg.model.dropout,
        hidden_dim=cfg.model.head_hidden_dim,
    )
