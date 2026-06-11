from typing import List

import torch.nn as nn


def replace_head(model, num_classes: int):
    fc_in_features = model.fc.in_features
    model.fc = nn.Linear(fc_in_features, num_classes)
    return model


def freeze_layers(model: nn.Module, prefix_layers_to_train: List[str]):
    for name, param in model.named_parameters():
        should_train = any(name.startswith(prefix) for prefix in prefix_layers_to_train)
        param.requires_grad = should_train
    return model


def freeze_bn_stats(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def freeze_all_bn(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


class FineTuneResNet(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, freeze_bn: str = "none"):
        super().__init__()
        self.backbone = backbone
        replace_head(self.backbone, num_classes)
        self.freeze_bn = freeze_bn

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn == "stats":
            freeze_bn_stats(self)
        elif self.freeze_bn == "all":
            freeze_all_bn(self)
        return self

    def forward(self, x):
        return self.backbone(x)
