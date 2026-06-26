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

    return model


def freeze_all_bn(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    return model


def unfreeze_all_layers(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = True
    return model
