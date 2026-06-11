from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


class ZeroOneScale:
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        return (img - self.min_val) / (self.max_val - self.min_val)


class ZeroCenter:
    def __init__(self, mean: float):
        self.mean = mean

    def __call__(self, img):
        return img - self.mean


def mixup(X, y, alpha: float, num_classes: int, device: torch.device):
    """
    Mixup augmentation.

    Args:
        X: Tensor of shape (batch_size, channels, height, width)
        y: Tensor of shape (batch_size,)
        alpha: Float
        num_classes: Int
    """
    lambda_ = np.random.beta(alpha, alpha)

    idx = torch.randperm(X.shape[0], device=device)

    X_mix = lambda_ * X + (1 - lambda_) * X[idx]
    y_onehot = F.one_hot(y, num_classes).float()
    y_mix = lambda_ * y_onehot + (1 - lambda_) * y_onehot[idx]
    return X_mix, y_mix


class Cutup:
    def __init__(
        self,
        size: int,
        fill_value: int | Sequence[int | float] | torch.Tensor = 0,
        count: int = 1,
    ):
        self.size = size
        self.fill_value = fill_value
        self.count = count

        if isinstance(fill_value, (list, tuple)):
            self.fill_value = torch.tensor(fill_value).view(3, 1, 1)
        elif isinstance(fill_value, torch.Tensor):
            self.fill_value = fill_value.view(-1, 1, 1)
        else:
            self.fill_value = fill_value

    def __call__(self, img):
        _, height, width = img.shape
        img_cut = img.clone()

        for _ in range(self.count):
            x_center = np.random.choice(np.arange(width)).item()
            y_center = np.random.choice(np.arange(height)).item()

            x_start = max(0, x_center - self.size // 2)
            x_end = min(width, x_start + self.size)
            y_start = max(0, y_center - self.size // 2)
            y_end = min(height, y_start + self.size)

            img_cut[:, y_start:y_end, x_start:x_end] = self.fill_value

        return img_cut


def cutmix(X, y, alpha, num_classes, device):
    batch_size, _, height, width = X.shape
    lambda_ = np.random.beta(alpha, alpha)
    idx = torch.randperm(batch_size, device=device)

    cut_ratio = (1 - lambda_) ** 0.5
    cut_x = int(width * cut_ratio)
    cut_y = int(height * cut_ratio)

    x_center = np.random.randint(width)
    y_center = np.random.randint(height)

    x1 = np.clip(x_center - cut_x // 2, 0, width)
    x2 = np.clip(x_center + cut_x // 2, 0, width)
    y1 = np.clip(y_center - cut_y // 2, 0, height)
    y2 = np.clip(y_center + cut_y // 2, 0, height)

    X_mix = X.clone()
    patches = X[idx, :, y1:y2, x1:x2]
    X_mix[:, :, y1:y2, x1:x2] = patches

    lambda_ = 1 - (x2 - x1) * (y2 - y1) / (height * width)
    y_onehot = F.one_hot(y, num_classes).float()
    y_mix = lambda_ * y_onehot + (1 - lambda_) * y_onehot[idx]
    return X_mix, y_mix
