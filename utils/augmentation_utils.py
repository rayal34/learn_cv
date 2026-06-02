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
    def __init__(self, size: int, fill_value: int = 0, count: int = 1):
        self.size = size
        self.fill_value = fill_value
        self.count = count

    def __call__(self, img):
        _, height, width = img.shape

        for _ in range(self.count):
            x_center = np.random.choice(np.arange(width)).item()
            y_center = np.random.choice(np.arange(height)).item()

            x_start = max(0, x_center - self.size // 2)
            x_end = min(width, x_start + self.size)
            y_start = max(0, y_center - self.size // 2)
            y_end = min(height, y_start + self.size)

            img[:, x_start:x_end, y_start:y_end] = self.fill_value

        return img
