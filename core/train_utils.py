import random
from datetime import datetime

import numpy as np
import torch
from torchsummary import summary

from core.io import save_model


class EarlyStoppingWithCheckpoint:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        patience: int = 5,
        min_delta: float = 0.0,
        higher_is_better: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.model_path = model_path
        self.model_name = model_name
        self.higher_is_better = higher_is_better

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float, model: torch.nn.Module):
        if self.best_score is None:
            # initialize the best score to the current score
            self.best_score = score

        if self.higher_is_better:
            improved = True if score > self.best_score + self.min_delta else False
        else:
            improved = True if score < self.best_score - self.min_delta else False

        if improved:
            self.best_score = score
            save_model(model, self.model_path, f"{self.model_name}.pt")
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def compute_update_scale(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict:
    update_scales = {}
    lr = optimizer.param_groups[0]["lr"]
    for name, params in model.named_parameters():
        if params.grad is not None:
            grad_norm = params.grad.norm().item()
            weight_norm = params.data.norm().item()
            update_scale = lr * grad_norm / (weight_norm + 1e-8)
            update_scales[name] = update_scale

    return update_scales


def generate_default_exp_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def print_model_summary(model: torch.nn.Module, input_shape: tuple[int, int, int]):

    summary(model, input_shape, device="cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
