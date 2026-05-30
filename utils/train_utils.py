import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from utils import general_utils


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


def compute_accuracy(scores, labels):
    return (scores.argmax(1) == labels).type(torch.float).sum().item()


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


def train_loop(
    dataloader: DataLoader,
    model,
    loss_fn,
    optimizer,
    device: torch.device,
):
    model.to(device)
    model.train()
    total_loss, correct = 0, 0
    n_samples = len(dataloader.dataset)  # type: ignore
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += compute_accuracy(pred, y)

    total_loss /= n_samples
    accuracy = correct / n_samples
    update_scales = compute_update_scale(model, optimizer)
    return total_loss, accuracy, update_scales


def eval_loop(dataloader: DataLoader, model, loss_fn, device: torch.device):
    model.to(device)
    model.eval()
    n_samples = len(dataloader.dataset)  # type: ignore
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += compute_accuracy(pred, y)

    test_loss /= n_samples
    accuracy = correct / n_samples
    return test_loss, accuracy


@general_utils.timer
def train_one_epoch(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model,
    loss_fn,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
):
    train_loss, train_acc, train_update_scales = train_loop(
        train_dataloader, model, loss_fn, optimizer, device
    )
    test_loss, test_acc = eval_loop(test_dataloader, model, loss_fn, device)
    if scheduler:
        scheduler.step(test_loss)
    return train_loss, train_acc, train_update_scales, test_loss, test_acc


def train_many_epochs(
    epochs: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    early_stopping: EarlyStoppingWithCheckpoint | None = None,
    writer: SummaryWriter | None = None,
) -> nn.Module:

    for epoch in range(epochs):
        train_loss, train_acc, train_update_scales, test_loss, test_acc = (
            train_one_epoch(
                train_dataloader,
                test_dataloader,
                model,
                loss_fn,
                device,
                optimizer,
                scheduler,
            )
        )

        print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"train_loss: {train_loss:.4f}  "
            f"train_acc: {train_acc:.4f}  "
            f"test_loss: {test_loss:.4f}  "
            f"test_acc: {test_acc:.4f}"
        )

        if writer:
            writer.add_scalar("Training Loss", train_loss, epoch)
            writer.add_scalar("Test Loss", test_loss, epoch)
            writer.add_scalar("Training Accuracy", train_acc, epoch)
            writer.add_scalar("Test Accuracy", test_acc, epoch)
            for layer_name, update_scale in train_update_scales.items():
                writer.add_scalar(
                    f"Gradient Update Scale {layer_name}", update_scale, epoch
                )

        if early_stopping:
            early_stopping(test_acc, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                return model

    return model


def save_model(model: torch.nn.Module, path: str, filename: str) -> None:
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, filename))


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


def get_optimizer_and_scheduler(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    scheduler_patience: int,
    scheduler_factor: float,
):
    """
    Factory function to initialize the optimizer and scheduler.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=scheduler_patience,
        factor=scheduler_factor,
    )
    return optimizer, scheduler
