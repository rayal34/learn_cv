import contextlib
import time
from typing import Optional

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cifar.from_scratch.config import ExperimentConfig
from core.eval_utils import compute_accuracy
from core.train_utils import EarlyStoppingWithCheckpoint, compute_update_scale


def train_loop(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    profiler_dir: Optional[str] = None,
):
    model.to(device)
    model.train()
    total_loss, correct = 0, 0
    n_samples = len(dataloader.dataset)  # type: ignore

    if profiler_dir is not None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        prof_ctx = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        prof_ctx = contextlib.nullcontext()

    with prof_ctx as prof:
        for step, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            correct += compute_accuracy(logits, y)

            if profiler_dir is not None:
                prof.step()  # type: ignore
                if step > 10:
                    break

    total_loss /= n_samples
    accuracy = correct / n_samples
    update_scales = compute_update_scale(model, optimizer)
    return total_loss, accuracy, update_scales


def eval_loop(
    dataloader: DataLoader, model, device: torch.device, loss_fn: torch.nn.Module
):
    model.to(device)
    model.eval()
    n_samples = len(dataloader.dataset)  # type: ignore
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            test_loss += loss_fn(logits, y).item()
            correct += compute_accuracy(logits, y)

    test_loss /= n_samples
    accuracy = correct / n_samples
    return test_loss, accuracy


def train_many_epochs(
    epochs: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: torch.nn.Module,
    train_loss_fn: torch.nn.Module,
    eval_loss_fn: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scheduler_update_freq: str = "epoch",
    early_stopping: Optional[EarlyStoppingWithCheckpoint] = None,
    writer: Optional[SummaryWriter] = None,
    profiler_dir: Optional[str] = None,
) -> nn.Module:

    assert scheduler_update_freq in ["step", "epoch"], (
        "scheduler_update_freq must be 'step' or 'epoch'"
    )

    for epoch in range(epochs):
        start_time = time.perf_counter()

        train_loss, train_acc, train_update_scales = train_loop(
            dataloader=train_dataloader,
            model=model,
            loss_fn=train_loss_fn,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler if scheduler_update_freq == "step" else None,
            profiler_dir=profiler_dir,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        test_loss, test_acc = eval_loop(test_dataloader, model, device, eval_loss_fn)

        if scheduler and scheduler_update_freq == "epoch":
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        epoch_time = time.perf_counter() - start_time
        print(
            f"Epoch {epoch + 1}/{epochs}  | "
            f"train_loss: {train_loss:.4f}  |  "
            f"train_acc: {train_acc:.4f}  | "
            f"current_lr: {current_lr:.6f}  |  "
            f"test_loss: {test_loss:.4f}  |  "
            f"test_acc: {test_acc:.4f}  |  "
            f"time: {epoch_time:.1f}s"
        )

        if writer:
            writer.add_scalar("Loss/training", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Accuracy/training", train_acc, epoch)
            writer.add_scalar("Accuracy/test", test_acc, epoch)
            writer.add_scalar("Learning Rate", current_lr, epoch)
            for layer_name, update_scale in train_update_scales.items():
                writer.add_scalar(
                    f"Gradient Update Scale/{layer_name}", update_scale, epoch
                )

        if early_stopping:
            early_stopping(test_acc, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                return model

    return model


def get_optimizer_and_scheduler(
    model: torch.nn.Module,
    exp_config: ExperimentConfig,
    train_dataloader: DataLoader,
):

    decay_params = [
        p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2
    ]
    no_decay_params = [
        p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2
    ]

    optimizer = getattr(optim, exp_config.optimizer.type)(
        [
            {
                "params": decay_params,
                "weight_decay": exp_config.optimizer.params["weight_decay"],
            },
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=exp_config.optimizer.params["lr"],
    )

    scheduler_cls = getattr(optim.lr_scheduler, exp_config.scheduler.type)
    scheduler_params = dict(exp_config.scheduler.params)

    if exp_config.scheduler.type == "OneCycleLR":
        scheduler_params["steps_per_epoch"] = len(train_dataloader)
        scheduler_params["epochs"] = exp_config.training.num_epochs

    scheduler = scheduler_cls(optimizer, **scheduler_params)

    return optimizer, scheduler
