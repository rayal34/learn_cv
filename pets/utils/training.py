import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import complete_box_iou_loss

from core import eval_utils, train_utils
from pets.config import ExperimentConfig


def profile(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    profiler_dir: str = "runs/profiler",
    max_steps: int = 11,
):
    model.train()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, (img, box, label) in enumerate(dataloader):
            img, box, label = img.to(device), box.to(device), label.to(device)
            logits, box_regs = model(img)

            clf_loss = F.cross_entropy(logits, label, reduction="sum")
            reg_loss = complete_box_iou_loss(box_regs, box, reduction="sum")

            loss = clf_loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            prof.step()
            if step >= max_steps:
                break


def train_loop(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
):
    model.train()
    total_clf_loss, total_reg_loss, correct = 0, 0, 0
    n_samples = len(dataloader.dataset)  # type: ignore

    for img, box, label in dataloader:
        img, box, label = img.to(device), box.to(device), label.to(device)
        logits, box_regs = model(img)

        clf_loss = F.cross_entropy(logits, label, reduction="sum")
        reg_loss = complete_box_iou_loss(box_regs, box, reduction="sum")

        loss = clf_loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_clf_loss += clf_loss.item()
        total_reg_loss += reg_loss.item()

        correct += eval_utils.compute_accuracy(logits, label)

    total_clf_loss /= n_samples
    total_reg_loss /= n_samples
    accuracy = correct / n_samples
    update_scales = train_utils.compute_update_scale(model, optimizer)
    return total_clf_loss, total_reg_loss, accuracy, update_scales


def eval_loop(dataloader: DataLoader, model, device: torch.device):
    model.eval()
    n_samples = len(dataloader.dataset)  # type: ignore
    test_clf_loss, test_reg_loss, correct = 0, 0, 0
    with torch.no_grad():
        for img, box, label in dataloader:
            img, box, label = img.to(device), box.to(device), label.to(device)
            logits, box_regs = model(img)
            clf_loss = F.cross_entropy(logits, label, reduction="sum")
            reg_loss = complete_box_iou_loss(box_regs, box, reduction="sum")

            test_clf_loss += clf_loss.item()
            test_reg_loss += reg_loss.item()

            correct += eval_utils.compute_accuracy(logits, label)

    test_clf_loss /= n_samples
    test_reg_loss /= n_samples
    accuracy = correct / n_samples
    return test_clf_loss, test_reg_loss, accuracy


def train_many_epochs(
    epochs: int,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scheduler_update_freq: str = "epoch",
    early_stopping: Optional[train_utils.EarlyStoppingWithCheckpoint] = None,
    writer: Optional[SummaryWriter] = None,
) -> nn.Module:

    assert scheduler_update_freq in ["step", "epoch"], (
        "scheduler_update_freq must be 'step' or 'epoch'"
    )

    model = model.to(device)
    for epoch in range(epochs):
        start_time = time.perf_counter()

        train_clf_loss, train_reg_loss, train_acc, train_update_scales = train_loop(
            dataloader=train_dataloader,
            model=model,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler if scheduler_update_freq == "step" else None,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        test_clf_loss, test_reg_loss, test_acc = eval_loop(
            test_dataloader, model, device
        )

        if scheduler and scheduler_update_freq == "epoch":
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_clf_loss)
            else:
                scheduler.step()

        epoch_time = time.perf_counter() - start_time
        print(
            f"Epoch {epoch + 1}/{epochs}  | "
            f"train_clf_loss: {train_clf_loss:.4f}  |  "
            f"train_reg_loss: {train_reg_loss:.4f}  |  "
            f"train_acc: {train_acc:.4f}  | "
            f"current_lr: {current_lr:.6f}  |  "
            f"test_clf_loss: {test_clf_loss:.4f}  |  "
            f"test_reg_loss: {test_reg_loss:.4f}  |  "
            f"test_acc: {test_acc:.4f}  |  "
            f"time: {epoch_time:.1f}s"
        )

        if writer:
            writer.add_scalar("Clf_loss/training", train_clf_loss, epoch)
            writer.add_scalar("Reg_loss/training", train_reg_loss, epoch)
            writer.add_scalar("Clf_loss/test", test_clf_loss, epoch)
            writer.add_scalar("Reg_loss/test", test_reg_loss, epoch)
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

    head_decay, head_no_decay, backbone_decay, backbone_no_decay = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            if param.dim() >= 2:
                backbone_decay.append(param)
            else:
                backbone_no_decay.append(param)
        else:
            if param.dim() >= 2:
                head_decay.append(param)
            else:
                head_no_decay.append(param)

    parameter_specs = [
        {
            "params": head_decay,
            "weight_decay": exp_config.optimizer.params["head_weight_decay"],
            "lr": exp_config.optimizer.params["head_lr"],
        },
        {
            "params": head_no_decay,
            "weight_decay": 0.0,
            "lr": exp_config.optimizer.params["head_lr"],
        },
        {
            "params": backbone_decay,
            "weight_decay": exp_config.optimizer.params["backbone_weight_decay"],
            "lr": exp_config.optimizer.params["backbone_lr"],
        },
        {
            "params": backbone_no_decay,
            "weight_decay": 0.0,
            "lr": exp_config.optimizer.params["backbone_lr"],
        },
    ]
    optimizer = getattr(optim, exp_config.optimizer.type)(parameter_specs)

    scheduler_cls = getattr(optim.lr_scheduler, exp_config.scheduler.type)
    scheduler_params = dict(exp_config.scheduler.params)

    if exp_config.scheduler.type == "OneCycleLR":
        scheduler_params["steps_per_epoch"] = len(train_dataloader)
        scheduler_params["epochs"] = exp_config.training.num_epochs

    scheduler = scheduler_cls(optimizer, **scheduler_params)

    return optimizer, scheduler
