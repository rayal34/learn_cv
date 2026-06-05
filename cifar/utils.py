import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from cifar.config import ExperimentConfig


def get_optimizer_and_scheduler(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    exp_config: ExperimentConfig,
    train_dataloader: DataLoader,
):

    decay_params = [
        p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2
    ]
    no_decay_params = [
        p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
    scheduler_cls = getattr(lr_scheduler, exp_config.scheduler.type)
    scheduler_params = dict(exp_config.scheduler.params)

    if exp_config.scheduler.type == "OneCycleLR":
        scheduler_params["steps_per_epoch"] = len(train_dataloader)
        scheduler_params["epochs"] = exp_config.training.num_epochs

    scheduler = scheduler_cls(optimizer, **scheduler_params)

    return optimizer, scheduler
