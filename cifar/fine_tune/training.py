import torch
from torch import optim
from torch.utils.data import DataLoader

from cifar.fine_tune.config import ExperimentConfig


def get_optimizer_and_scheduler(
    model: torch.nn.Module,
    exp_config: ExperimentConfig,
    train_dataloader: DataLoader,
):

    head_decay, head_no_decay, backbone_decay, backbone_no_decay = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name:
            if param.dim() >= 2:
                head_decay.append(param)
            else:
                head_no_decay.append(param)
        else:
            if param.dim() >= 2:
                backbone_decay.append(param)
            else:
                backbone_no_decay.append(param)

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
