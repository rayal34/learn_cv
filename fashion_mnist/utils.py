import torch

from fashion_mnist.config import ExperimentConfig


def get_optimizer_and_scheduler(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    exp_config: ExperimentConfig,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler_cls = getattr(torch.optim.lr_scheduler, exp_config.scheduler.type)
    scheduler = scheduler_cls(optimizer, **exp_config.scheduler.params)

    return optimizer, scheduler
