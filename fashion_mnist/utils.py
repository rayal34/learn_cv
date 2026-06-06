import torch

from fashion_mnist.config import ExperimentConfig


def get_optimizer_and_scheduler(
    model: torch.nn.Module,
    exp_config: ExperimentConfig,
):

    decay_params = [
        p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2
    ]
    no_decay_params = [
        p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2
    ]

    optimizer = getattr(torch.optim, exp_config.optimizer.type)(
        [
            {
                "params": decay_params,
                "weight_decay": exp_config.optimizer.params["weight_decay"],
            },
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=exp_config.optimizer.params["lr"],
    )

    scheduler_cls = getattr(torch.optim.lr_scheduler, exp_config.scheduler.type)
    scheduler = scheduler_cls(optimizer, **exp_config.scheduler.params)

    return optimizer, scheduler
