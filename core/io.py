import os

import torch


def save_model(model: torch.nn.Module, path: str, filename: str) -> None:
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, filename))
