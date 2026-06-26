import torch


def compute_accuracy(scores, labels):
    if labels.ndim > 1:
        labels = labels.argmax(dim=1)
    return (scores.argmax(1) == labels).type(torch.float).sum().item()
