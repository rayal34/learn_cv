import torch
from utils.loss_functions import SoftCrossEntropyLoss


def test_soft_cross_entropy_loss_mean():
    loss_fn = SoftCrossEntropyLoss(reduction="mean")

    # Logits: shape (2, 3)
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    # Soft targets: shape (2, 3)
    soft_targets = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])

    # Calculate log_softmax manually
    # For row 0: max=3.0, exp(1-3)+exp(2-3)+exp(3-3) = e^-2 + e^-1 + 1 approx 0.1353 + 0.3679 + 1 = 1.5032
    # log_sum = log(1.5032) approx 0.4076
    # log_probs row 0: [1 - 3 - log_sum, 2 - 3 - log_sum, 3 - 3 - log_sum] = [-2.4076, -1.4076, -0.4076]
    # loss row 0: - (0.1 * -2.4076 + 0.2 * -1.4076 + 0.7 * -0.4076) = - (-0.24076 - 0.28152 - 0.28532) = 0.8076
    # Let PyTorch calculate the loss
    loss = loss_fn(logits, soft_targets)

    # For verification:
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    expected_loss = -(soft_targets * log_probs).sum(dim=1).mean()

    assert torch.allclose(loss, expected_loss)


def test_soft_cross_entropy_loss_sum():
    loss_fn = SoftCrossEntropyLoss(reduction="sum")

    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    soft_targets = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])

    loss = loss_fn(logits, soft_targets)

    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    expected_loss = -(soft_targets * log_probs).sum(dim=1).sum()

    assert torch.allclose(loss, expected_loss)


def test_soft_cross_entropy_loss_none():
    loss_fn = SoftCrossEntropyLoss(reduction="none")

    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    soft_targets = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])

    loss = loss_fn(logits, soft_targets)

    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    expected_loss = -(soft_targets * log_probs).sum(dim=1)

    assert torch.allclose(loss, expected_loss)
