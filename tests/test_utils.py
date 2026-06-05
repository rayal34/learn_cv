import os
import time
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.general_utils import timer
from utils.img_utils import compute_conv_layer_sizes, conv_output_size
from utils.train_utils import (
    EarlyStoppingWithCheckpoint,
    compute_accuracy,
    eval_loop,
    generate_default_exp_name,
    save_model,
    seed_everything,
    train_loop,
    train_many_epochs,
    train_one_epoch,
)

# ==========================================
# Tests for utils/general_utils.py
# ==========================================


def test_timer_decorator(capsys):
    @timer
    def dummy_func(x):
        time.sleep(0.01)
        return x * 2

    res = dummy_func(5)
    assert res == 10

    captured = capsys.readouterr()
    assert "Function dummy_func took" in captured.out
    assert "seconds" in captured.out


# ==========================================
# Tests for utils/img_utils.py
# ==========================================


def test_conv_output_size():
    # in_size=28, kernel=3, stride=1, padding=1 -> (28 - 3 + 2*1)//1 + 1 = 28
    assert conv_output_size(28, 3, 1, 1) == 28
    # in_size=28, kernel=3, stride=2, padding=0 -> (28 - 3)//2 + 1 = 13
    assert conv_output_size(28, 3, 2, 0) == 13
    # in_size=13, kernel=2, stride=2, padding=0 -> (13 - 2)//2 + 1 = 6
    assert conv_output_size(13, 2, 2, 0) == 6


def test_compute_conv_layer_sizes():
    class DummyLayer:
        def __init__(self, kernel_size, stride, padding):
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding

    layers = [
        DummyLayer(3, 1, 1),
        DummyLayer(3, 2, 0),
        DummyLayer(2, 2, 0),
    ]
    sizes = compute_conv_layer_sizes(28, layers)
    assert sizes == [28, 13, 6]


# ==========================================
# Tests for utils/train_utils.py
# ==========================================


def test_compute_accuracy():
    scores = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = torch.tensor([1, 0, 0])  # predictions: [1, 0, 1] vs [1, 0, 0] -> 2 correct
    acc = compute_accuracy(scores, labels)
    assert acc == 2.0


def test_early_stopping_with_checkpoint_higher_better(tmp_path):
    model_path = str(tmp_path / "models")
    model_name = "test_model"
    model = nn.Linear(2, 2)

    early_stopping = EarlyStoppingWithCheckpoint(
        model_path=model_path,
        model_name=model_name,
        patience=3,
        min_delta=0.01,
        higher_is_better=True,
    )

    # Initial score
    # First call sets best_score=0.5, but because 0.5 > 0.5 + 0.01 is False,
    # it counts as a non-improvement and increments counter to 1.
    early_stopping(0.5, model)
    assert early_stopping.best_score == 0.5
    assert early_stopping.counter == 1
    assert not early_stopping.early_stop
    assert not os.path.exists(os.path.join(model_path, f"{model_name}.pt"))

    # Score does not improve enough
    early_stopping(0.505, model)  # delta is 0.005 < 0.01
    assert early_stopping.best_score == 0.5
    assert early_stopping.counter == 2
    assert not early_stopping.early_stop
    assert not os.path.exists(os.path.join(model_path, f"{model_name}.pt"))

    # Score improves significantly
    early_stopping(0.6, model)
    assert early_stopping.best_score == 0.6
    assert early_stopping.counter == 0
    assert not early_stopping.early_stop
    assert os.path.exists(os.path.join(model_path, f"{model_name}.pt"))

    # Remove the file to see if it saves on improvement
    os.remove(os.path.join(model_path, f"{model_name}.pt"))

    # Keep not improving to trigger early stop
    early_stopping(0.6, model)
    assert early_stopping.counter == 1
    early_stopping(0.6, model)
    assert early_stopping.counter == 2
    assert not early_stopping.early_stop
    early_stopping(0.6, model)
    assert early_stopping.counter == 3
    assert early_stopping.early_stop


def test_early_stopping_with_checkpoint_lower_better(tmp_path):
    model_path = str(tmp_path / "models")
    model_name = "test_model_lower"
    model = nn.Linear(2, 2)

    early_stopping = EarlyStoppingWithCheckpoint(
        model_path=model_path,
        model_name=model_name,
        patience=2,
        min_delta=0.01,
        higher_is_better=False,
    )

    # Initial score
    # First call sets best_score=1.0, but because 1.0 < 1.0 - 0.01 is False,
    # it counts as a non-improvement and increments counter to 1.
    early_stopping(1.0, model)
    assert early_stopping.best_score == 1.0
    assert early_stopping.counter == 1

    # Score improves significantly (decreases)
    early_stopping(0.8, model)
    assert early_stopping.best_score == 0.8
    assert early_stopping.counter == 0
    assert os.path.exists(os.path.join(model_path, f"{model_name}.pt"))

    # Score does not improve
    early_stopping(0.85, model)
    assert early_stopping.counter == 1
    early_stopping(0.8, model)
    assert early_stopping.counter == 2
    assert early_stopping.early_stop


def test_seed_everything():
    seed_everything(123)
    val1 = torch.randn(5)
    seed_everything(123)
    val2 = torch.randn(5)
    assert torch.equal(val1, val2)


def test_save_model(tmp_path):
    model_path = str(tmp_path / "models")
    model = nn.Linear(3, 3)
    save_model(model, model_path, "linear.pt")
    assert os.path.exists(os.path.join(model_path, "linear.pt"))


def test_generate_default_exp_name():
    name = generate_default_exp_name()
    assert len(name) > 0
    # Expected format: YYYY-MM-DD_HH-MM-SS
    assert len(name.split("_")) == 2


@pytest.fixture
def dummy_train_setup():
    # Simple dataset: input of dim 2, output class binary
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=5)

    model = nn.Linear(2, 2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    return dataloader, model, loss_fn, optimizer, device


def test_train_and_eval_loops(dummy_train_setup):
    dataloader, model, loss_fn, optimizer, device = dummy_train_setup

    # Test train_loop
    loss, acc, update_scales = train_loop(dataloader, model, loss_fn, device, optimizer)
    assert loss >= 0
    assert 0.0 <= acc <= 1.0
    assert isinstance(update_scales, dict)

    # Test eval_loop
    eval_loss, eval_acc = eval_loop(dataloader, model, device, loss_fn)
    assert eval_loss >= 0
    assert 0.0 <= eval_acc <= 1.0


def test_train_one_epoch(dummy_train_setup):
    dataloader, model, loss_fn, optimizer, device = dummy_train_setup
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    train_loss, train_acc, train_update_scales, current_lr, test_loss, test_acc = (
        train_one_epoch(
            train_dataloader=dataloader,
            test_dataloader=dataloader,
            model=model,
            train_loss_fn=loss_fn,
            eval_loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    )
    assert train_loss >= 0
    assert train_acc >= 0
    assert isinstance(train_update_scales, dict)
    assert current_lr > 0
    assert test_loss >= 0
    assert test_acc >= 0


def test_train_many_epochs(dummy_train_setup, tmp_path):
    dataloader, model, loss_fn, optimizer, device = dummy_train_setup
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Run 2 epochs without early stopping
    model_out = train_many_epochs(
        epochs=2,
        train_dataloader=dataloader,
        test_dataloader=dataloader,
        model=model,
        train_loss_fn=loss_fn,
        eval_loss_fn=loss_fn,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=None,
    )
    assert model_out is model

    # Run with early stopping
    early_stopping = EarlyStoppingWithCheckpoint(
        model_path=str(tmp_path / "models"),
        model_name="test_many",
        patience=1,
    )
    # Run for 5 epochs; early stopping will trigger because dummy scores won't improve consistently
    train_many_epochs(
        epochs=5,
        train_dataloader=dataloader,
        test_dataloader=dataloader,
        model=model,
        train_loss_fn=loss_fn,
        eval_loss_fn=loss_fn,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
    )
    # Should stop early or complete
    assert early_stopping.best_score is not None


def test_train_one_epoch_custom_loop(dummy_train_setup):
    dataloader, model, loss_fn, optimizer, device = dummy_train_setup

    with patch(
        "utils.train_utils.train_loop", return_value=(0.123, 0.999, {"layer": 0.01})
    ):
        train_loss, train_acc, train_update_scales, _, _, _ = train_one_epoch(
            train_dataloader=dataloader,
            test_dataloader=dataloader,
            model=model,
            train_loss_fn=loss_fn,
            eval_loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
        )
    assert train_loss == 0.123
    assert train_acc == 0.999
    assert train_update_scales == {"layer": 0.01}
