import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.config import DataConfig, TrainingConfig
from core.eval_utils import compute_accuracy
from core.train_utils import (
    EarlyStoppingWithCheckpoint,
    generate_default_exp_name,
    save_model,
    seed_everything,
)

# ==========================================
# Tests for core/eval_utils.py
# ==========================================


def test_compute_accuracy():
    scores = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = torch.tensor([1, 0, 0])  # predictions: [1, 0, 1] vs [1, 0, 0] -> 2 correct
    acc = compute_accuracy(scores, labels)
    assert acc == 2.0


def test_compute_accuracy_one_hot():
    scores = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    acc = compute_accuracy(scores, labels)
    assert acc == 2.0


# ==========================================
# Tests for core/train_utils.py
# ==========================================


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


# ==========================================
# Tests for core/augmentations.py
# ==========================================
from core.augmentations import Cutup, ZeroCenter, cutmix, mixup


def test_zero_center():
    img = torch.tensor([10.0, 20.0, 30.0])
    transform = ZeroCenter(mean=15.0)
    assert torch.allclose(transform(img), torch.tensor([-5.0, 5.0, 15.0]))


def test_mixup():
    X = torch.ones(2, 3, 8, 8)
    y = torch.tensor([0, 1])
    device = torch.device("cpu")
    X_mix, y_mix = mixup(X, y, alpha=1.0, num_classes=2, device=device)
    assert X_mix.shape == (2, 3, 8, 8)
    assert y_mix.shape == (2, 2)
    assert torch.allclose(y_mix.sum(dim=1), torch.ones(2))


def test_cutup_single_count():
    img = torch.ones(3, 8, 8)
    cutup = Cutup(size=4, fill_value=0, count=1)
    out = cutup(img)
    assert out.shape == (3, 8, 8)
    assert (out == 0).any()


def test_cutup_fill_value_list_tuple():
    img = torch.ones(3, 8, 8)
    cutup_list = Cutup(size=4, fill_value=[0.0, 0.0, 0.0], count=1)
    out_list = cutup_list(img)
    assert out_list.shape == (3, 8, 8)

    cutup_tensor = Cutup(size=4, fill_value=torch.tensor([0.0, 0.0, 0.0]), count=1)
    out_tensor = cutup_tensor(img)
    assert out_tensor.shape == (3, 8, 8)


def test_cutmix():
    X = torch.ones(2, 3, 8, 8)
    y = torch.tensor([0, 1])
    device = torch.device("cpu")
    X_mix, y_mix = cutmix(X, y, alpha=1.0, num_classes=2, device=device)
    assert X_mix.shape == (2, 3, 8, 8)
    assert y_mix.shape == (2, 2)
    assert torch.allclose(y_mix.sum(dim=1), torch.ones(2))


# ==========================================
# Additional tests for core/training.py
# ==========================================


@patch("core.train_utils.torch.cuda.is_available")
@patch("core.train_utils.torch.cuda.manual_seed_all")
def test_seed_everything_cuda(mock_seed_all, mock_cuda_avail):
    mock_cuda_avail.return_value = True
    with patch("core.train_utils.torch.backends.cudnn") as mock_cudnn:
        seed_everything(42)
        mock_seed_all.assert_called_with(42)
        assert mock_cudnn.deterministic is True


@patch("core.train_utils.torch.cuda.is_available")
@patch("core.train_utils.torch.backends.mps.is_available")
@patch("core.train_utils.torch.mps.manual_seed")
def test_seed_everything_mps(mock_mps_seed, mock_mps_avail, mock_cuda_avail):
    mock_cuda_avail.return_value = False
    mock_mps_avail.return_value = True
    seed_everything(42)
    mock_mps_seed.assert_called_with(42)


# ==========================================
# Tests for core/config.py
# ==========================================


def test_data_config():
    from omegaconf import OmegaConf

    base_cfg = OmegaConf.structured(DataConfig)
    config = OmegaConf.merge(base_cfg, OmegaConf.load("tests/test_config.yaml").dataset)

    assert config.root == "/dummy/root"
    assert config.data_path == "/dummy/root/data"
    assert config.model_path == "/dummy/root/models"
    assert config.experiment_path == "/dummy/root/experiments"
    assert config.num_workers == 0
    assert config.pin_memory is False


def test_training_config():
    config = TrainingConfig(
        batch_size=64,
        num_epochs=10,
    )
    assert config.batch_size == 64
    assert config.num_epochs == 10
