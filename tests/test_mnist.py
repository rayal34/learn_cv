from unittest.mock import ANY, MagicMock, patch

import torch
import yaml
from omegaconf import MISSING

from core.config import DataConfig, GenericConfig, SchedulerConfig, TrainingConfig
from mnist.config import ExperimentConfig
from mnist.main import main
from mnist.utils.load_data import get_dataloaders, load_test_data, load_training_data
from models.config import SimpleCNNModelConfig

# ==========================================
# Tests for mnist/config.py
# ==========================================


def test_mnist_experiment_config_to_dict():
    dataset_cfg = DataConfig(
        root="/dummy_mnist",
        num_workers=0,
        pin_memory=False,
        train_images_filename="train_imgs",
        train_labels_filename="train_lbls",
        test_images_filename="test_imgs",
        test_labels_filename="test_lbls",
    )
    training_cfg = TrainingConfig(
        batch_size=128,
        num_epochs=3,
    )

    scheduler_cfg = SchedulerConfig(
        type="ReduceLROnPlateau",
        params={
            "patience": 2,
            "factor": 0.5,
        },
    )

    optimizer_cfg = GenericConfig(
        type="AdamW",
        params={
            "lr": 0.001,
            "weight_decay": 1e-4,
        },
    )

    exp_cfg = ExperimentConfig(
        name="test_mnist_exp",
        seed=42,
        dataset=dataset_cfg,
        training=training_cfg,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        model=SimpleCNNModelConfig(
            conv_layers=MISSING, fc_hidden=MISSING, dropout=MISSING
        ),
    )

    d = exp_cfg.to_dict()
    assert d["name"] == "test_mnist_exp"
    assert d["seed"] == 42
    assert d["dataset"]["root"] == "/dummy_mnist"
    assert d["training"]["batch_size"] == 128


# ==========================================
# Tests for mnist/load_data.py
# ==========================================


@patch("mnist.utils.load_data.datasets.MNIST")
def test_load_training_and_test_data(mock_mnist):
    mock_dataset = MagicMock()
    mock_mnist.return_value = mock_dataset

    ds_train = load_training_data("/dummy")
    assert ds_train is mock_dataset
    mock_mnist.assert_called_with(
        root="/dummy", train=True, download=True, transform=ANY
    )

    ds_test = load_test_data("/dummy")
    assert ds_test is mock_dataset
    mock_mnist.assert_called_with(
        root="/dummy", train=False, download=True, transform=ANY
    )


@patch("mnist.utils.load_data.load_training_data")
@patch("mnist.utils.load_data.load_test_data")
def test_get_dataloaders_mnist(mock_load_test, mock_load_train):
    mock_ds = MagicMock()
    # Mock length for DataLoader setup
    mock_ds.__len__.return_value = 10
    mock_load_train.return_value = mock_ds
    mock_load_test.return_value = mock_ds

    dataset_cfg = DataConfig(
        root="/dummy_mnist",
        num_workers=0,
        pin_memory=False,
        train_images_filename="train_imgs",
        train_labels_filename="train_lbls",
        test_images_filename="test_imgs",
        test_labels_filename="test_lbls",
    )
    training_cfg = TrainingConfig(
        batch_size=4,
        num_epochs=3,
    )
    exp_cfg = ExperimentConfig(
        name="test_mnist_dl",
        seed=42,
        dataset=dataset_cfg,
        training=training_cfg,
        model=MISSING,
        scheduler=MISSING,
        optimizer=MISSING,
    )

    train_dl, test_dl = get_dataloaders(exp_cfg)
    assert isinstance(train_dl, torch.utils.data.DataLoader)
    assert isinstance(test_dl, torch.utils.data.DataLoader)
    assert train_dl.batch_size == 4
    assert test_dl.batch_size == 4


# ==========================================
# Tests for mnist/main.py
# ==========================================


@patch("core.io.save_model")
@patch("mnist.main.training.train_many_epochs")
@patch("mnist.utils.load_data.get_dataloaders")
@patch("mnist.main.SummaryWriter")
@patch("mnist.main.torch.compile")
@patch("mnist.main.OmegaConf.merge")
def test_mnist_main(
    mock_merge,
    mock_compile,
    mock_summary_writer,
    mock_get_dataloaders,
    mock_train_many_epochs,
    mock_save_model,
    tmp_path,
):
    # Setup mock dataloaders
    mock_dl = MagicMock()
    mock_get_dataloaders.return_value = (mock_dl, mock_dl)

    # Setup mock exp_config returned by merge
    mock_exp = MagicMock()
    mock_exp.name = "test_mnist_run"
    mock_exp.seed = 100
    mock_exp.training.num_epochs = 1
    mock_exp.dataset.root = str(tmp_path)
    mock_exp.dataset.experiment_path = str(tmp_path / "experiments")
    mock_exp.dataset.model_path = str(tmp_path / "models")
    mock_exp.scheduler.type = "ReduceLROnPlateau"
    mock_exp.scheduler.params = {
        "patience": 2,
        "factor": 0.5,
    }
    mock_exp.optimizer.type = "AdamW"
    mock_exp.optimizer.params = {
        "lr": 0.001,
        "weight_decay": 1e-4,
    }
    mock_exp.dry_run = False

    mock_exp.model.conv_layers = []
    mock_exp.model.fc_hidden = []
    mock_exp.model.dropout = None
    mock_exp.to_dict.return_value = {}

    mock_merge.return_value = mock_exp

    mock_train_many_epochs.return_value = MagicMock()

    # Create valid dummy YAML config file
    config_dict = {}
    config_file = tmp_path / "mnist_experiment.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)

    # Call main
    main(str(config_file))

    # Verify main calls mock functions with correct args
    mock_get_dataloaders.assert_called_once()
    mock_compile.assert_called_once()
    mock_train_many_epochs.assert_called_once()
    mock_summary_writer.assert_called_once()
    mock_save_model.assert_called_once()
