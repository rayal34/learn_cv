from unittest.mock import MagicMock, patch

import numpy as np
import torch
import yaml
from base.config import (
    DataAugmentationConfig,
    DataConfig,
    EarlyStoppingConfig,
    GenericConfig,
    SchedulerConfig,
    TrainingConfig,
)
from cifar.config import ExperimentConfig
from cifar.load_data import (
    Cifar100Dataset,
    get_dataloaders,
    get_label_mappings,
    load_dataset,
    unpickle,
)
from cifar.main import main
from cifar.utils import get_optimizer_and_scheduler
from models.config import (
    ConvSpec,
    ResNetShallowModelConfig,
    ResNetStemConfig,
)
from omegaconf import MISSING


def test_cifar_experiment_config_to_dict():
    dataset_cfg = DataConfig(
        root="/dummy_cifar",
        num_workers=0,
        pin_memory=False,
        train_images_filename="train",
        train_labels_filename="train",
        test_images_filename="test",
        test_labels_filename="test",
    )
    training_cfg = TrainingConfig(
        batch_size=128,
        num_epochs=3,
    )
    train_augmentations_cfg = DataAugmentationConfig(
        dataset_augmentations=[],
        dataloader_augmentations=None,
    )
    scheduler_cfg = SchedulerConfig(
        type="OneCycleLR",
        params={"max_lr": 0.01},
    )
    optimizer_cfg = GenericConfig(
        type="AdamW",
        params={"lr": 0.001, "weight_decay": 1e-4},
    )
    early_stopping_cfg = EarlyStoppingConfig(
        patience=10,
        min_delta=0.0,
        higher_is_better=True,
    )
    stem_conv = ConvSpec(out_channels=16, kernel_size=3, padding=1, pool=None, stride=1)
    stem_config = ResNetStemConfig(conv=stem_conv, maxpool=None)
    model_cfg = ResNetShallowModelConfig(stem=stem_config, layers=[])

    exp_cfg = ExperimentConfig(
        name="test_cifar_exp",
        seed=42,
        dry_run=True,
        dataset=dataset_cfg,
        train_augmentations=train_augmentations_cfg,
        training=training_cfg,
        scheduler=scheduler_cfg,
        optimizer=optimizer_cfg,
        early_stopping=early_stopping_cfg,
        model=model_cfg,
    )

    d = exp_cfg.to_dict()
    assert d["name"] == "test_cifar_exp"
    assert d["seed"] == 42
    assert d["dry_run"] is True
    assert d["dataset"]["root"] == "/dummy_cifar"
    assert d["training"]["batch_size"] == 128
    assert d["early_stopping"]["patience"] == 10


def test_cifar_dataset():
    # 2 images, size 3 channels, 32 height, 32 width
    imgs = np.zeros((2, 3, 32, 32), dtype=np.uint8)
    labels = ["apple", "banana"]
    label_int_mapping = {"apple": 0, "banana": 1}

    dataset = Cifar100Dataset(imgs, labels, label_int_mapping)
    assert len(dataset) == 2

    img, lbl = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 32, 32)
    assert lbl == 0

    # Cover transforms path
    dataset_with_transforms = Cifar100Dataset(
        imgs, labels, label_int_mapping, transforms=lambda x: x + 1.0
    )
    img_transformed, _ = dataset_with_transforms[0]
    assert torch.allclose(img_transformed, torch.ones(3, 32, 32))


@patch("builtins.open")
@patch("pickle.load")
def test_unpickle(mock_pickle_load, mock_open):
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    mock_pickle_load.return_value = {"key": "value"}

    data = unpickle("dummy_path")
    assert data == {"key": "value"}
    mock_pickle_load.assert_called_once_with(mock_file, encoding="latin1")


@patch("cifar.load_data.unpickle")
def test_get_label_mappings(mock_unpickle):
    mock_unpickle.return_value = {"fine_label_names": ["apple", "banana"]}
    label_int_mapping, int_label_mapping = get_label_mappings("dummy_path")
    assert label_int_mapping == {"apple": 0, "banana": 1}
    assert int_label_mapping == {0: "apple", 1: "banana"}


@patch("cifar.load_data.unpickle")
@patch("cifar.load_data.get_label_mappings")
def test_load_dataset(mock_get_label_mappings, mock_unpickle):
    mock_get_label_mappings.return_value = (
        {"apple": 0, "banana": 1},
        {0: "apple", 1: "banana"},
    )
    # 4 images
    dummy_data = np.zeros((4, 3072), dtype=np.uint8)
    mock_unpickle.return_value = {
        "data": dummy_data,
        "fine_labels": [0, 1, 0, 1],
    }

    config_data = DataConfig(
        root="/dummy_cifar",
        num_workers=0,
        pin_memory=False,
    )

    dataset = load_dataset(config_data, train=True, dry_run=True)
    # dry_run clips to 500, we only passed 4 so we get 4
    assert len(dataset) == 4
    assert dataset.imgs.shape == (4, 3, 32, 32)


@patch("cifar.load_data.load_dataset")
def test_get_dataloaders(mock_load_dataset):
    imgs = np.zeros((8, 3, 32, 32), dtype=np.uint8)
    labels = ["apple"] * 8
    label_int_mapping = {"apple": 0}
    dummy_dataset = Cifar100Dataset(imgs, labels, label_int_mapping)

    mock_load_dataset.return_value = dummy_dataset

    dataset_cfg = DataConfig(
        root="/dummy_cifar",
        num_workers=0,
        pin_memory=False,
    )
    training_cfg = TrainingConfig(
        batch_size=4,
        num_epochs=2,
    )
    # Test dataloader with dataloader-level mixup/cutmix transforms
    train_augmentations_cfg = DataAugmentationConfig(
        dataset_augmentations=[],
        dataloader_augmentations=[
            GenericConfig(type="RandomErasing", params={"p": 0.5})
        ],
    )
    exp_cfg = ExperimentConfig(
        model=MISSING,
        scheduler=MISSING,
        optimizer=MISSING,
        name="test_cifar_dl",
        seed=42,
        dry_run=False,
        dataset=dataset_cfg,
        train_augmentations=train_augmentations_cfg,
        training=training_cfg,
    )

    train_dl, test_dl = get_dataloaders(exp_cfg)
    assert isinstance(train_dl, torch.utils.data.DataLoader)
    assert isinstance(test_dl, torch.utils.data.DataLoader)

    # 8 samples, batch_size=4 -> 2 batches
    assert len(train_dl) == 2
    assert len(test_dl) == 2

    # Draw a batch to test collate_fn/RandomChoice dataloader transform
    batch_img, batch_lbl = next(iter(train_dl))
    assert batch_img.shape == (4, 3, 32, 32)
    assert batch_lbl.shape == (4,)

    # Cover else branch where dataloader_augmentations is None
    exp_cfg_no_dl_aug = ExperimentConfig(
        model=MISSING,
        scheduler=MISSING,
        optimizer=MISSING,
        name="test_cifar_dl",
        seed=42,
        dry_run=False,
        dataset=dataset_cfg,
        train_augmentations=DataAugmentationConfig(
            dataset_augmentations=[], dataloader_augmentations=None
        ),
        training=training_cfg,
    )
    from torch.utils.data import default_collate

    train_dl2, _ = get_dataloaders(exp_cfg_no_dl_aug)
    assert train_dl2.collate_fn is default_collate


def test_get_optimizer_and_scheduler():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 32 * 32, 10),
    )

    optimizer_cfg = GenericConfig(
        type="AdamW",
        params={"lr": 0.001, "weight_decay": 1e-4},
    )
    scheduler_cfg = SchedulerConfig(
        type="OneCycleLR",
        params={"max_lr": 0.01},
    )
    training_cfg = TrainingConfig(
        batch_size=4,
        num_epochs=5,
    )
    exp_cfg = ExperimentConfig(
        model=MISSING,
        dataset=MISSING,
        train_augmentations=MISSING,
        name="test_cifar_opt",
        seed=42,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        training=training_cfg,
    )

    # Mock dataloader with 10 batches
    mock_dl = MagicMock()
    mock_dl.__len__.return_value = 10

    optimizer, scheduler = get_optimizer_and_scheduler(model, exp_cfg, mock_dl)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)


@patch("cifar.main.train_utils.save_model")
@patch("cifar.main.train_utils.train_many_epochs")
@patch("cifar.main.load_data.get_dataloaders")
@patch("cifar.main.SummaryWriter")
@patch("cifar.main.torch.compile")
@patch("cifar.main.OmegaConf.merge")
def test_cifar_main(
    mock_merge,
    mock_compile,
    mock_summary_writer,
    mock_get_dataloaders,
    mock_train_many_epochs,
    mock_save_model,
    tmp_path,
):
    mock_dl = MagicMock()
    mock_dl.__len__.return_value = 10
    mock_get_dataloaders.return_value = (mock_dl, mock_dl)

    mock_exp = MagicMock()
    mock_exp.name = "test_cifar_run"
    mock_exp.seed = 42
    mock_exp.training.num_epochs = 2
    mock_exp.dataset.root = str(tmp_path)
    mock_exp.dataset.experiment_path = str(tmp_path / "experiments")
    mock_exp.dataset.model_path = str(tmp_path / "models")
    mock_exp.scheduler.type = "OneCycleLR"
    mock_exp.scheduler.params = {
        "max_lr": 0.01,
        "pct_start": 0.1,
    }
    mock_exp.scheduler.update_freq = "step"
    mock_exp.optimizer.type = "AdamW"
    mock_exp.optimizer.params = {
        "lr": 0.001,
        "weight_decay": 1e-4,
    }
    mock_exp.dry_run = False
    mock_exp.early_stopping = MagicMock()
    mock_exp.early_stopping.patience = 5
    mock_exp.early_stopping.min_delta = 0.0
    mock_exp.early_stopping.higher_is_better = True

    # Small architecture
    mock_exp.model.stem.conv.out_channels = 16
    mock_exp.model.stem.conv.kernel_size = 3
    mock_exp.model.stem.conv.padding = 1
    mock_exp.model.stem.conv.stride = 1
    mock_exp.model.stem.maxpool = None
    mock_layer = MagicMock()
    mock_layer.out_channels = 16
    mock_layer.blocks = 1
    mock_layer.stride = 1
    mock_exp.model.layers = [mock_layer]
    mock_exp.to_dict.return_value = {}

    mock_merge.return_value = mock_exp

    mock_train_many_epochs.return_value = MagicMock()

    # Create dummy YAML config file
    config_dict = {}
    config_file = tmp_path / "cifar_experiment.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)

    # Call main
    main(str(config_file))

    # Verify main calls mock functions with correct args
    mock_get_dataloaders.assert_called_once()
    mock_train_many_epochs.assert_called_once()
    mock_summary_writer.assert_called_once()
    mock_save_model.assert_not_called()  # Because early_stopping is not None


@patch("cifar.main.train_utils.save_model")
@patch("cifar.main.train_utils.train_many_epochs")
@patch("cifar.main.load_data.get_dataloaders")
@patch("cifar.main.SummaryWriter")
@patch("cifar.main.torch.compile")
@patch("cifar.main.OmegaConf.merge")
@patch("cifar.main.torch.cuda.is_available")
@patch("cifar.main.torch.backends.mps.is_available")
def test_cifar_main_other_branches(
    mock_mps_avail,
    mock_cuda_avail,
    mock_merge,
    mock_compile,
    mock_summary_writer,
    mock_get_dataloaders,
    mock_train_many_epochs,
    mock_save_model,
    tmp_path,
):
    mock_cuda_avail.return_value = False
    mock_mps_avail.return_value = False

    mock_dl = MagicMock()
    mock_dl.__len__.return_value = 10
    mock_get_dataloaders.return_value = (mock_dl, mock_dl)

    mock_exp = MagicMock()
    mock_exp.name = "test_cifar_run_other"
    mock_exp.seed = 42
    mock_exp.training.num_epochs = 2
    mock_exp.dataset.root = str(tmp_path)
    mock_exp.dataset.experiment_path = str(tmp_path / "experiments")
    mock_exp.dataset.model_path = str(tmp_path / "models")
    mock_exp.scheduler.type = "OneCycleLR"
    mock_exp.scheduler.params = {
        "max_lr": 0.01,
        "pct_start": 0.1,
    }
    mock_exp.scheduler.update_freq = "step"
    mock_exp.optimizer.type = "AdamW"
    mock_exp.optimizer.params = {
        "lr": 0.001,
        "weight_decay": 1e-4,
    }
    mock_exp.dry_run = True
    mock_exp.early_stopping = None

    # Small architecture
    mock_exp.model.stem.conv.out_channels = 16
    mock_exp.model.stem.conv.kernel_size = 3
    mock_exp.model.stem.conv.padding = 1
    mock_exp.model.stem.conv.stride = 1
    mock_exp.model.stem.maxpool = None
    mock_layer = MagicMock()
    mock_layer.out_channels = 16
    mock_layer.blocks = 1
    mock_layer.stride = 1
    mock_exp.model.layers = [mock_layer]
    mock_exp.to_dict.return_value = {}

    mock_merge.return_value = mock_exp

    mock_train_many_epochs.return_value = MagicMock()

    # Create dummy YAML config file
    config_dict = {}
    config_file = tmp_path / "cifar_experiment_other.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)

    # Call main
    main(str(config_file), profile=False)

    # Verify calls
    mock_get_dataloaders.assert_called_once()
    mock_train_many_epochs.assert_called_once()
    mock_save_model.assert_called_once()
