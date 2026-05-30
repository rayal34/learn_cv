import io
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import yaml
from base.config import (
    DataAugmentationConfig,
    DataConfig,
    ExperimentConfig,
    TrainingConfig,
)
from fashion_mnist.load_data import (
    FashionMNISTDataset,
    get_dataloaders,
    load_dataset,
    load_images,
    load_labels,
)
from fashion_mnist.main import main
from models.config import SimpleCNNModelConfig
from utils.augmentation_utils import ZeroOneScale

# ==========================================
# Tests for fashion_mnist/config.py
# ==========================================


def test_data_augmentation_config():
    cfg = DataAugmentationConfig()
    assert cfg.h_flip_prob == 0.5
    assert cfg.rotate_range == [-5.0, 5.0]
    assert cfg.crop_padding == 2


def test_experiment_config_to_dict():
    dataset_cfg = DataConfig(
        root="/dummy",
        train_images_filename="train_imgs",
        train_labels_filename="train_lbls",
        test_images_filename="test_imgs",
        test_labels_filename="test_lbls",
    )
    training_cfg = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        early_stopping_patience=3,
        scheduler_patience=1,
        scheduler_factor=0.1,
        num_epochs=5,
        weight_decay=1e-5,
        early_stopping=False,
    )

    exp_cfg = ExperimentConfig(
        name="test_experiment",
        seed=123,
        dataset=dataset_cfg,
        training=training_cfg,
        model=SimpleCNNModelConfig(),
    )

    d = exp_cfg.to_dict()
    assert d["name"] == "test_experiment"
    assert d["seed"] == 123
    assert d["dataset"]["root"] == "/dummy"
    assert d["training"]["batch_size"] == 32
    assert d["data_augmentations"]["h_flip_prob"] == 0.5


# ==========================================
# Tests for fashion_mnist/load_data.py
# ==========================================


def test_zero_one_scale():
    scaler = ZeroOneScale(min_val=0.0, max_val=255.0)
    img = torch.tensor([0.0, 127.5, 255.0])
    scaled = scaler(img)
    assert torch.allclose(scaled, torch.tensor([0.0, 0.5, 1.0]))

    img = torch.zeros(size=(1, 28, 28))
    non_zero_idx = 10
    non_zero_val = 100
    img[0][non_zero_idx][non_zero_idx] = non_zero_val
    scaled = scaler(img)
    assert torch.allclose(scaled, img / 255.0)


def test_fashion_mnist_dataset():
    images = np.random.randint(0, 256, (10, 1, 28, 28)).astype(np.uint8)
    labels = np.random.randint(0, 10, (10,)).astype(np.uint8)

    dataset = FashionMNISTDataset(images, labels, transforms=None)
    assert len(dataset) == 10

    img, lbl = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 28, 28)
    assert isinstance(lbl, torch.Tensor)
    assert lbl.dtype == torch.long


def test_load_images():
    # 16-byte header + 1 * 1 * 28 * 28 = 784 bytes of dummy pixel data
    dummy_pixels = np.arange(784, dtype=np.uint8)
    header = bytes([0] * 16)
    decompressed_bytes = header + dummy_pixels.tobytes()

    with patch("gzip.open", return_value=io.BytesIO(decompressed_bytes)):
        images = load_images("dummy_file.gz")

    assert images.shape == (1, 1, 28, 28)
    assert np.array_equal(images.flatten(), dummy_pixels)


def test_load_labels():
    # 8-byte header + 5 labels
    dummy_labels = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    header = bytes([0] * 8)
    decompressed_bytes = header + dummy_labels.tobytes()

    with patch("gzip.open", return_value=io.BytesIO(decompressed_bytes)):
        labels = load_labels("dummy_lbl.gz")

    assert labels.shape == (5,)
    assert np.array_equal(labels, dummy_labels)


@patch("fashion_mnist.load_data.load_images")
@patch("fashion_mnist.load_data.load_labels")
def test_load_dataset(mock_load_labels, mock_load_images):
    mock_load_images.return_value = np.zeros((10, 1, 28, 28), dtype=np.uint8)
    mock_load_labels.return_value = np.ones((10,), dtype=np.uint8)

    dataset_cfg = DataConfig(
        root="/dummy",
        train_images_filename="train_imgs",
        train_labels_filename="train_lbls",
        test_images_filename="test_imgs",
        test_labels_filename="test_lbls",
    )

    dataset = load_dataset(dataset_cfg, train=True)
    assert len(dataset) == 10
    assert dataset.images.shape == (10, 1, 28, 28)


@patch("fashion_mnist.load_data.load_dataset")
def test_get_dataloaders(mock_load_dataset):
    # Setup dummy datasets
    images = np.zeros((8, 1, 28, 28), dtype=np.uint8)
    labels = np.zeros((8,), dtype=np.uint8)
    dummy_dataset = FashionMNISTDataset(images, labels)

    mock_load_dataset.return_value = dummy_dataset

    dataset_cfg = DataConfig(
        root="/dummy",
        train_images_filename="train_imgs",
        train_labels_filename="train_lbls",
        test_images_filename="test_imgs",
        test_labels_filename="test_lbls",
    )
    training_cfg = TrainingConfig(
        learning_rate=0.01,
        batch_size=4,
        early_stopping_patience=2,
        scheduler_patience=1,
        scheduler_factor=0.5,
        num_epochs=2,
        weight_decay=1e-4,
        early_stopping=False,
    )

    exp_cfg = ExperimentConfig(
        name="test_dl", dataset=dataset_cfg, training=training_cfg
    )

    train_dl, test_dl = get_dataloaders(exp_cfg)
    assert isinstance(train_dl, torch.utils.data.DataLoader)
    assert isinstance(test_dl, torch.utils.data.DataLoader)

    # 8 samples, batch_size=4 -> 2 batches
    assert len(train_dl) == 2
    assert len(test_dl) == 2


# ==========================================
# Tests for fashion_mnist/main.py
# ==========================================


@patch("fashion_mnist.main.train_utils.train_many_epochs")
@patch("fashion_mnist.main.load_data.get_dataloaders")
@patch("fashion_mnist.main.SummaryWriter")
@patch("fashion_mnist.main.OmegaConf.merge")
def test_fashion_mnist_main(
    mock_merge,
    mock_summary_writer,
    mock_get_dataloaders,
    mock_train_many_epochs,
    tmp_path,
):
    # Setup mock dataloaders
    mock_dl = MagicMock()
    mock_get_dataloaders.return_value = (mock_dl, mock_dl)

    # Setup mock exp_config returned by merge
    mock_exp = MagicMock()
    mock_exp.name = "test_fashion_mnist_run"
    mock_exp.seed = 42
    mock_exp.training.learning_rate = 0.001
    mock_exp.training.weight_decay = 1e-4
    mock_exp.training.scheduler_patience = 2
    mock_exp.training.scheduler_factor = 0.5
    mock_exp.training.early_stopping = True
    mock_exp.training.early_stopping_patience = 3
    mock_exp.training.num_epochs = 2
    mock_exp.dataset.root = str(tmp_path)
    mock_exp.dataset.experiment_path = str(tmp_path / "experiments")
    mock_exp.dataset.model_path = str(tmp_path / "models")
    mock_exp.dry_run = False

    # Small architecture
    mock_exp.model.conv_layers = []
    mock_exp.model.fc_hidden = []
    mock_exp.model.dropout = None
    mock_exp.to_dict.return_value = {}

    mock_merge.return_value = mock_exp

    mock_train_many_epochs.return_value = MagicMock()

    # Create valid dummy YAML config file
    config_dict = {}
    config_file = tmp_path / "fashion_experiment.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)

    # Call main
    main(str(config_file))

    # Verify main calls mock functions with correct args
    mock_get_dataloaders.assert_called_once()
    mock_train_many_epochs.assert_called_once()
    mock_summary_writer.assert_called_once()
