import logging
import os
import xml.etree.ElementTree as ET

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import tv_tensors
from torchvision.transforms import v2

from models.constants import RESNET_INPUT_SIZE, RESNET_MEANS, RESNET_STDS
from pets import config
from pets.config import DataConfig
from pets.constants import IMAGE_SIZE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_train_transforms(config: config.DataAugmentationConfig):
    return [getattr(v2, aug.type)(**aug.params) for aug in config.dataset_augmentations]


def get_pre_augmentation_transforms():
    return [v2.ToImage(), v2.Resize((RESNET_INPUT_SIZE, RESNET_INPUT_SIZE))]


def get_post_augmentation_transforms():

    return [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=RESNET_MEANS, std=RESNET_STDS),
    ]


def get_dataloaders(config: config.ExperimentConfig):

    train_transforms = get_train_transforms(config.train_augmentations)

    pre_augmentation_transforms = get_pre_augmentation_transforms()
    post_augmentation_transforms = get_post_augmentation_transforms()

    train_data = load_dataset(
        config.dataset,
        label=config.label,
        split="train",
        dry_run=config.dry_run,
        transforms=v2.Compose(
            pre_augmentation_transforms
            + train_transforms
            + post_augmentation_transforms
        ),
    )

    test_data = load_dataset(
        config.dataset,
        label=config.label,
        split="val",
        dry_run=config.dry_run,
        transforms=v2.Compose(
            pre_augmentation_transforms + post_augmentation_transforms
        ),
    )

    if config.train_augmentations.dataloader_augmentations is not None:
        transforms = [
            getattr(v2, aug.type)(**aug.params)
            for aug in config.train_augmentations.dataloader_augmentations
        ]
        dataloader_transforms = v2.RandomChoice(transforms)

        def collate_fn(batch):
            imgs, boxes, labels = default_collate(batch)
            imgs, labels = dataloader_transforms(imgs, labels)
            return imgs, boxes, labels
    else:
        collate_fn = None

    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=config.training.batch_size,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_data,
        shuffle=False,
        batch_size=config.training.batch_size,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )

    return train_dataloader, test_dataloader


class PetsDataset(Dataset):
    def __init__(self, images, labels, boxes, transforms=None):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2)
        self.labels = torch.from_numpy(labels)
        self.height = self.images.shape[2]
        self.width = self.images.shape[3]
        self.boxes = tv_tensors.BoundingBoxes(
            torch.from_numpy(boxes),
            format="XYXY",
            canvas_size=(IMAGE_SIZE, IMAGE_SIZE),
        )  # type: ignore

        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.images[index]
        box = self.boxes[index]
        if self.transforms:
            img, box = self.transforms(img, box)

        normalized_box = box.clone()
        normalized_box[0] /= self.width
        normalized_box[1] /= self.height
        normalized_box[2] /= self.width
        normalized_box[3] /= self.height
        normalized_box_tensor = normalized_box.as_subclass(torch.Tensor).to(
            torch.float32
        )
        label = self.labels[index]

        return img.as_subclass(torch.Tensor), normalized_box_tensor, label


def parse_xml(xml_path: str) -> tuple:
    """
    Parses a Pascal VOC-style annotation XML.
    Returns: boxes (N,4) tensor [xmin, ymin, xmax, ymax], labels (list of str),
             (img_width, img_height) from the XML <size> tag.
    """
    if not os.path.exists(xml_path):
        logger.warning(f"XML file not found: {xml_path}")
        return (None, None, None, None)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    obj = root.find("object")
    bndbox = obj.find("bndbox")  # type: ignore
    xmin = float(bndbox.find("xmin").text)  # type: ignore
    ymin = float(bndbox.find("ymin").text)  # type: ignore
    xmax = float(bndbox.find("xmax").text)  # type: ignore
    ymax = float(bndbox.find("ymax").text)  # type: ignore

    return (xmin, ymin, xmax, ymax)


def get_bboxes(annot_dir: str, image_ids: list[str]) -> pl.DataFrame:

    bboxes = []

    for image_id in image_ids:
        xml_path = os.path.join(annot_dir, "xmls", f"{image_id}.xml")

        result = parse_xml(xml_path)
        if all(r is None for r in result):
            logger.warning(f"Skipping {image_id} due to missing XML file")

        bboxes.append([image_id, *result])

    columns = [
        "image_id",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    return pl.DataFrame(bboxes, schema=columns)


def get_images(image_dir: str, filenames: list[str]) -> pl.Series:
    imgs = []
    for filename in filenames:
        img_path = os.path.join(image_dir, f"{filename}.jpg")
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        imgs.append(img)

    imgs_np = np.asarray(imgs)

    return pl.Series(imgs_np, dtype=pl.Array(pl.UInt8, shape=imgs_np.shape[1:]))


def parse_text_file(filename: str) -> pl.DataFrame:
    columns = ["image_id", "class_id", "species_id", "breed_id"]
    df = pl.read_csv(filename, separator=" ", has_header=False, new_columns=columns)
    df = df.with_columns(
        (pl.col("class_id") - 1).alias("class_id").cast(pl.Int32),
        (pl.col("species_id") - 1).alias("species_id").cast(pl.Int32),
    )
    return df


def create_trainval_datasets(config: DataConfig, val_ratio: float = 0.2) -> None:
    image_dir = os.path.join(config.data_path, "images")
    annot_dir = os.path.join(config.data_path, "annotations")

    df = parse_text_file(os.path.join(annot_dir, "trainval.txt"))

    df = df.sample(fraction=1.0, shuffle=True)
    train_end_idx = int(len(df) * (1 - val_ratio))
    df = df.with_columns(
        pl.when(pl.int_range(0, pl.len()) <= train_end_idx)
        .then(pl.lit("train"))
        .otherwise(pl.lit("val"))
        .alias("split")
    )

    image_ids = df["image_id"].to_list()
    bboxes = get_bboxes(annot_dir, image_ids)
    df = df.join(bboxes, on="image_id", how="left")

    n_missing_bbox = len(df.filter(pl.col("xmin").is_null()))
    if n_missing_bbox > 0:
        logger.warning(f"Number of images without bounding box: {n_missing_bbox}")
        logger.warning("These images will be removed from the dataset.")

        df = df.filter(pl.col("xmin").is_not_null())
        image_ids = df["image_id"].to_list()

    images = get_images(image_dir, image_ids)
    df = df.with_columns(images.alias("images"))

    coordinate_cols = ["xmin", "ymin", "xmax", "ymax"]
    for split in ["train", "val"]:
        save_path = config.train_path if split == "train" else config.val_path
        split_df = df.filter(pl.col("split") == split)
        logging.info(f"Saving the {split} data to {save_path}")
        np.savez_compressed(
            save_path,
            class_id=split_df["class_id"].to_numpy(),
            species_id=split_df["species_id"].to_numpy(),
            bboxes=split_df[coordinate_cols].to_numpy(),
            images=split_df["images"].to_numpy(),
        )


def create_test_datasets(config: DataConfig) -> None:
    image_dir = os.path.join(config.data_path, "images")
    annot_dir = os.path.join(config.data_path, "annotations")

    filename = "test.txt"
    df = parse_text_file(os.path.join(annot_dir, filename))

    df = df.with_columns(pl.lit("test").alias("split"))
    image_ids = df["image_id"].to_list()
    images = get_images(image_dir, image_ids)
    df = df.with_columns(images.alias("images"))

    logger.info(f"Saving the test data to {config.test_path}")
    np.savez_compressed(
        config.test_path,
        classes=df["class_id"].to_numpy(),
        species=df["species_id"].to_numpy(),
        bboxes=np.nan,
        images=df["images"].to_numpy(),
    )


def load_preprocessed_dataset(config: DataConfig, split: str):
    if split == "train":
        path = config.train_path
    elif split == "val":
        path = config.val_path
    else:
        path = config.test_path

    data = np.load(path)
    return (
        data["class_id"],
        data["species_id"],
        data["bboxes"],
        data["images"],
    )


def load_dataset(
    config: DataConfig,
    label: str,
    split: str,
    dry_run: bool = False,
    transforms=None,
):

    classes, species, boxes, images = load_preprocessed_dataset(config, split)

    if label == "class":
        labels = classes
    elif label == "species":
        labels = species
    else:
        raise ValueError(f"Invalid label: {label}")

    if dry_run:
        labels = labels[:500]
        boxes = boxes[:500]
        images = images[:500]

    return PetsDataset(images, labels, boxes, transforms=transforms)
