import os
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import polars as pl
import torch
from models.constants import RESNET_INPUT_SIZE, RESNET_MEANS, RESNET_STDS
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import tv_tensors
from torchvision.transforms import v2

from pets import config
from pets.config import DataConfig
from pets.constants import IMAGE_SIZE


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
        train=True,
        dry_run=config.dry_run,
        transforms=v2.Compose(
            pre_augmentation_transforms
            + train_transforms
            + post_augmentation_transforms
        ),
    )

    test_data = load_dataset(
        config.dataset,
        train=False,
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

        canvas_size = getattr(box, "canvas_size", self.boxes.canvas_size)
        height, width = canvas_size
        normalized_box = box.clone()
        normalized_box[0] /= width
        normalized_box[1] /= height
        normalized_box[2] /= width
        normalized_box[3] /= height
        normalized_box_tensor = normalized_box.as_subclass(torch.Tensor).to(
            torch.float32
        )
        label = self.labels[index]

        return img.as_subclass(torch.Tensor), normalized_box_tensor, label


def parse_voc_xml(xml_path):
    """
    Parses a Pascal VOC-style annotation XML.
    Returns: boxes (N,4) tensor [xmin, ymin, xmax, ymax], labels (list of str),
             (img_width, img_height) from the XML <size> tag.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    img_width = int(size_node.find("width").text)  # type: ignore
    img_height = int(size_node.find("height").text)  # type: ignore

    obj = root.find("object")
    name = obj.find("name").text  # type: ignore
    bnd = obj.find("bndbox")  # type: ignore
    xmin = float(bnd.find("xmin").text)  # type: ignore
    ymin = float(bnd.find("ymin").text)  # type: ignore
    xmax = float(bnd.find("xmax").text)  # type: ignore
    ymax = float(bnd.find("ymax").text)  # type: ignore

    return {
        "box": (xmin, ymin, xmax, ymax),
        "label": name,
        "size": (img_width, img_height),
    }


def xml_to_df(annot_dir, image_dir):

    data_list = []

    image_files = sorted(os.listdir(image_dir))

    for image_file in image_files:
        filename_without_ext = ".".join(image_file.split(".")[:-1])
        annot_file = filename_without_ext + ".xml"

        annot_path = os.path.join(annot_dir, annot_file)

        result = parse_voc_xml(annot_path)
        box = result["box"]
        label = result["label"]
        width = result["size"][0]
        height = result["size"][1]

        data_list.append([image_file, width, height, label, *box])

    columns = [
        "filename",
        "width",
        "height",
        "label",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    xml_df = pl.DataFrame(data_list, schema=columns)

    return xml_df


def create_train_val_split(
    data_df: pl.DataFrame, val_ratio: float = 0.2, stratify: bool = True, seed: int = 42
) -> pl.DataFrame:
    n_rows = len(data_df)
    stratify_by = data_df["label"].to_numpy() if stratify else None

    train_indices, _ = train_test_split(
        np.arange(n_rows),
        test_size=val_ratio,
        stratify=stratify_by,
        random_state=seed,
    )

    is_train = np.zeros(n_rows, dtype=bool)
    is_train[train_indices] = True
    data_df = data_df.with_columns(pl.Series(name="is_train", values=is_train))
    return data_df


def get_labels(
    data_df: pl.DataFrame, label_encoder: Optional[LabelEncoder] = None
) -> tuple[np.ndarray, LabelEncoder]:
    if label_encoder is None:
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(data_df["label"]), label_encoder
    else:
        return label_encoder.transform(data_df["label"]), label_encoder


def get_bounding_boxes(data_df: pl.DataFrame, coordinate_cols: list[str]) -> np.ndarray:
    boxes = []
    coordinates = data_df.select(coordinate_cols).to_dicts()
    for coordinate in coordinates:
        box = [coordinate[col] for col in coordinate_cols]
        boxes.append(box)
    return np.asarray(boxes)


def get_images(data_df: pl.DataFrame, image_dir: str) -> np.ndarray:
    imgs = []
    filenames = data_df["filename"].to_list()
    for filename in filenames:
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        imgs.append(img)
    return np.asarray(imgs)  # (N, H, W, C)


def preprocess_dataset(config: DataConfig) -> None:
    image_dir = config.image_dir
    annot_dir = config.annot_dir

    data_df = xml_to_df(annot_dir, image_dir)
    data_df = create_train_val_split(data_df)

    train_df = data_df.filter(pl.col("is_train"))
    val_df = data_df.filter(~pl.col("is_train"))

    coordinate_cols = ["xmin", "ymin", "xmax", "ymax"]

    train_labels, label_encoder = get_labels(train_df, None)
    val_labels, _ = get_labels(val_df, label_encoder)

    train_boxes = get_bounding_boxes(train_df, coordinate_cols)
    val_boxes = get_bounding_boxes(val_df, coordinate_cols)

    train_images = get_images(train_df, image_dir)
    val_images = get_images(val_df, image_dir)

    np.savez_compressed(
        config.train_path,
        labels=train_labels,
        boxes=train_boxes,
        images=train_images,
    )
    np.savez_compressed(
        config.val_path,
        labels=val_labels,
        boxes=val_boxes,
        images=val_images,
    )


def load_preprocessed_dataset(config: DataConfig, train: bool = True):
    if train:
        path = config.train_path
    else:
        path = config.val_path

    data = np.load(path)
    return data["labels"], data["boxes"], data["images"]


def load_dataset(
    config: DataConfig, train: bool = True, dry_run: bool = False, transforms=None
):

    data_exists = os.path.exists(config.train_path if train else config.val_path)

    if data_exists:
        print("Loading preprocessed dataset...")
        labels, boxes, images = load_preprocessed_dataset(config, train)
    else:
        print("Preprocessing dataset...")
        preprocess_dataset(config)
        labels, boxes, images = load_preprocessed_dataset(config, train)

    if dry_run:
        labels = labels[:500]
        boxes = boxes[:500]
        images = images[:500]

    return PetsDataset(images, labels, boxes, transforms=transforms)
