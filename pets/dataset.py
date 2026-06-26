import os
import xml.etree.ElementTree as ET
from typing import Optional

import cv2
import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from pets.config import DataConfig
from pets.constants import IMAGE_SIZE


class PetsDataset(Dataset):
    def __init__(self, images, labels, boxes, transforms=None):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.boxes = torch.from_numpy(boxes).float()

        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.images[index]
        if self.transforms:
            img = self.transforms(img)
        label = self.labels[index]
        box = self.boxes[index]

        return img, box, label


def extract_xml(annot_dir, image_dir):

    tree = ET.parse(annot_dir)
    root = tree.getroot()

    img = cv2.imread(image_dir)
    assert img is not None, f"Failed to load image: {image_dir}"
    height, width = img.shape[:2]

    xmin_el = root.find(".//xmin")
    ymin_el = root.find(".//ymin")
    xmax_el = root.find(".//xmax")
    ymax_el = root.find(".//ymax")
    name_el = root.find(".//name")
    filename_el = root.find(".//filename")

    assert xmin_el is not None and xmin_el.text is not None
    assert ymin_el is not None and ymin_el.text is not None
    assert xmax_el is not None and xmax_el.text is not None
    assert ymax_el is not None and ymax_el.text is not None
    assert name_el is not None and name_el.text is not None
    assert filename_el is not None and filename_el.text is not None

    x1 = float(xmin_el.text)
    y1 = float(ymin_el.text)
    x2 = float(xmax_el.text)
    y2 = float(ymax_el.text)

    class_name = name_el.text

    filename = filename_el.text

    return filename, width, height, class_name, x1, y1, x2, y2


def xml_to_df(annot_dir, image_dir):

    xml_list = []

    image_files = os.listdir(image_dir)

    for image_file in image_files:
        filename_without_ext = ".".join(image_file.split(".")[:-1])
        annot_file = filename_without_ext + ".xml"

        annot_path = os.path.join(annot_dir, annot_file)
        img_path = os.path.join(image_dir, image_file)

        value = extract_xml(annot_path, img_path)

        xml_list.append(value)

    columns = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    xml_df = pl.DataFrame(xml_list, schema=columns)

    return xml_df


def create_train_val_split(
    data_df: pl.DataFrame, val_ratio: float = 0.2
) -> pl.DataFrame:
    n_rows = len(data_df)
    data_df = data_df.with_columns(
        (
            pl.int_range(0, n_rows).sample(fraction=1.0, with_replacement=True) / n_rows
            > val_ratio
        ).alias("is_train")
    )
    return data_df


def normalize_coordinates(
    data_df: pl.DataFrame, coordinate_cols: list[str]
) -> pl.DataFrame:
    data_df = data_df.with_columns(
        *[(pl.col(col) / IMAGE_SIZE) for col in coordinate_cols]
    )
    return data_df


def get_labels(
    data_df: pl.DataFrame, label_encoder: Optional[LabelEncoder] = None
) -> tuple[np.ndarray, LabelEncoder]:
    if label_encoder is None:
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(data_df["class"]), label_encoder
    else:
        return label_encoder.transform(data_df["class"]), label_encoder


def get_bounding_boxes(data_df: pl.DataFrame, coordinate_cols: list[str]) -> np.ndarray:
    boxes = []
    coordinates = data_df.select(coordinate_cols).to_dicts()
    for coordinate in coordinates:
        box = [coordinate[col] for col in coordinate_cols]
        boxes.append(box)
    return np.asarray(boxes)


def get_images(data_df: pl.DataFrame, image_dir: str) -> np.ndarray:
    images = []
    filenames = data_df["filename"].to_list()
    for filename in filenames:
        img = cv2.imread(os.path.join(image_dir, filename))
        assert img is not None, f"Failed to load image: {filename}"
        image = cv2.resize(img, dsize=(IMAGE_SIZE, IMAGE_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return np.asarray(images).transpose(0, 3, 1, 2)


def preprocess_dataset(config: DataConfig) -> None:
    image_dir = config.image_dir
    annot_dir = config.annot_dir

    data_df = xml_to_df(annot_dir, image_dir)
    data_df = create_train_val_split(data_df)

    train_df = data_df.filter(pl.col("is_train"))
    val_df = data_df.filter(~pl.col("is_train"))

    coordinate_cols = ["xmin", "ymin", "xmax", "ymax"]

    train_df = normalize_coordinates(train_df, coordinate_cols)
    val_df = normalize_coordinates(val_df, coordinate_cols)

    train_labels, label_encoder = get_labels(train_df)
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
