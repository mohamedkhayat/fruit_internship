# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later for me for every file

import random
import numpy as np
from torch.utils.data import Dataset
import pathlib
import cv2
import yaml
from albumentations import Compose


class DET_DS(Dataset):
    """
    A custom dataset class for object detection tasks.

    This dataset class loads images and their corresponding labels from specified directories,
    applies transformations if provided, and returns the processed image along with target annotations.

    Attributes:
        root_dir (str): The root directory containing the dataset.
        type (str): The type of dataset (e.g., 'train', 'val', 'test').
        image_dir (str): The subdirectory containing the images.
        label_dir (str): The subdirectory containing the label files.
        config_file (str): The path to the configuration file containing class names.
        transforms (Albumentations Compose, optional): A function or object to apply transformations to the images and annotations.
        input_size (int): The input size for the images (default is 224).
        image_paths (list): A list of valid image file paths.
        labels (list): A list of class names.
        id2lbl (dict): A mapping from class IDs to class names.
        lbl2id (dict): A mapping from class names to class IDs.

    Methods:
        __len__(): Returns the number of valid images in the dataset.
        __getitem__(idx): Returns the processed image and target annotations for the given index.

    Args:
        root_dir (str): The root directory containing the dataset.
        type (str): The type of dataset (e.g., 'train', 'val', 'test').
        image_dir (str): The subdirectory containing the images.
        label_dir (str): The subdirectory containing the label files.
        config_file (str): The path to the configuration file containing class names.
        transforms (Albumentations Compose, optional): A function or object to apply transformations to the images and annotations.
        input_size (int, optional): The input size for the images (default is 224).

    Raises:
        FileNotFoundError: If the configuration file or label files are not found.
        ValueError: If an image cannot be loaded or is invalid.
    """

    def __init__(
        self,
        root_dir: str,
        type: str,
        image_dir: str,
        label_dir: str,
        config_file: str,
        transforms: Compose = None,
        input_size: int = 224,
        processor=None,
    ):
        self.root_dir = pathlib.Path("data", root_dir)
        self.type = type
        self.image_dir = self.root_dir / image_dir / self.type
        self.label_dir = self.root_dir / label_dir / self.type
        self.transforms = transforms
        self.input_size = input_size
        self.config_dir = self.root_dir / config_file
        self.processor = processor
        raw_paths = sorted(list(pathlib.Path(self.image_dir).glob("*.jpg")))

        with open(self.config_dir, "r") as f:
            config = yaml.safe_load(f)

        self.labels = [name for name in config["names"]]
        self.id2lbl = dict(enumerate(self.labels))
        self.lbl2id = {v: k for k, v in self.id2lbl.items()}

        num_dropped = 0
        valid_imgs = []
        valid_labels = []
        for p in raw_paths:
            label_path = pathlib.Path(self.label_dir) / (p.stem + ".txt")
            if cv2.imread(str(p)) is not None and label_path.exists():
                valid_imgs.append(p)
                valid_labels.append(label_path)
            else:
                num_dropped += 1
                if cv2.imread(str(p)) is None:
                    print(f"[WARN] dropping bad image {p.name}")

                if not label_path.exists():
                    print(f"[WARN] dropping image {p.name} due to missing label")

        print(f"dropped {num_dropped} images from {type}")

        self.image_paths = valid_imgs
        self.label_paths = valid_labels

    def __len__(self):
        """
        Returns:
            int: The number of valid images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the processed image and target annotations for the given index.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: A tuple containing:
                - img (numpy.ndarray): The processed image.
                - target (dict): A dictionary containing target annotations, including:
                    - image_id (int): The index of the image.
                    - annotations (list): A list of dictionaries with bounding box, category ID, area, and iscrowd flag.
                    - orig_size (torch.Tensor): The original size of the image (height, width).
        """
        image_path = self.image_paths[idx]
        label_path = pathlib.Path(self.label_dir) / (image_path.stem + ".txt")

        img = cv2.imread(image_path)
        if img is None:
            new_idx = random.randrange(len(self.image_paths))
            print("img is empty")
            return self.__getitem__(new_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        height, width = img.shape[:2]
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, cx, cy, w, h = map(float, line.strip().split())

                x1 = (cx - w / 2) * width
                y1 = (cy - h / 2) * height
                box_w = w * width
                box_h = h * height
                boxes.append([x1, y1, box_w, box_h])
                labels.append(int(cls))

        boxes, labels = (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
        )
        if self.transforms:
            augmented = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["labels"]

        target = format_for_hf_processor(boxes, labels, idx)

        if hasattr(self, "processor") and self.processor:
            result = self.processor(
                images=img,
                annotations=target,
                return_tensors="pt",
                do_pad=True,
            )
            result = {k: v[0] for k, v in result.items()}
            return result
        else:
            raise AttributeError("No Processor in dataset")

    def get_raw_item(self, idx: int):
        """
        Fetches a raw, untransformed image and its annotations.
        This is a helper method for multi-sample augmentations like Mosaic.
        """
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        img = cv2.imread(str(image_path))
        if img is None:  # Handle potential bad image
            return self.get_raw_item(np.random.randint(0, len(self)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        height, width, _ = img.shape

        with open(label_path, "r") as f:
            for line in f.readlines():
                if not line.strip():
                    continue
                cls, cx, cy, w, h = map(float, line.strip().split())
                x1 = (cx - w / 2) * width
                y1 = (cy - h / 2) * height
                box_w = w * width
                box_h = h * height
                boxes.append([x1, y1, box_w, box_h])
                labels.append(int(cls))

        return img, np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def format_for_hf_processor(boxes, labels, idx):
    """Convert back to HF format"""
    return {
        "image_id": idx,
        "annotations": [
            {
                "bbox": box.tolist() if hasattr(box, "tolist") else box,
                "category_id": int(label),
                "area": float(box[2] * box[3]),
                "iscrowd": 0,
            }
            for box, label in zip(boxes, labels)
        ],
    }
