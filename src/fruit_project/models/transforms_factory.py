# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Dict
import albumentations as A
import os
from omegaconf import DictConfig

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def get_transforms(cfg: DictConfig, id2label: Dict[int, str]) -> Dict[str, A.Compose]:
    """
    Generates a dictionary of Albumentations transformations for training and testing.
    Args:
        cfg (DictConfig): Configuration object containing the following attributes:
    Returns:
        dict: A dictionary with keys "train" and "test"
    """
    bbox_params = get_bbox_params(cfg)
    box_labels = [k for k in id2label.keys()]
    hard_train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                rotate=(-5, 5),
                fill=(114, 114, 114),
                p=0.3,
            ),
            A.Perspective(
                scale=(0.01, 0.035), fit_output=True, fill=(114, 114, 114), p=0.15
            ),
            A.ConstrainedCoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.05, 0.25),
                hole_width_range=(0.05, 0.25),
                fill=(0, 0, 0),
                bbox_labels=box_labels,
                p=0.3,
            ),
            A.OneOf(
                [
                    A.RGBShift(10, 10, 10, p=0.2),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=10,
                        val_shift_limit=10,
                        p=0.6,
                    ),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=5, p=0.5),
                    A.MotionBlur(blur_limit=5, p=0.5),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        ensure_safe_range=True,
                        p=0.5,
                    ),
                    A.RandomToneCurve(p=0.7),
                ],
                p=0.3,
            ),
            A.CLAHE(clip_limit=2.0, p=0.3),
        ],
        bbox_params=bbox_params,
    )
    safe_train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.Blur(blur_limit=3, p=0.5),
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                ],
                p=0.1,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, ensure_safe_range=True, p=0.2
            ),
            A.CLAHE(clip_limit=1.5, p=0.1),
        ],
        bbox_params=bbox_params,
    )

    transforms = {
        "train": hard_train_transforms if cfg.aug == "hard" else safe_train_transforms,
        "train_easy": safe_train_transforms,
        "test": A.Compose([A.NoOp()], bbox_params=bbox_params),
    }
    return transforms


def get_bbox_params(cfg):
    return A.BboxParams(
        format="coco",
        label_fields=["labels"],
        clip=False,
        filter_invalid_bboxes=True,
        min_visibility=cfg.min_viz,
        min_area=cfg.min_area,
        min_width=cfg.min_width,
        min_height=cfg.min_height,
    )
