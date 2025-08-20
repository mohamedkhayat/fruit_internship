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
            A.Compose(
                [
                    A.SmallestMaxSize(max_size=cfg.model.input_size, p=1.0),
                    A.RandomSizedBBoxSafeCrop(
                        height=cfg.model.input_size, width=cfg.model.input_size, p=1.0
                    ),
                ],
                p=0.2,
            ),
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.Affine(
                        scale=(0.8, 1.2),
                        translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                        rotate=(-5, 5),
                        fill=(114, 114, 114),
                        p=0.5,
                    ),
                    A.Perspective(
                        scale=(0.02, 0.05),
                        fit_output=True,
                        fill=(114, 114, 114),
                        p=0.5,
                    ),
                ],
                p=0.4,
            ),
            A.ConstrainedCoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill=(114, 114, 114),
                bbox_labels=box_labels,
                p=0.2,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        ensure_safe_range=True,
                        p=0.5,
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.RandomToneCurve(p=0.5),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.5,
                    ),
                    A.RGBShift(
                        r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5
                    ),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=3, p=0.5),
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.25), p=0.1),
                    A.MedianBlur(blur_limit=3, p=0.2),
                ],
                p=0.1,
            ),
            A.CLAHE(clip_limit=1.5, p=0.3),
        ],
        bbox_params=bbox_params,
    )
    safe_train_transforms = A.Compose(
        [
            A.Compose(
                [
                    A.SmallestMaxSize(max_size=cfg.model.input_size, p=1.0),
                    A.RandomSizedBBoxSafeCrop(
                        height=cfg.model.input_size, width=cfg.model.input_size, p=1.0
                    ),
                ],
                p=0.2,
            ),
            A.HorizontalFlip(p=0.5),
            A.Perspective(
                scale=(0.02, 0.05), fit_output=True, fill=(114, 114, 114), p=0.1
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=3, p=0.5),
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                ],
                p=0.1,
            ),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.1),
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
        clip=True,
        filter_invalid_bboxes=True,
        min_visibility=cfg.min_viz,
        min_area=cfg.min_area,
        min_width=cfg.min_width,
        min_height=cfg.min_height,
    )
