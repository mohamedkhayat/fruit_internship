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

    hard_train_transforms = A.Compose(
        [
            A.RandomSizedBBoxSafeCrop(
                height=cfg.model.input_size,
                width=cfg.model.input_size,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.SafeRotate(limit=0.1, p=0.3),
            A.OneOf(
                [
                    A.RGBShift(10, 10, 10, p=0.2),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=15,
                        val_shift_limit=10,
                        p=0.6,
                    ),
                ],
                p=0.2,
            ),
            A.ConstrainedCoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.15, 0.2),
                hole_width_range=(0.15, 0.2),
                bbox_labels=[k for k in id2label.keys()],
                fill=0,
                p=0.2,
            ),
            A.Perspective(p=0.1),
            A.OneOf(
                [
                    A.Blur(blur_limit=7, p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(ensure_safe_range=True, p=0.8),
                    A.RandomToneCurve(p=0.7),
                ],
                p=0.5,
            ),
            A.CLAHE(clip_limit=2.0, p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            clip=True,
            min_visibility=cfg.min_viz,
            min_area=cfg.min_area,
            min_width=cfg.min_width,
            min_height=cfg.min_height,
        ),
    )
    safe_train_transforms = A.Compose(
        [
            A.RandomSizedBBoxSafeCrop(
                height=cfg.model.input_size,
                width=cfg.model.input_size,
                erosion_rate=0.1,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.SafeRotate(limit=0.1, p=0.2),
            A.OneOf(
                [
                    A.Blur(blur_limit=5, p=0.5),
                    A.MotionBlur(blur_limit=5, p=0.5),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                ],
                p=0.2,
            ),
            A.RandomBrightnessContrast(ensure_safe_range=True, p=0.3),
            A.CLAHE(clip_limit=1.5, p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            clip=True,
            min_visibility=cfg.min_viz,
            min_area=cfg.min_area,
            min_width=cfg.min_width,
            min_height=cfg.min_height,
        ),
    )

    transforms = {
        "train": hard_train_transforms if cfg.aug == "hard" else safe_train_transforms,
        "train_easy": safe_train_transforms,
        "test": A.Compose(
            [A.NoOp()],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["labels"],
                clip=True,
            ),
        ),
    }
    return transforms
