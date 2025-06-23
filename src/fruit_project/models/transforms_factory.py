import albumentations as A
import os
import cv2
from omegaconf import DictConfig

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def get_transforms(cfg: DictConfig):
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
                erosion_rate=0.2,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.RGBShift(15, 15, 15, p=1.0),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=10,
                        p=1.0,
                    ),
                    # A.ToGray(p=1.0),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1.0),
                    A.RandomToneCurve(p=1.0),
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
                p=0.8,
                erosion_rate=0.2,
            ),
            A.HorizontalFlip(p=0.5),
            A.RGBShift(
                p=0.5,
                b_shift_limit=(-15.0, 15.0),
                g_shift_limit=(-15.0, 15.0),
                r_shift_limit=(-15.0, 15.0),
            ),
            A.RandomBrightnessContrast(
                p=0.5,
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
            ),
            A.HueSaturationValue(
                p=0.3,
                hue_shift_limit=(-10.0, 10.0),
                sat_shift_limit=(-25.0, 25.0),
                val_shift_limit=(-10.0, 10.0),
            ),
            A.CLAHE(p=0.2, clip_limit=(1.0, 2.0)),
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
        "test": A.Compose(
            [
                A.LongestMaxSize(max_size=cfg.model.input_size, p=1.0),
                A.PadIfNeeded(
                    min_height=cfg.model.input_size,
                    min_width=cfg.model.input_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["labels"],
                clip=True,
            ),
        ),
    }
    return transforms
