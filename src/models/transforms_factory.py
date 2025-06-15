import albumentations as A
import os
import cv2

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def get_transforms(input_size):
    transforms = {
        "train": A.Compose(
            [
                A.RandomSizedBBoxSafeCrop(
                    height=input_size,
                    width=input_size,
                    erosion_rate=0.8,
                    p=0.8,
                ),
                A.HorizontalFlip(p=0.5),
                A.RGBShift(15, 15, 15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(10, 25, 10, p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.2),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                clip=True,
                min_area=25,
                min_width=1,
                min_height=1,
            ),
        ),
        "test": A.Compose(
            [
                A.LongestMaxSize(max_size=input_size, p=1.0),
                A.PadIfNeeded(
                    min_height=input_size,
                    min_width=input_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    p=1.0,
                ),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                clip=True,
                min_area=1,
                min_width=1,
                min_height=1,
            ),
        ),
    }
    return transforms
