import albumentations as A
import os
import cv2

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
supported_models = ["vit_base", "tiny_vit", "efficientnet_v2_s"]


def get_transforms(processor, model):
    if model == "vit_base":
        height, width = processor.size["height"], processor.size["width"]
        mean, std = processor.image_mean, processor.image_std
        interp = cv2.INTER_NEAREST
        do_resize = processor.do_resize
        do_rescale = processor.do_rescale
        rescale_factor = processor.rescale_factor
        resample = processor.resample
        do_normalize = processor.do_normalize
        interp = cv2.INTER_CUBIC if resample == 2 else cv2.INTER_LINEAR

    elif model == "tiny_vit":
        interp_map = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
        interp = interp_map.get(processor["interpolation"], cv2.INTER_LINEAR)
        height, width = processor["input_size"][-2:]
        resize_size = int(height / processor["crop_pct"])
        mean, std = processor["mean"], processor["std"]
        interp = processor["interpolation"]
        interp = interp_map[interp]
        do_resize = True
        do_rescale = True
        rescale_factor = 1.0 / 255.0
        resample = True
        do_normalize = True

    elif model in ["efficientnet_v2_s", "efficientnet_b0"]:
        tv_preset = processor.transforms()
        height = width = tv_preset.crop_size[0]
        resize_size = tv_preset.resize_size[0]
        mean = tv_preset.mean
        std = tv_preset.std
        interp = cv2.INTER_LINEAR
        do_resize = True
        do_rescale = True
        rescale_factor = 1.0 / 255.0
        resample = True
        do_normalize = True
    else:
        raise ValueError(
            f"model you gave does not have a config, add it to transforms factory"
        )

    train_head_transforms = []
    test_head_transforms = []

    if do_resize:
        train_head_transforms.append(
            A.RandomResizedCrop(
                size=(height, width),
                scale=(0.8, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
                interpolation=interp,
            ),
        )

        if model == "vit_base":
            test_head_transforms = [
                A.Resize(height=height, width=width, interpolation=interp),
            ]

        else:
            test_head_transforms.extend(
                [
                    A.SmallestMaxSize(max_size=resize_size, interpolation=interp),
                    A.CenterCrop(height=height, width=width),
                ]
            )

    tail_transforms = []

    max_value_norm = 255.0

    if do_rescale:
        tail_transforms.append(A.ToFloat(max_value=round(1 / rescale_factor)))
        max_value_norm = 1.0

    if do_normalize:
        tail_transforms.append(A.Normalize(mean, std, max_pixel_value=max_value_norm))

    tail_transforms.append(A.ToTensorV2())

    transforms = {
        "train": A.Compose(
            train_head_transforms
            + [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.8),
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(0.05, 0.15),
                    hole_width_range=(0.05, 0.15),
                    fill=0,
                    p=0.7,
                ),
                A.GaussianBlur(
                    sigma_limit=(0.2, 0.5),
                    blur_limit=0,
                    p=0.7,
                ),
            ]
            + tail_transforms
        ),
        "test": A.Compose(test_head_transforms + tail_transforms),
    }
    return transforms
