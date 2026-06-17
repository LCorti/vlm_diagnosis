from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# # # # # # # # # # # # # # # #
# General util functions      #
# # # # # # # # # # # # # # # #


def load_img(img_path: str | Path) -> Image.Image:
    return Image.open(img_path).convert("RGB")


def save_img(pil_img: Image.Image, file_path: str | Path) -> None:
    rgb_img = pil_img.convert("RGB")
    rgb_img.save(file_path)


def from_array(array: np.array) -> Image.Image:
    return Image.fromarray(array)


def resize_img(pil_img: Image.Image) -> Image.Image:
    size = get_size(pil_img.size)
    pil_img = pil_img.resize(size)
    return pil_img


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Utils functions taken from the official implementation of IETrans.  #
# https://github.com/waxnkw/IETrans-SGG.pytorch                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_size(image_size: tuple) -> tuple:
    min_size = 600
    max_size = 1000
    w, h = image_size
    size = min_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return (ow, oh)


def draw_single_box(
    pic: Image.Image, box: dict, color: str = "red", draw_info: Any = None
) -> None:
    draw = ImageDraw.Draw(pic)
    draw.rectangle(
        (
            (box["top_left_x"], box["top_left_y"]),
            (box["bottom_right_x"], box["bottom_right_y"]),
        ),
        outline=color,
    )
    if draw_info:
        draw.rectangle(
            (
                (box["top_left_x"], box["top_left_y"]),
                (box["top_left_x"] + 50, box["top_left_y"] + 10),
            ),
            fill=color,
        )
        info = draw_info
        draw.text((box["top_left_x"], box["top_left_y"]), info)


def draw_bboxes(pil_img: Image.Image, bboxes: list[dict]) -> Image.Image:
    size = get_size(pil_img.size)
    pil_img = pil_img.resize(size)

    for idx, bbox in enumerate(bboxes):
        draw_single_box(
            pil_img, bboxes[idx], draw_info=bbox["bb_label"]["bb_label_full"]
        )
    return pil_img


def list_relations(relations: list[dict]) -> list[str]:
    rel_triples = []
    for idx, rel in enumerate(relations):
        curr_rel = "{} \t: {}, {}, {}".format(
            idx,
            rel["from_concept"]["bb_label_full"],
            rel["rel_label"]["rel_label_text"],
            rel["to_concept"]["bb_label_full"],
        )
        rel_triples.append(curr_rel)
    return rel_triples


# # # # # # # # # # # # # # # #
# Image processing for MLLMs  #
# # # # # # # # # # # # # # # #


def get_image_tensor(
    image: Image.Image, input_size: int = 448, max_num: int = 12
) -> torch.Tensor:
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def build_transform(input_size: Any):
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
