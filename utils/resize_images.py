import os
import sys

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from PIL import Image
from config.loaders.dataset_config_loader import DatasetConfig
from utils.data_io import make_dir

"""
Quick script to pre-process images for crowd-sourcing step.
Images are directly saved there. See sk_collection_webapp directory.
"""


def get_size(image_size):
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


def resize_image(pil_img):
    size = get_size(pil_img.size)
    pil_img = pil_img.resize(size)
    return pil_img


def save_image(pil_img, file_path):
    rgb_img = pil_img.convert("RGB")
    rgb_img.save(file_path)


def main():
    dataset_config = DatasetConfig()
    ds_list = dataset_config.get_ds_list()
    # Skip the holdout classes.
    ds_list.remove("vqav2_holdout")
    out_dir = "../data/datasets/"

    for ds in ds_list:
        ds_paths = dataset_config.get_ds_paths(ds)

        for c in ds_paths:
            curr_imgs_path = ds_paths[c]["imgs"]
            curr_out_dir = f"{out_dir}/{ds}/imgs_resized/{c}"
            make_dir(curr_out_dir)

            img_list = os.listdir(f"../{curr_imgs_path}")
            for img_name in img_list:
                img_path = f"../{curr_imgs_path}/{img_name}"
                img = Image.open(img_path)
                img = resize_image(img)
                save_image(img, f"{curr_out_dir}/{img_name}")


if __name__ == "__main__":
    main()
