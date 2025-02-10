import os
import sys

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from PIL import Image
from config_loaders.dataset_config_loader import DatasetConfig
from utils.data_io import make_dir
from utils.image_utils import load_img, save_img, resize_img

"""
Quick script to pre-process images for crowd-sourcing step.
Images are directly saved there. See sk_collection_webapp directory.
"""


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
                img = load_img(img_path)
                img = resize_img(img)
                save_img(img, f"{curr_out_dir}/{img_name}")


if __name__ == "__main__":
    main()
