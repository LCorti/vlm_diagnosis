import sys

from pathlib import Path

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.dataset_handler import DatasetHandler
from utils.data_io import make_dir
from utils.image_utils import load_img, save_img, resize_img

"""
Quick script to pre-process images for crowd-sourcing step.
Images are directly saved there. See sk_collection_webapp directory.
"""


if __name__ == "__main__":
    ds_hdl = DatasetHandler()
    ds_list = ds_hdl.get_ds_list()
    # Skip the holdout classes.
    if "vqav2_holdout" in ds_list:
        ds_list.remove("vqav2_holdout")
    base_dir = Path("..")
    out_dir = base_dir.joinpath("data", "datasets")

    for ds in ds_list:
        ds_hdl.set_curr_ds(ds)

        for c in ds_hdl.get_classes():
            ds_hdl.set_curr_class()
            curr_imgs_path = ds_hdl.get_imgs_path()
            curr_out_dir = out_dir.joinpath(ds_hdl.get_imgs_resized_path())
            make_dir(curr_out_dir)

            img_list = [img for img in base_dir.joinpath(curr_imgs_path).iterdir()]
            for img_name in img_list:
                img_path = base_dir.joinpath(curr_imgs_path, img_name)
                img = load_img(img_path)
                img = resize_img(img)
                save_img(img, curr_out_dir.joinpath(img_name))
