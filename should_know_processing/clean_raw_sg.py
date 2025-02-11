import os
import sys

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.dataset_config_loader import DatasetConfig
from config_loaders.sk_config_loader import SKConfig
from pathlib import Path
from utils.data_io import make_dir, load_json, save_jsonl
from utils.image_utils import load_img


def get_unique_concepts(raw_sg):
    unique_concepts_bbox = {}
    unique_concepts_rel = {}

    for img in raw_sg["imgs"]:
        concepts_bbox = []
        concepts_rel = {"from": [], "to": []}

        # Check bounding boxes
        for bbox in img["bboxes"]:
            concepts_bbox.append(bbox["bb_label"]["bb_label_text"])
        concepts_bbox = list(set(concepts_bbox))
        unique_concepts_bbox[img["img_id"]] = concepts_bbox

        # Check relations
        for rel in img["relations"]:
            concepts_rel["from"].append(rel["from_concept"]["bb_label_text"])
            concepts_rel["to"].append(rel["to_concept"]["bb_label_text"])
        concepts_rel["from"] = list(set(concepts_rel["from"]))
        concepts_rel["to"] = list(set(concepts_rel["to"]))
        unique_concepts_rel[img["img_id"]] = concepts_rel
    return unique_concepts_bbox, unique_concepts_rel


def get_all_concepts(unique_concepts_bbox, unique_concepts_rel):
    all_concepts_in_rel = {}
    all_concepts_not_in_rel = {}

    for img in unique_concepts_bbox:
        concepts_rel_from = unique_concepts_rel[img]["from"]
        concepts_rel_to = unique_concepts_rel[img]["to"]
        all_concepts_rel_img = set(concepts_rel_from + concepts_rel_to)

        concepts_in_rel = [
            c for c in unique_concepts_bbox[img] if c in all_concepts_rel_img
        ]
        concepts_not_in_rel = [
            c
            for c in unique_concepts_bbox[img]
            if c not in all_concepts_rel_img
        ]

        all_concepts_in_rel[img] = concepts_in_rel
        all_concepts_not_in_rel[img] = concepts_not_in_rel
    return all_concepts_in_rel, all_concepts_not_in_rel


if __name__ == "__main__":
    # Load dataset config
    ds_config = DatasetConfig()
    ds_list = ds_config.get_ds_list()
    ds_list.remove("vqav2_holdout")

    # Load SK config
    sk_config = SKConfig()

    for dataset in ds_list:
        # Get current paths
        ds_paths = ds_config.get_ds_paths(dataset)
        sk_paths = sk_config.get_sk_paths(dataset)

        for ds_class in ds_paths:
            # Load images
            # base_img_dir = f"../{ds_paths[ds_class]['imgs']}"
            # img_dict = {
            #     Path(img_file).stem: load_img(f"{base_img_dir}/{img_file}")
            #     for img_file in os.listdir(base_img_dir)
            # }

            # Load Raw scene graphs
            base_dir = sk_paths[ds_class]["dir"]
            raw_sg_file = sk_paths[ds_class]["sg_raw"]
            raw_sg = load_json(f"../{base_dir}/{raw_sg_file}")

            # Get lists of concepts
            unique_c_bbox, unique_c_rel = get_unique_concepts(raw_sg)
            all_c_in_rel, all_c_not_in_rel = get_all_concepts(
                unique_c_bbox, unique_c_rel
            )

            for img in raw_sg["imgs"]:
                in_rel = []
                not_in_rel = []

                # Matched (all_concepts_in_rel)
                for c in all_c_in_rel[img["img_id"]]:
                    for bbox in img["bboxes"]:
                        if bbox["bb_label"]["bb_label_text"] == c:
                            in_rel.append(bbox)

                # Not matched (all_concepts_not_in_rel)
                for c in all_c_not_in_rel[img["img_id"]]:
                    for bbox in img["bboxes"]:
                        if bbox["bb_label"]["bb_label_text"] == c:
                            not_in_rel.append(bbox)

                img["bboxes_rel"] = in_rel
                img["bboxes_not_rel"] = not_in_rel
                # Getting rid of the 'bboxes' field as they were just processed
                del img["bboxes"]

            # Save current batch of clean scene graphs
            out_folder = f"../{base_dir}/scene_graphs"
            make_dir(out_folder)

            save_jsonl(
                raw_sg["imgs"],
                f"../{base_dir}/{sk_paths[ds_class]['sg_clean']}",
            )
