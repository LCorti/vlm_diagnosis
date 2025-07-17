import numpy as np
import sys

from pathlib import Path

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.sk_handler import SKHandler
from utils.data_io import make_dir, load_jsonl, save_jsonl

MAX_RELS = 15
MIN_RELS = 10
MIN_RELS_MMBENCH = 5
SAMPLE_SIZE = 75


def crete_rel_id(relation: dict) -> str:
    return "{}-{}-{}".format(
        relation["from_concept"]["bb_label"]["bb_label_idx"],
        relation["rel_label"]["rel_label_idx"],
        relation["to_concept"]["bb_label"]["bb_label_idx"],
    )


def get_n_rels(images: dict) -> list:
    return [(i["img_id"], len(i["rel_clusters_unique"])) for i in images]


if __name__ == "__main__":
    # Load SK config
    sk_hdl = SKHandler()
    ds_list = sk_hdl.get_sk_list()
    ds_list.remove("vqav2_holdout")

    sk_to_keep = {}

    all_n_rels = {}

    for dataset in ds_list:
        print(f"Looking at {dataset}")
        # Get current paths
        sk_hdl.set_curr_ds(dataset)
        sk_to_keep[dataset] = {}
        all_n_rels[dataset] = {}

        for ds_class in sk_hdl.get_classes():
            print(f"- Looking at {ds_class}")
            # Load merged scene graphs
            sk_hdl.set_curr_class(ds_class)
            merged_sg = load_jsonl(Path("..", sk_hdl.get_sg_merged_path()))

            # Save number of relations for plotting
            all_n_rels[dataset][ds_class] = []
            all_n_rels[dataset][ds_class] = get_n_rels(merged_sg)

            # We do not sample from LLaVa-Bench. It is small already.
            if dataset == "llava-bench":
                sk_to_keep[dataset][ds_class] = merged_sg
                print(f"--  Took {len(sk_to_keep[dataset][ds_class])} images as is.")
            else:
                sk_to_keep[dataset][ds_class] = []

                # Sample the other datasets based on thresholds. Two steps:
                # 1. Filter out images that have a certain number of relations
                candidate_images = []
                for img in merged_sg:
                    curr_n_rel = len(img["rel_clusters_unique"])
                    if dataset == "mmbench":
                        if curr_n_rel >= MIN_RELS_MMBENCH and curr_n_rel <= MAX_RELS:
                            candidate_images.append(img)
                    else:
                        if curr_n_rel >= MIN_RELS and curr_n_rel <= MAX_RELS:
                            candidate_images.append(img)
                print(f"--  Found {len(candidate_images)} candidate images.")

                # 2. Sample that subset
                if len(candidate_images) < SAMPLE_SIZE:
                    sk_to_keep[dataset][ds_class] = candidate_images
                else:
                    candidate_image_ids = [img["img_id"] for img in candidate_images]
                    np.random.shuffle(candidate_image_ids)
                    sampled_image_ids = np.random.choice(
                        candidate_image_ids, size=SAMPLE_SIZE, replace=False
                    )
                    # Find and save data given ID
                    for sampled_idx in sampled_image_ids:
                        match_img = next(
                            (sg for sg in merged_sg if sg["img_id"] == sampled_idx),
                            None,
                        )
                        if match_img:
                            sk_to_keep[dataset][ds_class].append(match_img)
                        else:
                            print("Error.")
                print(f"---   Sampled {len(sk_to_keep[dataset][ds_class])} images.")

            # Add relation ids for easier matching when crowdsourcing annotations
            for img in sk_to_keep[dataset][ds_class]:
                for rel in img["rel_clusters_unique"]:
                    rel["rel_id"] = crete_rel_id(rel)

            # Save data to disk
            out_file = sk_hdl.get_sg_crowd_path()
            out_folder = out_file.parent
            make_dir(out_folder)
            save_jsonl(sk_to_keep[dataset][ds_class], Path("..", out_file))
