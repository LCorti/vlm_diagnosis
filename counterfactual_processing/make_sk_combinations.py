import numpy as np
import os
import sys

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.sk_config_loader import SKConfig
from pathlib import Path
from utils.data_io import load_json, load_jsonl, make_dir, save_json
from utils.graph_utils import compute_concepts_powerset


def get_concept_dict(relations):
    concepts = {}

    for rel in relations:
        from_concept = rel["from_concept"]
        to_concept = rel["to_concept"]

        if from_concept["bb_label"]["bb_label_full"] not in concepts:
            concepts[from_concept["bb_label"]["bb_label_full"]] = {
                "top_left_x": from_concept["top_left_x"],
                "top_left_y": from_concept["top_left_y"],
                "bottom_right_x": from_concept["bottom_right_x"],
                "bottom_right_y": from_concept["bottom_right_y"],
                "width": from_concept["width"],
                "height": from_concept["height"],
            }
        if to_concept["bb_label"]["bb_label_full"] not in concepts:
            concepts[to_concept["bb_label"]["bb_label_full"]] = {
                "top_left_x": to_concept["top_left_x"],
                "top_left_y": to_concept["top_left_y"],
                "bottom_right_x": to_concept["bottom_right_x"],
                "bottom_right_y": to_concept["bottom_right_y"],
                "width": to_concept["width"],
                "height": to_concept["height"],
            }
    return concepts


def lookup_sg(img_id, sk_list):
    return next((sk_data for sk_data in sk_list if sk_data["img_id"] == img_id), None)


# def sample_powerset(powerset, ratio=0.1):
#     sample = {}
#     for size in powerset:
#         if len(powerset[size]) <= 100:
#             sample[size] = powerset[size]
#         else:
#             num_to_keep = int(len(powerset[size]) * ratio)
#             positions = list(range(len(powerset[size])))
#             index = np.random.choice(positions, num_to_keep, replace=False)
#             # print(len(powerset[size]))
#             # print(index)
#             # print(type(powerset[size]))
#             sample[size] = [powerset[size][i] for i in index]
#     return sample


def sample_powerset(powerset, max_sample=100):
    sample = {}
    for size in powerset:
        if len(powerset[size]) <= max_sample:
            sample[size] = powerset[size]
        else:
            positions = list(range(len(powerset[size])))
            index = np.random.choice(positions, max_sample, replace=False)
            sample[size] = [powerset[size][i] for i in index]
    return sample


def compute_powerset_size(powerset):
    flat_powerset = []
    for size in powerset:
        flat_powerset.extend(powerset[size])
    return len(flat_powerset)


if __name__ == "__main__":
    # Load SK config
    sk_config = SKConfig()
    ds_list = sk_config.get_sk_list()
    ds_list.remove("vqav2_holdout")

    for dataset in ds_list:
        print(f"Looking at {dataset}.")
        # Get current paths
        sk_paths = sk_config.get_sk_paths(dataset)
        ds_sk = {}
        combinations_details = {}
        base_dir = Path(__file__).parent.parent
        base_path_details = base_dir.joinpath(
            f"./data/datasets/{dataset}/imgs_occluded/"
        )
        make_dir(base_path_details)
        path_comb_details = base_path_details.joinpath("details.json")
        # Load questions
        questions_file = f"../data/datasets/{dataset}/q_crowd.json"
        questions = load_json(questions_file)

        for curr_q in questions:
            ds_class = curr_q["class"]
            # Load SG data if not already present
            if ds_class not in ds_sk:
                base_sk_dir = sk_paths[ds_class]["dir"]
                curr_sg_file = sk_paths[ds_class]["sg_crowd"]
                curr_sg = load_jsonl(f"../{base_sk_dir}/{curr_sg_file}")
                ds_sk[ds_class] = curr_sg
            # Retrieve corresponding SG data
            curr_sg = lookup_sg(curr_q["img"], ds_sk[ds_class])
            if not curr_sg:
                raise

            # Get list of concepts
            concept_dict = get_concept_dict(curr_sg["rel_clusters_unique"])
            # Compute powerset (+ store details)
            print(
                f"Question {curr_q['question_id']} -- img {curr_q['img']}: dealing with {len(concept_dict.keys())} concepts."
            )
            powerset_dict = compute_concepts_powerset(
                list(concept_dict.keys()), return_dict=True
            )
            total_size = compute_powerset_size(powerset_dict)
            print(f">> For a total of {total_size} combinations")

            # Get actual data and sample it
            powerset_dict = sample_powerset(powerset_dict, max_sample=200)
            total_size = compute_powerset_size(powerset_dict)
            print(f">> Sampled down to {total_size} combinations")
            combinations_details[curr_q["question_id"]] = powerset_dict

        # Save summary (images already saved)
        print("... Saving occlusion details...")
        save_json(combinations_details, path_comb_details)
        print("Saved.")
