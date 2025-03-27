import cv2
import numpy as np
import os
import sys

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.sk_config_loader import SKConfig
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from utils.data_io import load_json, load_jsonl, make_dir, save_json
from utils.image_utils import load_img, save_img, from_array


def bbox_to_polygon(bb):
    poly = Polygon(
        [
            (bb["top_left_x"], bb["top_left_y"]),
            (bb["top_left_x"] + bb["width"], bb["top_left_y"]),
            (bb["bottom_right_x"], bb["bottom_right_y"]),
            (bb["top_left_x"], bb["top_left_y"] + bb["height"]),
        ]
    )
    return poly


def run_occlusion(img, concept_dict, concepts_to_occlude):
    concepts_to_skip = list(set(concept_dict.keys()) - set(concepts_to_occlude))
    # Make polygons out of bounding boxes
    list_to_occlude = [
        bbox_to_polygon(concept_dict[concept]) for concept in concepts_to_occlude
    ]
    list_to_skip = [
        bbox_to_polygon(concept_dict[concept]) for concept in concepts_to_skip
    ]
    # Compute patch to occlude
    area_to_occlude = unary_union(list_to_occlude)
    area_to_keep = unary_union(list_to_skip)
    patch_to_occlude = area_to_occlude.difference(area_to_keep)
    # Consider original polygon if difference returns an empty Polygon
    if patch_to_occlude.is_empty:
        patch_to_occlude = area_to_occlude

    # If a single polygon is returned, convert to multipolygon for processing
    if isinstance(patch_to_occlude, Polygon):
        patch_to_occlude = MultiPolygon([patch_to_occlude])

    masked_img = np.array(img)
    # Handle Multipolygon collection
    for poly in patch_to_occlude.geoms:
        # Shapely includes an extra point to close a Poly. Not needed for cv2
        pts = [np.array(poly.exterior.coords, dtype=np.int32)[:-1]]
        # Make masked image
        masked_img = cv2.fillPoly(masked_img, pts=pts, color=(0, 0, 0))

    # np array to PIL img
    pil_masked_img = from_array(masked_img)
    return pil_masked_img


def occlude_image(img, base_out_path, concept_dict, concept_combinations):
    occlusion_data = []
    counter = 0
    for combination in concept_combinations:
        # Prepare output object
        new_entry = {
            "path": f"{base_out_path}_{counter}.jpg",
            "occluded_concepts": combination,
        }
        # Run occlusion
        occluded_img = run_occlusion(img, concept_dict, combination)
        save_img(occluded_img, new_entry["path"])
        occlusion_data.append(new_entry)
        counter += 1
    return occlusion_data


def lookup_sg(img_id, sk_list):
    return next((sk_data for sk_data in sk_list if sk_data["img_id"] == img_id), None)


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


if __name__ == "__main__":
    # Load SK config
    sk_config = SKConfig()
    ds_list = sk_config.get_sk_list()
    ds_list.remove("vqav2_holdout")

    for dataset in ds_list:
        # Load images
        print(f"Looking at {dataset}.")
        # Get current paths
        sk_paths = sk_config.get_sk_paths(dataset)
        ds_sk = {}
        summary = {}
        powerset_details = {}
        base_dir = Path(__file__).parent.parent
        path_summary = base_dir.joinpath(
            f"./data/datasets/{dataset}/imgs_occluded/summary.json"
        )
        path_powerset_details = base_dir.joinpath(
            f"./data/datasets/{dataset}/imgs_occluded/details.json"
        )

        # Try to load combination data
        try:
            powerset_details = load_json(path_powerset_details)
        except Exception as _:
            print("Unable to find file with combination details.")
            raise

        # Load questions
        questions_file = f"../data/datasets/{dataset}/q_crowd.json"
        questions = load_json(questions_file)

        # Add summary entry for current dataset
        if dataset not in summary:
            summary[dataset] = {}

        for curr_q in questions:
            # Load image
            img_path = base_dir.joinpath(
                curr_q["img_path"].replace("imgs", "imgs_resized")
            )
            img = load_img(img_path)
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
            curr_powerset_detail = powerset_details[str(curr_q["question_id"])]

            # Run occlusion
            base_path_imgs = base_dir.joinpath(
                f"./data/datasets/{dataset}/imgs_occluded/{ds_class}/{curr_q['question_id']}"
            )

            make_dir(base_path_imgs)
            flat_list = [
                comb for subset in curr_powerset_detail.values() for comb in subset
            ]
            occluded_images = occlude_image(
                img,
                f"{base_path_imgs}/{curr_q['img']}",
                concept_dict,
                flat_list,
            )
            occluded_images = []

            # Update summary
            if ds_class not in summary[dataset]:
                summary[dataset][ds_class] = {}

            summary[dataset][ds_class][curr_q["question_id"]] = {
                "og_img_id": curr_q["img"],
                "occlusion": occluded_images,
            }

        # Save summary (images already saved)
        print("... Saving occlusion summary...")
        save_json(summary[dataset], path_summary)
        print("Saved.")
