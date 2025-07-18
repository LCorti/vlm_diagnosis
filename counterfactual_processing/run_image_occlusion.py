import cv2
import numpy as np
import sys

from pathlib import Path
from PIL import Image

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.sk_handler import SKHandler
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from utils.data_io import load_json, load_jsonl, make_dir, save_json
from utils.image_utils import load_img, save_img, from_array


def bbox_to_polygon(bb: dict) -> Polygon:
    poly = Polygon(
        [
            (bb["top_left_x"], bb["top_left_y"]),
            (bb["top_left_x"] + bb["width"], bb["top_left_y"]),
            (bb["bottom_right_x"], bb["bottom_right_y"]),
            (bb["top_left_x"], bb["top_left_y"] + bb["height"]),
        ]
    )
    return poly


def run_occlusion(img: Image, concept_dict: dict, concepts_to_occlude: list) -> Image:
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


def occlude_image(
    img: Image,
    base_out_path: str | Path,
    concept_dict: dict,
    concept_combinations: list[dict],
) -> dict:
    occlusion_data = []
    counter = 0
    for combination in concept_combinations:
        # Prepare output object
        full_path = f"{base_out_path}_{counter}.jpg"
        new_entry = {
            "path": f".{full_path.split('datasets')[1]}",
            "occluded_concepts": combination,
        }
        # Run occlusion
        occluded_img = run_occlusion(img, concept_dict, combination)
        save_img(occluded_img, full_path)
        occlusion_data.append(new_entry)
        counter += 1
    return occlusion_data


def lookup_sg(img_id: int, sk_list: list[dict]) -> dict:
    return next((sk_data for sk_data in sk_list if sk_data["img_id"] == img_id), None)


def get_concept_dict(relations: list[dict]) -> dict:
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
    sk_hdl = SKHandler()
    ds_list = sk_hdl.get_ds_list()
    if "vqav2_holdout" in ds_list:
        ds_list.remove("vqav2_holdout")

    for dataset in ds_list:
        # Load images
        print(f"Looking at {dataset}.")
        # Get current paths
        sk_hdl.set_curr_ds(dataset)
        ds_sk = {}
        summary = {}
        powerset_details = {}
        base_dir = Path(__file__).parent.parent
        path_summary = base_dir.joinpath(
            Path(".", "data", "datasets", dataset, "imgs_occluded", "summary.json")
        )
        path_powerset_details = base_dir.joinpath(
            Path(".", "data", "datasets", dataset, "imgs_occluded", "details.json")
        )

        # Try to load combination data
        try:
            powerset_details = load_json(path_powerset_details)
        except Exception as _:
            print("Unable to find file with combination details.")
            raise

        # Load questions
        questions_file = Path("..", "data", "datasets", dataset, "q_crowd.json")
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
            sk_hdl.set_curr_class(ds_class)
            # Load SG data if not already present
            if ds_class not in ds_sk:
                curr_sg = load_jsonl(Path("..", sk_hdl.get_sg_crowd_path()))
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
                Path(
                    ".",
                    "data",
                    "datasets",
                    dataset,
                    "imgs_occluded",
                    ds_class,
                    curr_q["question_id"],
                )
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
