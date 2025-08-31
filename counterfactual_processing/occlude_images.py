import cv2
import numpy as np
import sys

from pathlib import Path
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

import utils.data_io as data_io
import utils.graph_utils as graph_utils
import utils.image_utils as image_utils
from config_handlers.causal_handler import CausalHandler
from config_handlers.dataset_handler import DatasetHandler
from config_handlers.rk_handler import RKHandler


# Computing the powerset over the RK sets extracted for each data sample
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


def format_powerset(powerset: dict[list]) -> dict:
    details = {}
    for size, data in powerset.items():
        details[size] = {idx: {"combination": comb} for idx, comb in enumerate(data)}
    return details


def sample_powerset(powerset: dict, max_sample: int = 100) -> list[dict]:
    sample = {}
    for size in powerset:
        if len(powerset[size]) <= max_sample:
            sample[size] = powerset[size]
        else:
            positions = list(range(len(powerset[size])))
            index = np.random.choice(positions, max_sample, replace=False)
            sample[size] = {
                new_idx: powerset[size][old_idx]
                for new_idx, old_idx in enumerate(index)
            }
    return sample


def compute_powerset_size(powerset: dict) -> int:
    flat_powerset = []
    for size in powerset:
        flat_powerset.extend(powerset[size])
    return len(flat_powerset)


# Occluding images
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
    pil_masked_img = image_utils.from_array(masked_img)
    return pil_masked_img


def occlude_image(
    img: Image,
    base_out_path: str | Path,
    concepts: dict,
    powerset: dict[dict[list]],
) -> dict:
    img_idx_count = 0
    for size in powerset:
        for data in powerset[size].values():
            out_path = base_out_path.joinpath(f"{img_idx_count}.jpg")
            # access through combination["combination"]
            occ_img = run_occlusion(img, concepts, data["combination"])
            image_utils.save_img(occ_img, out_path)
            data["path"] = str(out_path.relative_to(Path(*out_path.parts[:3])))
            img_idx_count += 1
    return powerset


if __name__ == "__main__":
    # Load handlers
    rk_hdl = RKHandler()
    ds_hdl = DatasetHandler()
    causal_hdl = CausalHandler()
    model_list = rk_hdl.get_model_list()
    ds_list = ds_hdl.get_ds_list()
    all_models = model_list * 4
    all_models.sort()
    all_ds = ds_list * 4

    base_dir = Path("..")

    for model, ds in zip(all_models, all_ds):
        print(f"Running image occlusion for {model.upper()} -- {ds.upper()}")
        # Load final rk data
        rk_hdl.set_curr_model(model)
        rk_hdl.set_curr_ds(ds)
        rk_final = data_io.load_jsonl(base_dir.joinpath(rk_hdl.get_rk_final_path()))

        # Load image paths
        ds_hdl.set_curr_ds(ds)
        q_data = data_io.load_json(base_dir.joinpath(ds_hdl.get_crowd_questions_path()))
        img_paths = {
            q["question_id"]: q["img_path"].replace("/imgs/", "/imgs_resized/")
            for q in q_data
        }

        # Make the output dir for the occluded images
        causal_hdl.set_curr_model(model)
        causal_hdl.set_curr_ds(ds)

        # Only for running the code on remote VPS w/ storage attached
        storage_path = Path(Path(__file__).resolve().root, "data", "storage")
        imgs_occ_path = causal_hdl.get_imgs_occluded_path()
        if storage_path.exists():
            imgs_occ_path = storage_path.joinpath(Path(*imgs_occ_path.parts[1:]))
            print(f"External storage found. Saving to '{imgs_occ_path}'")
        else:
            # If not present, save within project
            imgs_occ_path = base_dir.joinpath(imgs_occ_path)
            print(f"External NOT storage found. Saving to {imgs_occ_path}")
        data_io.make_dir(imgs_occ_path)

        for rk in rk_final:
            det_dict = {
                "question_id": rk["question_id"],
                "concepts": get_concept_dict(rk["triple_objs"]),
            }
            print(f"Processing sample ID: {rk['question_id']}")

            # Compute combinations
            powerset_dict = graph_utils.compute_concepts_powerset(
                list(det_dict["concepts"].keys()), return_dict=True
            )
            # Get the actual bbox data for the combinations
            det_dict["powerset"] = format_powerset(powerset_dict)
            print(f"> Got {compute_powerset_size(det_dict['powerset'])} combos.")

            # Sample powerset (if needed)
            det_dict["powerset"] = sample_powerset(det_dict["powerset"], max_sample=100)
            print(f"> Sampled {compute_powerset_size(det_dict['powerset'])} combos.")

            # Save powerset data to file before proceeding with image occlusion
            data_io.append_to_jsonl([det_dict], imgs_occ_path.joinpath("details.jsonl"))

            # Load image and do occlusion
            img_path = base_dir.joinpath(img_paths[rk["question_id"]])
            img = image_utils.load_img(img_path)
            img_idx = img_path.stem
            curr_out_path = imgs_occ_path.joinpath(
                f"q_{det_dict['question_id']}_i_{img_idx}"
            )
            data_io.make_dir(curr_out_path)
            det_dict["powerset"] = occlude_image(
                img, curr_out_path, det_dict["concepts"], det_dict["powerset"]
            )
            data_io.append_to_jsonl(
                [det_dict], imgs_occ_path.joinpath("det_paths.jsonl")
            )
