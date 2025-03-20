import cv2
import numpy as np
import os
import sys

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.dataset_config_loader import DatasetConfig
from config_loaders.sk_config_loader import SKConfig
from shapely.geometry import Polygon
from shapely.measurement import distance
from utils.data_io import make_dir, load_jsonl, save_jsonl

FULL_DATA = False


def flatten_bboxes_data(bboxes):
    bboxes_list = [
        [
            int(bb["top_left_x"]),
            int(bb["top_left_y"]),
            int(bb["bottom_right_x"]),
            int(bb["bottom_right_y"]),
        ]
        for bb in bboxes
    ]
    return bboxes_list


def compute_clusters(bboxes, eps):
    # Run 3 times the clustering algorithm, adjusting the threshold
    # to get a good number of clusters.
    clusters = {
        "t1": cv2.groupRectangles(bboxes, 1, eps - 0.05)[0],
        "t2": cv2.groupRectangles(bboxes, 1, eps)[0],
        "t3": cv2.groupRectangles(bboxes, 1, eps + 0.05)[0],
    }
    return clusters


def format_cluster_bbox(bb):
    formatted_bb = {
        "top_left_x": int(bb[0]),
        "top_left_y": int(bb[1]),
        "bottom_right_x": int(bb[2]),
        "bottom_right_y": int(bb[3]),
        "width": int(abs(bb[2] - bb[0])),
        "height": int(abs(bb[3] - bb[1])),
    }
    return formatted_bb


def bbox_to_polygon(bb):
    poly = Polygon(
        [
            (bb["top_left_x"], bb["top_left_y"]),
            (bb["top_left_x"] + bb["width"], bb["top_left_y"]),
            (bb["top_left_x"], bb["top_left_y"] + bb["height"]),
            (bb["bottom_right_x"], bb["bottom_right_y"]),
        ]
    )
    return poly


def overlap(bb1, bb2):
    rect1 = bbox_to_polygon(bb1)
    rect2 = bbox_to_polygon(bb2)
    return rect1.intersects(rect2)


def compute_distance(bb1, bb2):
    rect1 = bbox_to_polygon(bb1)
    rect2 = bbox_to_polygon(bb2)
    return distance(rect1, rect2)


def lookup_bbox(img_data, bb_rel, key):
    for bb in img_data["bboxes_rel"]:
        if (
            bb_rel[key]["bb_label_idx"] == bb["bb_label"]["bb_label_idx"]
            and bb_rel[key]["bb_label_text"] == bb["bb_label"]["bb_label_text"]
            and bb_rel[key]["bb_label_full"] == bb["bb_label"]["bb_label_full"]
        ):
            return bb


def resolve_concept(img_data, bb_rel, bb_c, key):
    if (
        bb_rel[key]["bb_label_idx"] == bb_c["bb_label"]["bb_label_idx"]
        and bb_rel[key]["bb_label_text"] == bb_c["bb_label"]["bb_label_text"]
        and bb_rel[key]["bb_label_full"] == bb_c["bb_label"]["bb_label_full"]
    ):
        return bb_c
    else:
        return lookup_bbox(img_data, bb_rel, key)


def remove_duplicate_rels(relations):
    no_dupes = {}
    for rel in relations:
        from_concept_idx = rel["from_concept"]["bb_label"]["bb_label_idx"]
        rel_label_idx = rel["rel_label"]["rel_label_idx"]
        to_concept_idx = rel["to_concept"]["bb_label"]["bb_label_idx"]
        dict_key = f"{from_concept_idx}_{rel_label_idx}_{to_concept_idx}"

        if dict_key not in no_dupes:
            no_dupes[dict_key] = rel
    return list(no_dupes.values())


def retrieve_img(clean_sg, img_idx):
    return next(img for img in clean_sg if img["img_id"] == img_idx)


if __name__ == "__main__":
    # Load dataset config
    ds_config = DatasetConfig()
    ds_list = ds_config.get_ds_list()

    # Load SK config
    sk_config = SKConfig()

    # Threshold for clustering bboxes
    eps = 0.32

    for dataset in ds_list:
        # Get current paths
        ds_paths = ds_config.get_ds_paths(dataset)
        sk_paths = sk_config.get_sk_paths(dataset)

        for ds_class in ds_paths:
            # Load clean scene graphs
            base_dir = sk_paths[ds_class]["dir"]
            clean_sg_file = sk_paths[ds_class]["sg_clean"]
            clean_sg = load_jsonl(f"../{base_dir}/{clean_sg_file}")

            bb_clusters = {}

            # Compute (unlabelled) clusters for bounding boxes
            for img in clean_sg:
                if img["img_id"] not in bb_clusters:
                    bb_clusters[img["img_id"]] = []

                bboxes_list = flatten_bboxes_data(img["bboxes_rel"])

                curr_clusters = compute_clusters(bboxes_list, eps)
                len_clusters = [len(curr_clusters[bbs]) for bbs in curr_clusters]
                max_clusters = max(len_clusters)
                cluster_bboxes = list(curr_clusters.values())[
                    len_clusters.index(max_clusters)
                ]

                for c_bb in cluster_bboxes:
                    bb_clusters[img["img_id"]].append(format_cluster_bbox(c_bb))

            # Resolve cluster labels
            for img_idx in bb_clusters:
                curr_img = retrieve_img(clean_sg, img_idx)
                curr_img["bboxes_clusters"] = []

                for bb_c in bb_clusters[img_idx]:
                    bb_overlaps = []
                    bboxes = curr_img["bboxes_rel"]

                    # Get all overlapping bboxes
                    for bb in bboxes:
                        if overlap(bb_c, bb):
                            bb_overlaps.append(bb)
                    curr_img["bboxes_clusters"].append(bb_c)

                    if len(bb_overlaps) != 0:
                        # get labels
                        all_labels = [bb["bb_label"] for bb in bb_overlaps]
                        labels_dict = {}
                        for label in all_labels:
                            if label["bb_label_text"] not in labels_dict:
                                labels_dict[label["bb_label_text"]] = 0
                            labels_dict[label["bb_label_text"]] += 1

                        # Get info of closest bbox
                        min_dist = np.inf
                        for bb in bb_overlaps:
                            curr_d = compute_distance(bb_c, bb)
                            if curr_d < min_dist:
                                bb_c["bb_label"] = bb["bb_label"]

                        # Update relationships with cluster info
                        curr_img["rel_clusters"] = []
                        for bb_rel in curr_img["relations"]:
                            new_rel = {}
                            new_rel["from_concept"] = resolve_concept(
                                curr_img, bb_rel, bb_c, "from_concept"
                            )
                            new_rel["to_concept"] = resolve_concept(
                                curr_img, bb_rel, bb_c, "to_concept"
                            )
                            new_rel["rel_label"] = bb_rel["rel_label"]
                            curr_img["rel_clusters"].append(new_rel)
                    else:
                        curr_img["rel_clusters"] = []

                    # Fix duplicate relationships
                    curr_img["rel_clusters_unique"] = remove_duplicate_rels(
                        curr_img["rel_clusters"]
                    )

            # Save class data to file
            out_folder = f"../{base_dir}/scene_graphs"
            make_dir(out_folder)

            if not FULL_DATA:
                for img in clean_sg:
                    del img["relations"]
                    del img["bboxes_not_rel"]
                    del img["rel_clusters"]

            save_jsonl(clean_sg, f"../{base_dir}/{sk_paths[ds_class]['sg_merged']}")
