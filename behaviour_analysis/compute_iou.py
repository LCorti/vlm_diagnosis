import json
import os
import numpy as np
import utils.data_io as data_io

from pathlib import Path

MODELS = ["InternVL2", "LLaVa-1.6", "MiniGPT-4", "ShareGPT4V"]
DATASETS = ["llava", "mmbench", "seed", "vqav2"]

ALL_DATASETS = DATASETS * 4
ALL_MODELS = MODELS * 4
ALL_MODELS.sort()


# Helper functions for node matching
def node_match(node1, node2):
    return node1["label"] == node2["label"]


def edge_match(edge1, edge2):
    return edge1["label"] == edge2["label"]


# Helper functions for loading and saving data
def load_sk_data(sk_data_paths):
    sk_data = {ds: {} for ds in DATASETS}
    for sk_dp, ds in zip(sk_data_paths, DATASETS):
        graph_files = os.listdir(sk_dp)

        for gf in graph_files:
            g_idx = int(Path(gf).stem.split("_")[1]) - 1
            sk_data[ds][g_idx] = data_io.load_pickle(f"{sk_dp}/{gf}")

    return sk_data


def load_rk_data(rk_data_paths):
    rk_data = {m: {ds: {} for ds in DATASETS} for m in MODELS}
    for rk_dp, ds, model in zip(rk_data_paths, ALL_DATASETS, ALL_MODELS):
        graph_files = os.listdir(rk_dp)

        for gf in graph_files:
            g_idx = int(Path(gf).stem.split("_")[1])
            rk_data[model][ds][g_idx] = data_io.load_pickle(f"{rk_dp}/{gf}")

    return rk_data


def save_iou_data(curr_iou_data, out_path):
    with open(out_path, "w") as fp:
        json.dump(curr_iou_data, fp)


if __name__ == "__main__":
    # Create data paths
    sk_data_paths = [f"./sk/{ds}/graphs/ann" for ds in DATASETS]
    rk_data_paths = [f"./rk/{m}/{ds}/graphs" for m in MODELS for ds in DATASETS]

    # Load SK and RK data
    sk_data = load_sk_data(sk_data_paths)
    rk_data = load_rk_data(rk_data_paths)
    iou_data = {m: {ds: {} for ds in DATASETS} for m in MODELS}

    # Go!
    for ds, model in zip(ALL_DATASETS, ALL_MODELS):
        print(f">> Computing iou for {model} on {ds}...")
        print(f"... Need to compute IOU for {len(rk_data[model][ds])} SK-RK pairs.")

        # Iterating over the REALLY-KNOWS (skipping SKs for which we do not have a RK)
        for g_idx in rk_data[model][ds]:
            curr_sk = sk_data[ds][g_idx]
            curr_rk = rk_data[model][ds][g_idx]

            # compute intersection and union of node labels
            curr_sk_nodes_set = set([curr_sk.nodes[n]["label"] for n in curr_sk.nodes])
            curr_rk_nodes_set = set([curr_rk.nodes[n]["label"] for n in curr_rk.nodes])
            curr_nodes_intersection = len(curr_sk_nodes_set & curr_rk_nodes_set)
            curr_nodes_union = len(curr_sk_nodes_set | curr_rk_nodes_set)

            # compute intersection and union of edge labels
            curr_sk_edges_set = set(
                [curr_sk[e[0]][e[1]]["label"] for e in curr_sk.edges]
            )
            curr_rk_edges_set = set(
                [curr_rk[e[0]][e[1]]["label"] for e in curr_rk.edges]
            )
            curr_edges_intersection = len(curr_sk_edges_set & curr_rk_edges_set)
            curr_edges_union = len(curr_sk_edges_set | curr_rk_edges_set)

            iou_data[model][ds][g_idx] = {
                "sk_nodes": curr_sk.number_of_nodes(),
                "sk_edges": curr_sk.number_of_edges(),
                "rk_nodes": curr_rk.number_of_nodes(),
                "rk_edges": curr_rk.number_of_nodes(),
                "iou_nodes": curr_nodes_intersection / curr_nodes_union,
                "iou_edges": curr_edges_intersection / curr_edges_union,
            }

        # Compute avg and std IOU
        summary_iou = {
            "avg_iou_nodes": np.mean(
                [e["iou_nodes"] for e in iou_data[model][ds].values()]
            ),
            "std_iou_nodes": np.std(
                [e["iou_nodes"] for e in iou_data[model][ds].values()]
            ),
            "avg_iou_edges": np.mean(
                [e["iou_edges"] for e in iou_data[model][ds].values()]
            ),
            "std_iou_edges": np.std(
                [e["iou_edges"] for e in iou_data[model][ds].values()]
            ),
        }

        # Save IOU data
        print(f"Saving IOU values for {model}...")
        out_path = f"./rk/{model}/{ds}"
        save_iou_data(iou_data[model][ds], f"{out_path}/iou_samples.json")
        save_iou_data(summary_iou, f"{out_path}/iou_summary.json")
        print("Saved!")
