import json
import networkx as nx
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
            sk_data[ds][g_idx] = data_io.load_graph_pickle(f"{sk_dp}/{gf}")

    return sk_data


def load_rk_data(rk_data_paths):
    rk_data = {m: {ds: {} for ds in DATASETS} for m in MODELS}
    for rk_dp, ds, model in zip(rk_data_paths, ALL_DATASETS, ALL_MODELS):
        graph_files = os.listdir(rk_dp)

        for gf in graph_files:
            g_idx = int(Path(gf).stem.split("_")[1])
            rk_data[model][ds][g_idx] = data_io.load_graph_pickle(f"{rk_dp}/{gf}")

    return rk_data


def save_ged_data(curr_ged_data, out_path):
    with open(out_path, "w") as fp:
        json.dump(curr_ged_data, fp)


if __name__ == "__main__":
    # Create data paths
    sk_data_paths = [f"./sk/{ds}/graphs/ann" for ds in DATASETS]
    rk_data_paths = [f"./rk/{m}/{ds}/graphs" for m in MODELS for ds in DATASETS]

    # Load SK and RK data
    sk_data = load_sk_data(sk_data_paths)
    rk_data = load_rk_data(rk_data_paths)
    ged_data = {m: {ds: {} for ds in DATASETS} for m in MODELS}

    # Go!
    for ds, model in zip(ALL_DATASETS, ALL_MODELS):
        print(f">> Computing GED for {model} on {ds}...")
        print(f"... Need to compute GED for {len(rk_data[model][ds])} SK-RK pairs.")

        # Iterating over the REALLY-KNOWS (skipping SKs for which we do not have a RK)
        for g_idx in rk_data[model][ds]:
            curr_rk = rk_data[model][ds][g_idx]
            curr_sk = sk_data[ds][g_idx]
            ged_data[model][ds][g_idx] = {
                "sk_nodes": curr_sk.number_of_nodes(),
                "sk_edges": curr_sk.number_of_edges(),
                "rk_nodes": curr_rk.number_of_nodes(),
                "rk_edges": curr_rk.number_of_nodes(),
                "ged": nx.graph_edit_distance(
                    curr_sk,
                    curr_rk,
                    node_match=node_match,
                    edge_match=edge_match,
                    timeout=10,
                ),
            }

        # Compute avg and std GED
        summary_ged = {
            "avg_ged": np.mean([e["ged"] for e in ged_data[model][ds].values()]),
            "std_ged": np.std([e["ged"] for e in ged_data[model][ds].values()]),
        }

        # Save GED data
        print(f"Saving GED values for {model}...")
        out_path = f"./rk/{model}/{ds}"
        save_ged_data(ged_data[model][ds], f"{out_path}/ged_samples.json")
        save_ged_data(summary_ged, f"{out_path}/ged_summary.json")
        print(f"Saved!")
