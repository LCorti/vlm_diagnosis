import networkx as nx
import numpy as np
import torch
import sys

from pathlib import Path
from sentence_transformers import SentenceTransformer

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.rk_handler import RKHandler
from config_handlers.sk_handler import SKHandler
from utils import data_io, graph_utils


# Graph Edit Distance
def node_match(node1: dict, node2: dict) -> bool:
    return node1["label"] == node2["label"]


def edge_match(edge1: dict, edge2: dict) -> bool:
    return edge1["label"] == edge2["label"]


def compute_ged(sk_graph: nx.DiGraph, rk_graph: nx.DiGraph) -> float:
    return nx.graph_edit_distance(
        sk_graph, rk_graph, node_match=node_match, edge_match=edge_match, timeout=10
    )


# IOU
def compute_iou(sk_graph: nx.DiGraph, rk_graph: nx.DiGraph) -> float:
    sk_nodes = set([sk_graph.nodes[n]["label"] for n in sk_graph.nodes])
    rk_nodes = set([rk_graph.nodes[n]["label"] for n in rk_graph.nodes])
    sk_edges = set([sk_graph[e[0]][e[1]]["label"] for e in sk_graph.edges])
    rk_edges = set([rk_graph[e[0]][e[1]]["label"] for e in rk_graph.edges])

    nodes_inter = len(sk_nodes & rk_nodes)
    nodes_union = len(sk_nodes | rk_nodes)
    edges_inter = len(sk_edges & rk_edges)
    edges_union = len(sk_edges | rk_edges)

    return {
        "iou_nodes": nodes_inter / nodes_union,
        "iou_edges": edges_inter / edges_union,
    }


# Cosine Similarity
def compute_cosine(
    emb_model: SentenceTransformer, sk_graph: nx.DiGraph, rk_graph: nx.DiGraph
) -> float:
    sk_text = graph_utils.nx_graph_to_str(sk_graph, template=False)
    rk_text = graph_utils.nx_graph_to_str(rk_graph, template=False)
    sk_batch = np.array(
        [emb_model.encode(sk, normalize_embeddings=True) for sk in sk_text]
    )
    rk_batch = np.array(
        [emb_model.encode(rk, normalize_embeddings=True) for rk in rk_text]
    )

    if len(sk_batch) == 0 or len(rk_batch) == 0:
        print("Problem!")
        return 0.0
    else:
        sk_emb = torch.FloatTensor(np.mean(sk_batch, axis=0)).unsqueeze(0)
        rk_emb = torch.FloatTensor(np.mean(rk_batch, axis=0)).unsqueeze(0)
    return emb_model.similarity_pairwise(sk_emb, rk_emb).tolist()[0]


# Data loading functions
def load_sk(sk_hdl: SKHandler) -> dict:
    sk_data = {}
    for ds_class in sk_hdl.get_classes():
        sk_hdl.set_curr_class(ds_class)
        graphs_path = sk_hdl.get_exp_pkl_path()
        sk_data.update(data_io.load_pickle(Path("..", graphs_path)))
    return sk_data


def load_rk(rk_hdl: RKHandler) -> dict:
    return data_io.load_pickle(Path("..", rk_hdl.get_final_pkl_path()))


if __name__ == "__main__":
    sk_hdl = SKHandler()
    rk_hdl = RKHandler()
    # emb_model_path = "ibm-granite/granite-embedding-english-r2"
    emb_model_path = "sentence-transformers/all-mpnet-base-v2"
    if emb_model_path == "sentence-transformers/all-mpnet-base-v2":
        emb_model_suffix = "mpnet"
    elif emb_model_path == "ibm-granite/granite-embedding-english-r2":
        emb_model_suffix = "granite"
    else:
        emb_model_suffix = ""
    emb_model = SentenceTransformer(emb_model_path)
    ds_list = sk_hdl.get_ds_list()
    model_list = rk_hdl.get_model_list()
    all_datasets = sk_hdl.get_ds_list() * len(model_list)
    all_models = rk_hdl.get_model_list() * len(ds_list)
    all_models.sort()

    # Just init the final dict
    sim_sample = {}
    sim_summary = {}

    for ds, model in zip(all_datasets, all_models):
        print(f"Working on {ds.upper()} x {model.upper()}")
        sk_hdl.set_curr_ds(ds)
        rk_hdl.set_curr_model(model)
        rk_hdl.set_curr_ds(ds)

        sk_data = load_sk(sk_hdl)
        rk_data = load_rk(rk_hdl)

        if model not in sim_sample:
            sim_sample[model] = {}
        if ds not in sim_sample[model]:
            sim_sample[model][ds] = {}

        if model not in sim_summary:
            sim_summary[model] = {}
        if ds not in sim_summary[model]:
            sim_summary[model][ds] = {}

        print(f"> {len(rk_data)} samples to process...")
        for g_idx in rk_data.keys():
            print(f"Processing {g_idx}")
            # Compute sample-level measures
            measures = {
                "ged": compute_ged(sk_data[g_idx], rk_data[g_idx]),
                "iou": compute_iou(sk_data[g_idx], rk_data[g_idx]),
                "cosine": compute_cosine(emb_model, sk_data[g_idx], rk_data[g_idx]),
            }
            sim_sample[model][ds][g_idx] = measures

            # Append individual measures to summary for later
            if "ged" not in sim_summary[model][ds]:
                sim_summary[model][ds]["ged"] = [measures["ged"]]
            else:
                sim_summary[model][ds]["ged"].append(measures["ged"])

            if "iou" not in sim_summary[model][ds]:
                sim_summary[model][ds]["iou"] = {
                    "iou_nodes": [measures["iou"]["iou_nodes"]],
                    "iou_edges": [measures["iou"]["iou_edges"]],
                }
            else:
                sim_summary[model][ds]["iou"]["iou_nodes"].append(
                    measures["iou"]["iou_nodes"]
                )
                sim_summary[model][ds]["iou"]["iou_edges"].append(
                    measures["iou"]["iou_edges"]
                )

            if "cosine" not in sim_summary[model][ds]:
                sim_summary[model][ds]["cosine"] = [measures["cosine"]]
            else:
                sim_summary[model][ds]["cosine"].append(measures["cosine"])

        # Compute mean values for measures
        sim_summary[model][ds]["ged_avg"] = np.mean(sim_summary[model][ds]["ged"])
        sim_summary[model][ds]["ged_std"] = np.std(sim_summary[model][ds]["ged"])
        sim_summary[model][ds].pop("ged", None)

        sim_summary[model][ds]["iou"]["iou_nodes_avg"] = np.mean(
            sim_summary[model][ds]["iou"]["iou_nodes"]
        )
        sim_summary[model][ds]["iou"]["iou_nodes_std"] = np.std(
            sim_summary[model][ds]["iou"]["iou_nodes"]
        )
        sim_summary[model][ds]["iou"].pop("iou_nodes", None)

        sim_summary[model][ds]["iou"]["iou_edges_avg"] = np.mean(
            sim_summary[model][ds]["iou"]["iou_edges"]
        )
        sim_summary[model][ds]["iou"]["iou_edges_std"] = np.std(
            sim_summary[model][ds]["iou"]["iou_edges"]
        )
        sim_summary[model][ds]["iou"].pop("iou_edges", None)

        summary_avg = np.mean(sim_summary[model][ds]["cosine"])
        sim_summary[model][ds]["cosine_avg"] = (
            summary_avg.item() if not np.isnan(summary_avg) else 0.0
        )
        summary_std = np.std(sim_summary[model][ds]["cosine"])
        sim_summary[model][ds]["cosine_std"] = (
            summary_std.item() if not np.isnan(summary_std) else 0.0
        )
        sim_summary[model][ds].pop("cosine", None)

        # Save
        out_path = Path("raw_stats", f"sim_{emb_model_suffix}", model, ds)
        data_io.make_dir(out_path)
        data_io.save_json(sim_sample[model][ds], out_path.joinpath("samples.json"))
        data_io.save_json(sim_summary[model][ds], out_path.joinpath("summary.json"))
