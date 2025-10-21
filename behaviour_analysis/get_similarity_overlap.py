import networkx as nx
import numpy as np
import sys

from collections import OrderedDict
from pathlib import Path
from sentence_transformers import SentenceTransformer

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.rk_handler import RKHandler
from config_handlers.sk_handler import SKHandler
from utils import data_io, graph_utils


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


# Parse nx graphs
def parse_graph(g: nx.DiGraph) -> dict:
    g_dict = []
    for edge in g.edges:
        entry = {
            "from_concept": edge[0],
            "rel": g[edge[0]][edge[1]]["label"],
            "to_concept": edge[1],
        }
        entry["text"] = graph_utils.nx_triple_to_str(
            (entry["from_concept"], entry["rel"], entry["to_concept"]), template=False
        )
        g_dict.append(entry)
    return g_dict


# Cosine Similarity
def compute_cosine(
    emb_model: SentenceTransformer, sk_graph: nx.DiGraph, rk_graph: nx.DiGraph
) -> list[dict]:
    # Create sets
    sk_data = parse_graph(sk_graph)
    rk_data = parse_graph(rk_graph)
    all_sk_embs = np.array(
        [emb_model.encode(sk["text"], normalize_embeddings=True) for sk in sk_data]
    )
    all_rk_embs = np.array(
        [emb_model.encode(rk["text"], normalize_embeddings=True) for rk in rk_data]
    )
    # Guardrails for when there's nothing that is embedded.
    # Just return RK data with cosine = 0.0
    if all_sk_embs.size == 0 or all_rk_embs.size == 0:
        for rk in rk_data:
            rk["sk_similar"]["cosine"] = 0.0
        return rk_data

    # This is a ndarray of shape [len(all_rk_embs), len(all_sk_embs)]
    cos_vals = emb_model.similarity(all_rk_embs, all_sk_embs).numpy()
    # print(cos_vals.shape)

    sks_seen = set()
    cos_copy = cos_vals.copy()
    # With this we have, for each row, a list of indices pointing to the unsorted array
    # but in the position of what would be the sorted array.
    # [2, 0, 1] -> the 0th element of the sorted array is the 2nd of the unsorted one
    max_ids = np.argsort(cos_copy, axis=1, stable=True)
    # print(max_ids)
    # Do argsort over flat array with cosine values. Then, unravel the index to get
    # the corresponding coordinates in the ndarray `cos_copy`. Record only the first
    # occurrence of a `rk_idx` so that we get the order in which we need to process
    # the RKs.
    ordered_rk_ids = OrderedDict()
    for idx in np.flip(np.argsort(cos_copy.flatten(), stable=True)):
        rk_idx, sk_idx = np.unravel_index(idx, cos_copy.shape)
        if rk_idx not in ordered_rk_ids:
            ordered_rk_ids[rk_idx.item()] = sk_idx.item()
        else:
            continue
    # print(ordered_rk_ids)
    # print("=" * 40)

    print(f"Processing {cos_copy.shape[0]} RKs with {cos_copy.shape[1]} SKs")
    """
    - |RK|=|SK|: This is the good one, all of the RK should get a matching SK
    - |RK|<|SK|: Here, the model is definitely missing out on some specs
    - |RK|>|SK|: Here the model has more behaviours than it should. Note that these 
        "extra" behaviours are among the least similar ones and they might be wrong.
    """
    for rk_idx, sk_idx in ordered_rk_ids.items():
        # Handle the different amounts of RKs and SKs
        # Easy case: if len(RK) <= len(SK), no need to do anything
        # If len(RK) > len(SK), the last len(RK)-len(SK) RKs won't be matched
        if len(sks_seen) >= cos_copy.shape[1]:
            print(f"Unable to match RK {rk_idx}: exhausted the available SKs")
            rk_data[rk_idx]["sk_similar"] = {"cosine": 0.0}
            continue

        # print(f"Processing RK: {rk_idx}")
        curr_sk_idx = sk_idx
        # If we already saw sk_idx, scan the rk_idx-th row in reverse (from the max)
        i = -1
        while curr_sk_idx in sks_seen:
            print(f"SK idx {curr_sk_idx} already used")
            # print(max_ids[rk_idx])
            curr_sk_idx = max_ids[rk_idx][i].item()
            i -= 1
        print(
            f"Maximum {cos_copy[rk_idx][curr_sk_idx]:.2f} taken at ({rk_idx}, {curr_sk_idx})"
        )
        # Update rk data with cosine sim and corresponding SK info
        rk_data[rk_idx]["sk_similar"] = {"cosine": cos_vals[rk_idx][curr_sk_idx].item()}
        rk_data[rk_idx]["sk_similar"].update(sk_data[curr_sk_idx])
        # Mark the SKs that have been used up
        sks_seen.add(curr_sk_idx)
        print(f"Updated sks_seen: {sks_seen}")
        print(f"SKs left: {cos_vals.shape[1] - len(sks_seen)}")
        print("=" * 40)

    return rk_data


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
            sim_sample[model][ds][g_idx] = compute_cosine(
                emb_model, sk_data[g_idx], rk_data[g_idx]
            )
            # Append individual measures to summary for later
            cosine_vals = [
                rk["sk_similar"]["cosine"]
                for rk in sim_sample[model][ds][g_idx]
                if rk["sk_similar"]["cosine"] is not None
            ]
            if "cosine" not in sim_summary[model][ds]:
                sim_summary[model][ds]["cosine"] = cosine_vals
            else:
                sim_summary[model][ds]["cosine"].extend(cosine_vals)

        # Compute mean values for measures
        sim_summary[model][ds]["cosine_avg"] = np.mean(sim_summary[model][ds]["cosine"])
        sim_summary[model][ds]["cosine_std"] = np.std(sim_summary[model][ds]["cosine"])
        sim_summary[model][ds].pop("cosine", None)

        # Save
        out_path = Path("raw_stats", f"sim_overlap_{emb_model_suffix}", model, ds)
        data_io.make_dir(out_path)
        data_io.save_json(sim_sample[model][ds], out_path.joinpath("samples.json"))
        data_io.save_json(sim_summary[model][ds], out_path.joinpath("summary.json"))
