import networkx as nx
import numpy as np
import sys

from pathlib import Path
from sentence_transformers import SentenceTransformer

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.rk_handler import RKHandler
from config_handlers.sk_handler import SKHandler
from utils import data_io, graph_utils


# Cosine Similarity
def compute_cosine(
    emb_model: SentenceTransformer, sk_graph: nx.DiGraph, rk_graph: nx.DiGraph
) -> list[dict]:
    # Convert graphs to a list of strings
    sk_templated = graph_utils.nx_graph_to_str(sk_graph, template=True)
    rk_templated = graph_utils.nx_graph_to_str(rk_graph, template=True)
    sk_strs = graph_utils.nx_graph_to_str(sk_graph, template=False)
    rk_strs = graph_utils.nx_graph_to_str(rk_graph, template=False)

    # Cache SK embeddings
    sk_embs = np.array(
        [emb_model.encode(sk, normalize_embeddings=True) for sk in sk_strs]
    )

    # Do cosine similarity
    res = []
    for rk_idx, rk_data in enumerate(rk_strs):
        # Embed current RK
        rk_emb = np.array(emb_model.encode(rk_data, normalize_embeddings=True))
        if len(sk_embs) == 0 or len(rk_emb) == 0:
            print("Problem!")
            return 0.0

        # Compute similarity values
        cosine_vals = emb_model.similarity(rk_emb, sk_embs).numpy()[0]
        # Save max cosine value
        max_cosine_idx = np.argmax(cosine_vals)
        res.append(
            {
                "rk_triple": graph_utils.triple_str_to_dict(rk_templated[rk_idx]),
                "sk_triple": graph_utils.triple_str_to_dict(
                    sk_templated[max_cosine_idx]
                ),
                "cosine": cosine_vals[max_cosine_idx].item(),
            }
        )

    return res


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
            # Compute triple-level measures
            measures = compute_cosine(emb_model, sk_data[g_idx], rk_data[g_idx])
            # Format data and aggregate for each sample as mean-of-max
            sim_sample[model][ds][g_idx] = {
                "measures": measures,
                "avg": np.mean([m["cosine"] for m in measures]).item(),
            }

            # Append individual measures to summary for later
            if "cosine" not in sim_summary[model][ds]:
                sim_summary[model][ds]["cosine"] = [sim_sample[model][ds][g_idx]["avg"]]
            else:
                sim_summary[model][ds]["cosine"].append(
                    sim_sample[model][ds][g_idx]["avg"]
                )

        # Compute mean values for measures
        sim_summary[model][ds]["cosine_avg"] = np.mean(sim_summary[model][ds]["cosine"])
        sim_summary[model][ds]["cosine_std"] = np.std(sim_summary[model][ds]["cosine"])
        sim_summary[model][ds].pop("cosine", None)

        # Save
        out_path = Path("raw_stats", f"sim_triples_{emb_model_suffix}", model, ds)
        data_io.make_dir(out_path)
        data_io.save_json(sim_sample[model][ds], out_path.joinpath("samples.json"))
        data_io.save_json(sim_summary[model][ds], out_path.joinpath("summary.json"))
