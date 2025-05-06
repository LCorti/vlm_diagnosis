import json
import os
import numpy as np
import torch
import utils.data_io as data_io
import utils.ops as ops

from pathlib import Path
from sentence_transformers import SentenceTransformer

MODELS = ["InternVL2", "LLaVa-1.6", "MiniGPT-4", "ShareGPT4V"]
DATASETS = ["llava", "mmbench", "seed", "vqav2"]

ALL_DATASETS = DATASETS * 4
ALL_MODELS = MODELS * 4
ALL_MODELS.sort()


# Helper functions for loading and saving data
def load_sk_data(sk_data_paths):
    sk_data = {ds: {} for ds in DATASETS}
    for sk_dp, ds in zip(sk_data_paths, DATASETS):
        graph_files = os.listdir(sk_dp)

        for gf in graph_files:
            g_idx = int(Path(gf).stem.split("_")[1]) - 1
            sk_data[ds][g_idx] = data_io.load_pickle(Path(sk_dp, gf))

    return sk_data


def load_rk_data(rk_data_paths):
    rk_data = {m: {ds: {} for ds in DATASETS} for m in MODELS}
    for rk_dp, ds, model in zip(rk_data_paths, ALL_DATASETS, ALL_MODELS):
        graph_files = os.listdir(rk_dp)

        for gf in graph_files:
            g_idx = int(Path(gf).stem.split("_")[1])
            rk_data[model][ds][g_idx] = data_io.load_pickle(f"{rk_dp}/{gf}")

    return rk_data


def save_sim_data(curr_sim_data, out_path):
    with open(out_path, "w") as fp:
        json.dump(curr_sim_data, fp)


if __name__ == "__main__":
    # Load sentence-transformers for computing embeddings
    emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Create data paths
    sk_data_paths = [Path(".", "sk", ds, "graphs", "ann") for ds in DATASETS]
    rk_data_paths = [
        Path(".", "rk", m, ds, "graphs") for m in MODELS for ds in DATASETS
    ]
    # Load SK and RK data
    sk_data = load_sk_data(sk_data_paths)
    rk_data = load_rk_data(rk_data_paths)
    sim_data = {m: {ds: {} for ds in DATASETS} for m in MODELS}

    # Go!
    for ds, model in zip(ALL_DATASETS, ALL_MODELS):
        print(f">> Computing similarity for {model} on {ds}...")
        print(
            f"... Need to compute similarity for {len(rk_data[model][ds])} SK-RK pairs."
        )

        # Iterating over the REALLY-KNOWS (skipping SKs for which we do not have a RK)
        for g_idx in rk_data[model][ds]:
            curr_sk = sk_data[ds][g_idx]
            curr_rk = rk_data[model][ds][g_idx]

            # Embed individual SKs and RKs
            curr_sk_text = ops.graph_to_text(curr_sk, template=False)
            curr_rk_text = ops.graph_to_text(curr_rk, template=False)

            curr_sk_batch = np.array(
                [emb_model.encode(sk, normalize_embeddings=True) for sk in curr_sk_text]
            )
            curr_rk_batch = np.array(
                [emb_model.encode(rk, normalize_embeddings=True) for rk in curr_rk_text]
            )

            # Embed single triplet and average to get "graph embedding"
            curr_sk_emb = torch.FloatTensor(np.mean(curr_sk_batch, axis=0)).unsqueeze(0)
            curr_rk_emb = torch.FloatTensor(np.mean(curr_rk_batch, axis=0)).unsqueeze(0)

            sim_data[model][ds][g_idx] = emb_model.similarity_pairwise(
                curr_sk_emb, curr_rk_emb
            ).tolist()[0]

        summary_sim = {
            "avg_sim": np.mean(list(sim_data[model][ds].values())),
            "std_sim": np.std(list(sim_data[model][ds].values())),
        }

        # Save similarity data
        print(f"Saving similarity values for {model}...")
        out_path = Path(".", "rk", model, ds)
        save_sim_data(sim_data[model][ds], out_path.joinpath("sim_samples.json"))
        save_sim_data(summary_sim, out_path.joinpath("sim_summary.json"))
        print("Saved!")
