import warnings

warnings.simplefilter(action="ignore")

import json
import numpy as np
import pandas as pd
import sys

from pathlib import Path
from typing import Tuple

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

import utils.data_io as data_io

from config_handlers.eval_handler import EvalHandler


def load_triples_data(emb_model="mpnet") -> Tuple[dict, dict]:
    samples = {m: {ds: {} for ds in ds_list} for m in model_list}
    summaries = {m: {ds: {} for ds in ds_list} for m in model_list}
    for model, ds in zip(all_models, all_ds):
        base_fp = Path("raw_stats", f"sim_overlap_{emb_model}")
        samples[model][ds] = data_io.load_json(
            base_fp.joinpath(model, ds, "samples.json")
        )
        summaries[model][ds] = data_io.load_json(
            base_fp.joinpath(model, ds, "summary.json")
        )

    return samples, summaries


def load_eval_responses():
    data = {m: {ds: {}} for m in model_list for ds in ds_list}
    for model, ds in zip(all_models, all_ds):
        eval_hdl.set_curr_model(model)
        eval_hdl.set_curr_ds(ds)
        data[model][ds] = data_io.load_json(Path("..", eval_hdl.get_eval_path()))
    return data


def get_behaviour_type(sim_val, ranges):
    temp = float(round(sim_val, prec))
    for idx, r in ranges.items():
        if temp >= r["start"] and temp <= r["end"]:
            return idx
    # Dirty fix for cosine values that go beyond 1.0 (for some reason)
    if temp >= 1.0:
        return 1
    print("Error; returning None")
    return None


def get_sample_type(sample_rows, ranges):
    """
    First implementation: check sizes and do average of cosine similarities
    """
    sample_cosine = sample_rows["sample_cosine"].iloc[0]
    # If the average similarity is within the lowest range, simply return
    if sample_cosine >= ranges[3]["start"] and sample_cosine <= ranges[3]["end"]:
        return (3, False)
    # Otherwise, have a look at the number of SKs and RKs that we have
    count_rk = len(sample_rows) - sample_rows["rk_triple"].isnull().sum()
    count_sk = len(sample_rows) - sample_rows["sk_triple"].isnull().sum()

    if count_rk <= count_sk:
        # Here, we might have type 1 behaviours
        if sample_cosine >= ranges[1]["start"] and sample_cosine <= ranges[1]["end"]:
            return (1, False)
        else:
            # Here, the similarity can't be from type 3 -- has to be type 2.
            # Let's ignore size for now and classify as type 2.
            return (2, True)
    else:
        # Here, we might have type 2 behaviours
        if sample_cosine >= ranges[2]["start"] and sample_cosine <= ranges[2]["end"]:
            return (2, False)
        else:
            # Here, the similarity can't be from type 3 -- has to be type 1.
            # Let's ignore size for now and classify as type 1.
            return (1, True)


if __name__ == "__main__":
    # Setup
    eval_hdl = EvalHandler()
    model_list = eval_hdl.get_model_list()
    """ Skipping MiniGPT4 as we have incomplete data. """
    model_list.remove("minigpt4")
    eval_hdl.set_curr_model(model_list[0])
    ds_list = eval_hdl.get_ds_list()

    all_ds = ds_list * len(model_list)
    all_models = model_list * len(ds_list)
    all_models.sort()

    OPEN_DATASETS = ["llava-bench", "mmbench"]
    MC_DATASETS = ["seed", "vqav2"]

    # Load eval data
    eval_data = load_eval_responses()
    emb_model = "mpnet"  # granite
    samples, summaries = load_triples_data(emb_model=emb_model)
    # Organise cosine similarity values
    sample_cos = {model: {ds: {} for ds in ds_list} for model in model_list}
    sample_avgs = {model: {ds: {} for ds in ds_list} for model in model_list}
    for model, ds in zip(all_models, all_ds):
        for s, s_data in samples[model][ds].items():
            sample_cos[model][ds][s] = [e["sk_similar"]["cosine"] for e in s_data]
            sample_avgs[model][ds][s] = np.mean(sample_cos[model][ds][s]).item()

    # Prepare dataframe
    df = pd.DataFrame()
    for model, ds in zip(all_models, all_ds):
        for q_idx in samples[model][ds]:
            # We have very few cosine similarities that are sligthly negative.
            # Clip those to 0.0. They are basically irrelevant.
            avg_cosine = (
                sample_avgs[model][ds][q_idx]
                if sample_avgs[model][ds][q_idx] > 0.0
                else 0.0
            )
            rows = []
            """
            Take care of empty measures
                - Qwen2.5-VL + VQAv2: '120162001', '333538005'
                - ShareGPT4V + SEED: '2358'
            """
            if len(samples[model][ds][q_idx]) > 0:
                for m in samples[model][ds][q_idx]:
                    if m["sk_similar"].keys() & {"from_concept", "rel", "to_concept"}:
                        sk_triple = {
                            "from_concept": m["sk_similar"]["from_concept"],
                            "relationship": m["sk_similar"]["rel"],
                            "to_concept": m["sk_similar"]["to_concept"],
                        }
                    else:
                        sk_triple = None

                    rows.append(
                        {
                            "question_id": str(q_idx),
                            "model": model,
                            "dataset": ds,
                            "sk_triple": sk_triple,
                            "rk_triple": {
                                "from_concept": m["from_concept"],
                                "relationship": m["rel"],
                                "to_concept": m["to_concept"],
                            },
                            "triple_cosine": m["sk_similar"]["cosine"]
                            if m["sk_similar"]["cosine"] > 0.0
                            else 0.0,
                        }
                    )
            else:
                rows.append(
                    {
                        "question_id": str(q_idx),
                        "model": model,
                        "dataset": ds,
                        "sk_triple": None,
                        "rk_triple": None,
                        "triple_cosine": 0.0,
                    }
                )

            temp_df = pd.DataFrame.from_records(rows)
            df = pd.concat([df, temp_df], ignore_index=True)

    print(f"Made dataframe with {len(df)} rows.")

    # Add sample_cosine
    df["sample_cosine"] = 0.0
    for m, ds in zip(all_models, all_ds):
        curr_df = df[(df["dataset"] == ds) & (df["model"] == m)]
        sample_ids = list(curr_df["question_id"].unique())
        for idx in sample_ids:
            curr_sample = curr_df[curr_df["question_id"] == idx]
            sample_sim = np.average(curr_sample["triple_cosine"]).item()
            df_row_ids = curr_sample.index  # These will be the ones to update
            df["sample_cosine"].iloc[df_row_ids] = sample_sim

    prec = 6
    # Compute mock columns
    strict_split = {
        1: {"start": 0.75, "end": 1.0},
        2: {"start": 0.333334, "end": 0.749999},
        3: {"start": 0.0, "end": 0.333333},
    }
    df["triple_type"] = df.apply(
        lambda x: get_behaviour_type(x["triple_cosine"], strict_split), axis=1
    )

    # Init 'sample_type' col
    df["sample_type"] = 0
    df["slack"] = False

    print("Added mock columns.")
    print("Starting sensitivity sweep...")

    n_points = 10
    t_min = 0.0
    t_max = 1.0
    thresholds = np.linspace(0, 1, n_points + 2)[1:-1]
    results = {}
    min_gap = 0.000001

    for m, ds in zip(all_models, all_ds):
        curr_df = df[(df["dataset"] == ds) & (df["model"] == m)].copy()
        sample_ids = list(curr_df["question_id"].unique())
        if m not in results:
            results[m] = {}
        if ds not in results[m]:
            results[m][ds] = {1: [], 2: [], 3: []}

        print(f"{m} x {ds} --> {len(curr_df)} rows.")

        for _, t1 in enumerate(thresholds):
            for _, t2 in enumerate(thresholds):
                # force order
                if t1 >= t2:
                    continue
                curr_split = {
                    1: {"start": round(t2.item(), prec), "end": t_max},
                    2: {
                        "start": round(t1.item() + min_gap, prec),
                        "end": round(t2.item() - min_gap, prec),
                    },
                    3: {"start": t_min, "end": round(t1.item(), prec)},
                }
                print(curr_split)

                for idx in sample_ids:
                    curr_sample = curr_df[curr_df["question_id"] == idx]
                    df_row_ids = curr_sample.index  # These will be the ones to update
                    # print(len(curr_df), df_row_ids)
                    sample_info = get_sample_type(curr_sample, curr_split)
                    curr_df.loc[df_row_ids, "sample_type"] = sample_info[0]

                # Compute stats for sample-level behaviours assigned

                for idx in curr_split.keys():
                    count = len(
                        curr_df[(~curr_df["slack"]) & (curr_df["sample_type"] == idx)][
                            "question_id"
                        ].unique()
                    )
                    results[m][ds][idx].append(((t2, t1), count))

    print("Done. Saving.")
    # Save results
    out_dir = Path("sensitivity_res")
    out_dir.mkdir(exist_ok=True)
    filename = Path(out_dir, f"sensitivity_agg_{n_points}.json")
    with open(filename, "w") as fp:
        json.dump(results, fp)
    print("Saved.")
