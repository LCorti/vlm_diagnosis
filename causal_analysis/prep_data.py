import argparse
import copy
import os
import pandas as pd
import sys
import warnings

from pathlib import Path
from word2num import Word2Num

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.dataset_config_loader import DatasetConfig
from config_loaders.ca_config_loader import CAConfig
from config_loaders.rk_config_loader import RKConfig
from config_loaders.sk_config_loader import SKConfig
from utils.data_io import load_jsonl, make_dir, save_jsonl

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Config for causal analysis.")
    parser.add_argument("--prompt_version", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def fix_word_numbers(parser, x):
    try:
        res = int(x)
    except Exception as _:
        res = parser.parse(x)
        if not res:
            res = x  # If parse fails, it is probably a string
        else:
            res = int(res)  # If parse works, make sure an int is returned
    finally:
        return res


if __name__ == "__main__":
    print("Configuring...")
    # Parse args
    args = parse_args()
    prompt_version = args.prompt_version
    dataset = args.dataset
    model = args.model

    #  Load config handlers
    ca_config = CAConfig()
    rk_config = RKConfig()
    sk_config = SKConfig()
    ds_config = DatasetConfig()
    ds_list = ds_config.get_ds_list()
    ds_list.remove("vqav2_holdout")

    # Load data
    base_dir = Path(__file__).parent.parent
    resps_path = rk_config.get_parsed_rk_paths(model, dataset).format(prompt_version)
    resps_path = base_dir.joinpath(resps_path).resolve()
    resps = load_jsonl(resps_path)

    alt_resps_path = rk_config.get_counterfactual_rk_paths(model, dataset)
    alt_resps_path = base_dir.joinpath(alt_resps_path).resolve()
    alt_resps = load_jsonl(alt_resps_path)

    # Get responses only for the max num of triplets
    # (i.e., ignore the rest of the powerset for now)
    resps_ids = set([q["question_id"] for q in resps])
    alt_resps_ids = set([q["question_id"] for q in alt_resps])

    # question_id: response
    dict_eval_resp = {
        q_idx: next(r["response"] for r in resps if r["question_id"] == q_idx)
        for q_idx in resps_ids
    }

    # question_id: response
    # this only gets the first response
    dict_counter_resp = {
        q_idx: next(r["response"] for r in alt_resps if r["question_id"] == q_idx)
        for q_idx in alt_resps_ids
    }

    # Based on the counterfactual responses we have, pick and add the original ones.
    all_resps = copy.deepcopy(alt_resps)
    for q_idx in alt_resps_ids:
        og_response = next((r for r in resps if r["question_id"] == q_idx), None)
        if og_response:
            all_resps.append(
                {
                    "question_id": og_response["question_id"],
                    "size_treatment": 0,
                    "occluded_concepts": [],
                    "response": og_response["response"],
                }
            )

    # Create dataframe out of counterfactual data and responses
    df = pd.DataFrame(all_resps)
    df.drop(["path"], axis=1, inplace=True)
    # Fix responses column
    df["response"] = df["response"].apply(lambda x: x.lower())

    # Create mapping between questions and the biggest size of constraints.
    dict_max_triplets = {
        q_idx: max(
            [int(q["size_treatment"]) for q in alt_resps if q["question_id"] == q_idx]
        )
        for q_idx in alt_resps_ids
    }

    # Make dictionary with {question_id: [concepts]}.
    # This is done to extract an actual list of concepts, not any other PD objects.
    dict_concepts = {
        q_idx: df[
            (df["question_id"] == q_idx)
            & (df["size_treatment"] == dict_max_triplets[q_idx])
        ]["occluded_concepts"].iloc[0]
        for q_idx in dict_max_triplets
    }

    # Fix numerical answers given as words (e.g., "Three")
    w2n = Word2Num()
    df["response"] = df["response"].apply(lambda x: fix_word_numbers(w2n, x))

    # Add column `y` representing the model answer
    # - If response is numeric, just use that
    # - If response is text, create lookup for the results of that question
    lookup_y = {}
    for idx, row in df.iterrows():
        q_idx = row["question_id"]
        if q_idx not in lookup_y:
            lookup_y[q_idx] = {}

        try:
            y = pd.to_numeric(row["response"])
            lookup_y[q_idx][row["response"]] = y
        except Exception as _:
            # Get all rows with same question ID
            curr_q = df[df["question_id"] == q_idx]
            # Create dict entry with list of unique responses
            lookup_y[q_idx] = {v: k for k, v in enumerate(curr_q["response"].unique())}
            y = lookup_y[q_idx][row["response"]]
        finally:
            df.loc[idx, "y"] = y

    # Make dictionary of dataframes w.r.t. question id
    df_dict = {q_idx: df[df["question_id"] == q_idx] for q_idx in dict_counter_resp}

    # Let us make binary columns indicating whether a concept was occluded or not.
    # By default, all concepts are present (1). Then, we zero out the ones occluded.
    # Note: done at this point because each question has a different number of concepts.

    # Adding 1 column (of zeros) to each sub-dataframe for each concept found
    for q_idx in df_dict:
        for concept in dict_concepts[q_idx]:
            df_dict[q_idx][concept] = 1
    # Correcting the dataframes based on occlusions
    for q_idx in df_dict:
        for id, row in df_dict[q_idx].iterrows():
            curr_concepts = row["occluded_concepts"]
            for c in curr_concepts:
                df_dict[q_idx].loc[id, c] = 0
        df_dict[q_idx].drop(["occluded_concepts"], axis=1, inplace=True)

    # `df_dict`
    # - Dictionary with question_ids as keys for which we have valid RKs
    # - Each entry in the dictionary is a dataframe with
    #   - The original response (no occlusion; size_treatment=0)
    #   - A bunch of rows with the counterfactual responses (combinations of concepts)
    #   - A series of binary columns matching 1:1 the concepts present or occluded

    # Save different dataframes to file for causal analysis
    out_data = []
    for q_idx in df_dict:
        for idx, row in df_dict[q_idx].iterrows():
            out_data.append(row.to_dict())

    ca_data_path = ca_config.get_ca_data_paths(model, dataset)
    out_file = base_dir.joinpath(ca_data_path).resolve()
    out_path = out_file.parent
    make_dir(out_path)
    save_jsonl(out_data, out_file)
