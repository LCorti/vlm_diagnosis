import argparse
import copy
import io
import networkx as nx
import numpy as np
import pandas as pd
import sys
import warnings

from contextlib import redirect_stdout
from dowhy import CausalModel
from econml.inference import BootstrapInference
from networkx import DiGraph
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from word2num import Word2Num

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

import utils.data_io as data_io
import utils.graph_utils as graph_utils

from config_handlers.causal_handler import CausalHandler
from config_handlers.rk_handler import RKHandler
# from utils.vis import save_graph_to_img

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Config for causal analysis.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--test_significance",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--conf_intervals",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--output_stderr",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--do_refute", type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


# Formatting utils
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


def rk_to_nx(raw_rk_rels: dict) -> DiGraph:
    rk_list = [
        {
            "from_concept": rk["from_concept"]["bb_label"]["bb_label_full"],
            "relationship": rk["rel_label"]["rel_label_text"],
            "to_concept": rk["to_concept"]["bb_label"]["bb_label_full"],
        }
        for rk in raw_rk_rels
    ]
    return graph_utils.create_nx_graph(rk_list)


def make_nx_graphs(resps):
    graphs = {}
    for r in resps:
        graphs[r["question_id"]] = rk_to_nx(r["triple_objs"])
    return graphs


def format_estimates(
    estimates, refutations=None, test_significance=False, std_error=False
):
    out_data = {}
    for q_idx in estimates:
        if q_idx not in out_data:
            out_data[q_idx] = {}

        for var_name, curr_est in estimates[q_idx].items():
            # var_name = curr_est._treatment_name[0]
            if var_name not in out_data[q_idx]:
                out_data[q_idx][var_name] = {}

            to_save = {
                "estimate": curr_est.value.item(),
                "estimand_type": curr_est.target_estimand.estimand_type.value,
                "estimand_expr": curr_est.realized_estimand_expr,
            }

            # Catch the output of the interpret() method
            f = io.StringIO()
            with redirect_stdout(f):
                curr_est.interpret()
            to_save["interpretation"] = f.getvalue().strip()

            if refutations:
                refute_data = refutations[q_idx][var_name]
                p_val = refute_data.refutation_result["p_value"]
                to_save["refute_test"] = {
                    "is_stat_significant": refute_data.refutation_result[
                        "is_statistically_significant"
                    ].item(),
                }
                if isinstance(p_val, tuple):
                    to_save["refute_test"]["p_value"] = list(p_val)
                elif isinstance(p_val, np.floating):
                    to_save["refute_test"]["p_value"] = p_val.item()

            if test_significance:
                p_val = curr_est.test_stat_significance()["p_value"]
                if isinstance(p_val, tuple):
                    to_save["p_value"] = list(p_val)
                elif isinstance(p_val, np.floating):
                    to_save["p_value"] = p_val.item()
                # to_save["significance"] = None

            if std_error:
                to_save["std_error"] = curr_est.get_standard_error()

            out_data[q_idx][var_name] = to_save
    return out_data


# Causal analysis
def run_causal_inference_DML(
    data,
    nx_graph,
    test_significance=False,
    confidence_intervals=False,
    do_refute=False,
):
    all_estimates = {}
    all_refutations = {}

    # Prepare data for multinomial logistic regressor
    # responses = data["y"]  # 1-d vector of the responses
    cols_to_skip = ["question_id", "size", "response"]
    cols_to_keep = list(set(data.keys()) - set(cols_to_skip))
    data_for_estimation = data[cols_to_keep]  # n_observations x regressors
    regressors = list(data_for_estimation.drop("y", axis=1).keys())

    for col in regressors:
        print(f"Working on {col}...")

        # Exclude the current 'col' from the variables that can condition the estimates
        effect_modifiers = [r for r in regressors if r != col]

        # effect_modifiers: estimates are separately computed (conditioned)
        # for each value of the effect_modifier.
        model = CausalModel(
            data=data_for_estimation,
            treatment=col,
            outcome="y",
            graph=nx_graph,
            effect_modifiers=effect_modifiers,
        )
        estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Define method params
        method_params = {
            "init_params": {
                "model_y": GradientBoostingRegressor(),
                "model_t": GradientBoostingRegressor(),
                "model_final": LassoCV(fit_intercept=False),
                "featurizer": PolynomialFeatures(degree=1, include_bias=False),
            },
            "fit_params": {},
        }
        if test_significance:
            method_params["fit_params"] = {
                "inference": BootstrapInference(n_bootstrap_samples=1, n_jobs=-1)
            }

        # Control and treatment values are correct since we
        # are occluding (removing) elements from images.
        ce_estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.econml.dml.DML",
            control_value=1,
            treatment_value=0,
            test_significance=test_significance,
            confidence_intervals=confidence_intervals,
            fit_estimator=True,
            method_params=method_params,
        )
        # Compute standard error for the estimate
        ce_estimate.get_standard_error()
        all_estimates[col] = ce_estimate

        # Run refutation with Placebo
        if do_refute:
            all_refutations[col] = model.refute_estimate(
                estimand,
                ce_estimate,
                method_name="placebo_treatment_refuter",
                show_progress_bar=False,
                placebo_type="permute",
            )

    return all_estimates, all_refutations


if __name__ == "__main__":
    """ Setup """
    # Parse arguments
    args = parse_args()
    MODEL = args.model
    DATASET = args.dataset
    TEST_SIGNIFICANCE = args.test_significance
    CONF_INTERVALS = args.conf_intervals
    OUTPUT_STDERR = args.output_stderr
    DO_REFUTE = args.do_refute

    # Load data handlers
    rk_hdl = RKHandler()
    rk_hdl.set_curr_model(MODEL)
    rk_hdl.set_curr_ds(DATASET)
    causal_hdl = CausalHandler()
    causal_hdl.set_curr_ds(DATASET)
    causal_hdl.set_curr_model(MODEL)

    # Load model responses
    base_dir = Path(__file__).parent
    resps = data_io.load_jsonl(base_dir.joinpath(rk_hdl.get_rk_final_path()))
    counter_resps = data_io.load_jsonl(
        base_dir.joinpath(causal_hdl.get_counter_resps_path())
    )

    """ Preparing data """
    resps_ids = set([q["question_id"] for q in resps])
    dict_counter_resp = {
        q_idx: next(r["response"] for r in counter_resps if r["question_id"] == q_idx)
        for q_idx in resps_ids
    }

    # Merge initial and counterfactual responses
    all_resps = copy.deepcopy(counter_resps)
    for q_idx in resps_ids:
        og_resp = next((r for r in resps if r["question_id"] == q_idx), None)
        if og_resp:
            all_resps.append(
                {
                    "question_id": og_resp["question_id"],
                    "size": 0,
                    "occluded": [],
                    "response": og_resp["response"],
                }
            )
        else:
            print(f"We have an issue with question {q_idx}.")

    # Prepare dataframes for causal analysis
    df = pd.DataFrame(all_resps)
    df.drop(["path"], axis=1, inplace=True)
    df["response"] = df["response"].apply(lambda x: x.lower())

    # Create mapping between questions and the biggest size of constraints.
    max_triplets = {
        q_idx: max([int(q["size"]) for q in all_resps if q["question_id"] == q_idx])
        for q_idx in resps_ids
    }
    # This is done to extract an actual list of concepts, not any other PD objects.
    dict_concepts = {
        q_idx: df[(df["question_id"] == q_idx) & (df["size"] == max_triplets[q_idx])][
            "occluded"
        ].iloc[0]
        for q_idx in max_triplets
    }

    # Resolve numerical answers given as words (e.g., "three")
    w2n = Word2Num()
    df["response"] = df["response"].apply(lambda x: fix_word_numbers(w2n, x))

    # Add column 'y' representing the answer
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

    """
    Create dictionary of dataframes -- 'df_dict'
    - Dictionary with question_ids as keys for which we have valid RKs
    - Each entry in the dictionary is a dataframe with
        - The original response (no occlusion; size_treatment=0)
        - A bunch of rows with the counterfactual responses (combinations of concepts)
        - A series of binary columns matching 1:1 the concepts present or occluded
    """
    # Make dictionary of dataframes w.r.t. question id
    df_dict = {q_idx: df[df["question_id"] == q_idx] for q_idx in dict_counter_resp}

    # Making binary columns indicating whether a concept was occluded or not.
    # By default, all concepts are present (1). Then, we zero out the ones occluded.
    # Note: each data sample has a different amount of concepts 'attached' to it.
    for q_idx in df_dict:
        for concept in dict_concepts[q_idx]:
            df_dict[q_idx][concept] = 1
    for q_idx in df_dict:
        for id, row in df_dict[q_idx].iterrows():
            curr_concepts = row["occluded"]
            for c in curr_concepts:
                df_dict[q_idx].loc[id, c] = 0
        df_dict[q_idx].drop(["occluded"], axis=1, inplace=True)

    """ Causal Analysis """
    # First, we create the NX graphs based on the data contained from 'resps'.
    # This is done so that we can create the '.dot' string required by 'dowhy'.
    rk_graphs = make_nx_graphs(resps)
    img_ids = list(rk_graphs.keys())
    print(f"Loaded {len(rk_graphs)} from {MODEL} + {DATASET}")
    print(f"Duplicate ids: {set([x for x in img_ids if img_ids.count(x) > 1])}")

    # Add to each graph a node corresponding to the outcome variable 'y'
    for rk in rk_graphs.values():
        rk.add_node("y", label="response")

        for node in list(rk.nodes()):
            if node == "y":
                continue
            rk.add_edge(node, "y")

    # Run causal analysis
    estimates = {}
    refutations = {}

    for q_idx, rk_data in rk_graphs.items():
        print(f">> Working with question # {q_idx}")
        curr_rk = copy.deepcopy(rk_data)
        curr_df = df_dict[q_idx]

        # removing self loops for causal analysis
        loops = list(nx.selfloop_edges(curr_rk))
        if loops:
            print(loops)
        else:
            print("No loops found.")
        print("=" * 30)

        curr_rk.remove_edges_from(nx.selfloop_edges(curr_rk))

        # Save graph image
        # curr_out_dir = f"{CE_OUT_DIR}/{MODEL}/{DATASET}"
        # make_dir(curr_out_dir)
        # save_graph_to_img(g_idx, graph, curr_out_dir)

        # Compute causal effects
        estimates[q_idx], refutations[q_idx] = run_causal_inference_DML(
            curr_df,
            curr_rk,
            test_significance=True,
            confidence_intervals=True,
            do_refute=True,
        )

    # Format and save results of causal analysis
    causal_out_path = base_dir.joinpath(causal_hdl.get_estimates_path()).resolve()
    print(f"Saving estimates to: {str(causal_out_path)}")
    data_io.make_dir(causal_out_path.parent)
    out_data = format_estimates(
        estimates, refutations=refutations, test_significance=True, std_error=True
    )
    data_io.save_json(out_data, causal_out_path, indent=None)
