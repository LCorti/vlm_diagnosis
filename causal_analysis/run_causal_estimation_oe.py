import argparse
import copy
import io
import numpy as np
import pandas as pd
import spacy
import sys
import warnings

from contextlib import redirect_stdout
from dowhy import CausalModel
from econml.inference import BootstrapInference
from importlib import import_module
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

import utils.data_io as data_io
import utils.graph_utils as graph_utils

from config_handlers.causal_handler import CausalHandler
from config_handlers.rk_handler import RKHandler

warnings.filterwarnings("ignore")

# Utils


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


# Credit: https://github.com/BramVanroy/spacy_download
def load_spacy(model_name: str, **kwargs) -> spacy.Language:
    """Load a spaCy model, download it if it has not been installed yet.
    :param model_name: the model name, e.g., en_core_web_sm
    :param kwargs: options passed to the spaCy loader, such as component exclusion, as you
    would with spacy.load()
    :return: an initialized spaCy Language
    :raises: SystemExit: if the model_name cannot be downloaded
    """
    try:
        model_module = import_module(model_name)
    except ModuleNotFoundError:
        spacy.cli.download(model_name)
        model_module = import_module(model_name)

    return model_module.load(**kwargs)


def get_nouns(doc: spacy.tokens.Doc, return_dict: bool = True) -> dict | list:
    accepted_pos_tags = ["NN", "NNP", "NNPS", "NNS"]
    """ If needed, add pronouns: ["PRP", "PRP$", "WP", "WP$"] """
    accepted_dep_tags = [
        "nsubj",
        "nsubjpass",
        "dobj",
        "attr",
        "agent",
        "oprd",
        "amod",
        "nummod",
        "compound",
        "nmod",
        "pobj",
        "conj",
    ]

    resp_concepts = []
    for token in doc:
        if token.tag_ in accepted_pos_tags:
            # Return lemma_ insted of text
            if return_dict:
                new_entry = {
                    "text": token.lemma_,
                    "tag": token.tag_,
                    "subtree": [t.text for t in list(token.subtree)],
                }
            else:
                new_entry = token.lemma_
            resp_concepts.append(new_entry)
    return resp_concepts


# Causal Inference utils


def make_nx_graphs(resps):
    graphs = {}
    for r in resps:
        print(f"Processing sample id: {r['question_id']}")
        graphs[str(r["question_id"])] = graph_utils.rk_to_nx(
            r["triple_objs"], return_full=True
        )
        print("-" * 40)
    return graphs


def run_causal_inference_DML(
    data,
    nx_graph,
    out_var,
    test_significance=False,
    confidence_intervals=False,
    get_stderr=False,
    do_refute=False,
):
    all_estimates = {}
    all_refutations = {}

    # Prepare data for multinomial logistic regressor
    # responses = data["y"]  # 1-d vector of the responses
    cols_to_skip = ["question_id", "size", "response"]
    cols_to_keep = []
    for col in list(set(data.keys()) - set(cols_to_skip)):
        if col.startswith("y_") and col != out_var:
            continue
        cols_to_keep.append(col)
    data_for_estimation = data[cols_to_keep]  # n_observations x regressors
    regressors = list(data_for_estimation.drop(out_var, axis=1).keys())

    for col in regressors:
        print(f"Working on {col}...")

        # Exclude the current 'col' from the variables that can condition the estimates
        effect_modifiers = [r for r in regressors if r != col]

        # effect_modifiers: estimates are separately computed (conditioned)
        # for each value of the effect_modifier.
        model = CausalModel(
            data=data_for_estimation,
            treatment=col,
            outcome=out_var,
            graph=nx_graph,
            effect_modifiers=effect_modifiers,
        )
        estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Define method params
        # If less than default n_splits (5), use only 1 split
        n_split = 2 if len(data) < 5 else 5
        method_params = {
            "init_params": {
                "model_y": GradientBoostingRegressor(),
                "model_t": GradientBoostingRegressor(),
                "model_final": LassoCV(fit_intercept=False, cv=n_split),
                "featurizer": PolynomialFeatures(degree=1, include_bias=False),
            },
            "fit_params": {},
        }
        if test_significance:
            method_params["fit_params"] = {
                "inference": BootstrapInference(n_bootstrap_samples=1, n_jobs=-1)
            }

        # Control and treatment values are correct since we are occluding (removing)
        # elements from images.
        try:
            ce_estimate = model.estimate_effect(
                estimand,
                method_name="backdoor.econml.dml.DML",
                control_value=1,
                treatment_value=0,
                fit_estimator=True,
                method_params=method_params,
            )
        except Exception as e:
            print("Unable to run estimation. Got the following exception:")
            print(e)
            ce_estimate = None
            all_estimates[col] = ce_estimate
            continue

        # Run refutation with Placebo method
        if do_refute:
            print("- Running refutation test for estimate...")
            try:
                all_refutations[col] = model.refute_estimate(
                    estimand,
                    ce_estimate,
                    method_name="placebo_treatment_refuter",
                    show_progress_bar=False,
                    placebo_type="permute",
                )
            except Exception as e:
                print(
                    "Unable to run refutation test of obtained estimate. Got the following exception:"
                )
                print(e)
                all_refutations[col] = None

        # The following calls are is wrapped in a try-except because it might fail
        # when test_significance, confidence_intervals, or get_stderr are True.
        # This is due to incomplete integration between the DoWhy and econML libraries.

        if test_significance:
            print("- Running significance test for estimate...")
            try:
                ce_estimate.test_stat_significance()
            except Exception as e:
                print(
                    "Unable to test significance of obtained estimate. Got the following exception:"
                )
                print(e)

        if confidence_intervals:
            print("- Getting confidence intervals for estimate...")
            try:
                ce_estimate.get_confidence_intervals()
            except Exception as e:
                print(
                    "Unable to get confidence intervals of the obtained estimate. Got the following exception:"
                )
                print(e)

        # Compute standard error for the estimate
        if get_stderr:
            print("- Getting standard error for estimate...")
            try:
                ce_estimate.get_standard_error()
            except Exception as e:
                print(
                    "Unable to get the standard error for the obtained estimate. Got the following exception:"
                )
                print(e)

        all_estimates[col] = ce_estimate

    return all_estimates, all_refutations


def format_estimates(
    estimates,
    refutations=None,
    test_significance=False,
    confidence_intervals=False,
    std_error=False,
):
    out_data = {}

    for var_name, curr_est in estimates.items():
        # var_name = curr_est._treatment_name[0]
        if var_name not in out_data:
            out_data[var_name] = {}

        if curr_est is None:
            continue

        out_data[var_name] = {
            "estimate": curr_est.value.item(),
            "estimand_type": curr_est.target_estimand.estimand_type.value,
            "estimand_expr": curr_est.realized_estimand_expr,
        }

        # Catch the output of the interpret() method
        f = io.StringIO()
        with redirect_stdout(f):
            curr_est.interpret()
        out_data[var_name]["interpretation"] = f.getvalue().strip()

        if test_significance:
            try:
                curr_est.test_stat_significance()
                p_val = curr_est.test_stat_significance()["p_value"]
                if isinstance(p_val, tuple):
                    out_data[var_name]["p_value"] = list(p_val)
                elif isinstance(p_val, np.floating):
                    out_data[var_name]["p_value"] = p_val.item()
            except Exception as _:
                out_data[var_name]["p_value"] = None

        if std_error:
            try:
                out_data[var_name]["std_error"] = curr_est.get_standard_error()
            except Exception as _:
                out_data[var_name]["std_error"] = None

        if confidence_intervals:
            try:
                out_data[var_name]["conf_intervals"] = (
                    curr_est.get_confidence_intervals()
                )
            except Exception as _:
                out_data[var_name]["conf_intervals"] = None

        if refutations:
            refute_data = refutations[var_name]
            if refute_data:
                p_val = refute_data.refutation_result["p_value"]
                out_data[var_name]["refute_test"] = {
                    "is_stat_significant": refute_data.refutation_result[
                        "is_statistically_significant"
                    ].item(),
                }
                if isinstance(p_val, tuple):
                    out_data[var_name]["refute_test"]["p_value"] = list(p_val)
                elif isinstance(p_val, np.floating):
                    out_data[var_name]["refute_test"]["p_value"] = p_val.item()
            else:
                out_data[var_name]["refute_test"] = {}
        else:
            out_data[var_name]["refute_test"] = {}

    return out_data


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

    if TEST_SIGNIFICANCE or CONF_INTERVALS or OUTPUT_STDERR:
        print(
            "WARNING: The integration of DoWhy and econML is incomplete. "
            "Computations for testing significance, retrieving confidence "
            "intervals, and standard error for the obtained estimate might fail."
        )

    # Load data handlers
    rk_hdl = RKHandler()
    rk_hdl.set_curr_model(MODEL)
    rk_hdl.set_curr_ds(DATASET)
    causal_hdl = CausalHandler()
    causal_hdl.set_curr_ds(DATASET)
    causal_hdl.set_curr_model(MODEL)

    base_dir = Path("..")
    resps = data_io.load_jsonl(base_dir.joinpath(rk_hdl.get_rk_final_path()))
    counter_resps = data_io.load_jsonl(
        base_dir.joinpath(causal_hdl.get_counter_resps_path())
    )

    print("=" * 60)
    print(f"Processing data for {MODEL} + {DATASET}...")
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
                    "question_id": str(og_resp["question_id"]),
                    "size": 0,
                    "occluded": [],
                    "response": og_resp["response"],
                }
            )
        else:
            print(f"We have an issue with question {q_idx}.")

    df = pd.DataFrame(all_resps)
    df.drop(["path"], axis=1, inplace=True)
    # Force question_id column type to string
    df["question_id"] = df["question_id"].map(str)
    df["response"] = df["response"].apply(lambda x: x.lower())

    max_triplets = {
        str(q_idx): max(
            [int(q["size"]) for q in all_resps if q["question_id"] == q_idx]
        )
        for q_idx in resps_ids
    }
    # This is done to extract an actual list of concepts, not any other PD objects.
    dict_concepts = {
        str(q_idx): df[
            (df["question_id"] == q_idx) & (df["size"] == max_triplets[q_idx])
        ]["occluded"].iloc[0]
        for q_idx in max_triplets
    }

    # Make dictionary of dataframes w.r.t. question id
    out_f = Path("data_oe")
    out_f.mkdir(parents=True, exist_ok=True)
    file_name = out_f.joinpath(f"df_dict_{MODEL}_{DATASET}.pkl")
    if file_name.exists():
        print("Found pre-computed 'df_dict'. Loading...")
        df_dict = pd.read_pickle(file_name)
        print("Creating unique names...")
        unique_og = {}
        for q_idx, q_df in df_dict.items():
            # Get all columns that start with "y_" -> these are the nouns to check
            unique_og[q_idx] = [
                col.replace("y_", "", 1) for col in q_df.columns if col.startswith("y_")
            ]
    else:
        df_dict = {
            str(q_idx): df[df["question_id"] == str(q_idx)]
            for q_idx in dict_counter_resp
        }

        # Taking the list of occluded concepts and splitting it
        for q_idx in df_dict:
            for concept in dict_concepts[q_idx]:
                df_dict[q_idx][concept] = 1
        for q_idx in df_dict:
            for id, row in df_dict[q_idx].iterrows():
                curr_concepts = row["occluded"]
                for c in curr_concepts:
                    df_dict[q_idx].loc[id, c] = 0
            df_dict[q_idx].drop(["occluded"], axis=1, inplace=True)

        print("Finish setting up df_dict.")
        spacy_model_name = "en_core_web_trf"
        print(f"Loading spaCy model: {spacy_model_name}...")
        to_exclude = ["ner", "parser"]  # do not load NER and dependency parsing modules
        nlp = load_spacy(spacy_model_name, exclude=to_exclude)

        """Only use the nouns that are present in the original response to do estimation."""

        # Prepare columns with nouns extracted from the responses
        unique_counter = {}
        unique_og = {}
        for q_idx, q_df in df_dict.items():
            # Parse original response (i.e., the non-counterfactual one) separately
            unique_counter[q_idx] = {}
            for idx in q_df.index.tolist():
                size = q_df.loc[idx]["size"]
                # These are lemmas already
                nouns = get_nouns(nlp(q_df.loc[idx]["response"]), return_dict=False)
                unique_counter[q_idx][idx] = list(set(nouns))
                if size == 0:
                    unique_og[q_idx] = list(set(nouns))

            # Add binary columns for each noun found in the OG response. Initialise with 1s.
            for noun in unique_og[q_idx]:
                q_df[f"y_{noun}"] = 1

            # Iterate over the rows and if a column is not included in the list
            for row_id in q_df.index:
                for noun in unique_og[q_idx]:
                    if noun not in unique_counter[q_idx][row_id]:
                        q_df.loc[row_id, f"y_{noun}"] = 0

            print(
                f"Handled question: {q_idx} -> {q_df.shape[0]} rows x {q_df.shape[1]} cols."
            )

        print("Completed creation of df_dict. Saving copy to disk...")
        pd.to_pickle(df_dict, file_name)

    # Prepare data for causal inference
    # First, we create the NX graphs based on the data contained from 'resps'.
    # This is done so that we can create the '.dot' string required by 'dowhy'.
    rk_graphs = make_nx_graphs(resps)
    img_ids = list(rk_graphs.keys())
    print(f"Loaded {len(rk_graphs)} samples.")
    print(f"Duplicate ids: {set([x for x in img_ids if img_ids.count(x) > 1])}")

    # Add to each graph a node corresponding to the outcome variable 'y'.
    # Here, since we have multiple variables, we need to add several of these nodes.
    # Can't add all of the y_i nodes to the graph because then the estimators don't know
    # what to do. Need to make separate graphs, each with a y_i.

    exp_rk_graphs = {}
    for rk_idx, rk_g in rk_graphs.items():
        exp_rk_graphs[rk_idx] = {}
        for noun in unique_og[rk_idx]:
            exp_rk_graphs[rk_idx][f"y_{noun}"] = copy.deepcopy(rk_g)
            exp_rk_graphs[rk_idx][f"y_{noun}"].add_node(f"y_{noun}", label=f"y_{noun}")

            for node in list(exp_rk_graphs[rk_idx][f"y_{noun}"].nodes()):
                if node == f"y_{noun}":
                    continue
                exp_rk_graphs[rk_idx][f"y_{noun}"].add_edge(node, f"y_{noun}")

    # Run causal analysis
    causal_out_path = base_dir.joinpath(causal_hdl.get_estimates_path()).resolve()
    data_io.make_dir(causal_out_path.parent)
    print(f"Saving estimates to: {str(causal_out_path)}")

    # Load data currently available
    if causal_out_path.exists():
        old_data = data_io.load_jsonl(causal_out_path)
        old_ids = set([od["question_id"] for od in old_data])
    else:
        print("First run.")
        old_ids = []

    # Do estimation
    for q_idx, rk_data in exp_rk_graphs.items():
        if q_idx in old_ids:
            print(f"Already processed id {q_idx}. Moving to next one.")
            continue

        print("=" * 60)
        print(f">> Working with question # {q_idx} -- Samples: {len(df_dict[q_idx])}")
        curr_df = df_dict[q_idx]

        outcome_vars = [v for v in df_dict[q_idx].columns if v.startswith("y_")]
        out_data = {"question_id": q_idx, "estimates": {}}
        for out_var in outcome_vars:
            curr_rk = rk_data[out_var].copy()
            print(f"Current outcome var: {out_var}")
            # Compute causal effects
            estimates, refutations = run_causal_inference_DML(
                curr_df,
                curr_rk,
                out_var,
                test_significance=TEST_SIGNIFICANCE,
                confidence_intervals=CONF_INTERVALS,
                get_stderr=OUTPUT_STDERR,
                do_refute=DO_REFUTE,
            )

            # Format and save results of causal analysis
            out_data["estimates"][out_var] = format_estimates(
                estimates,
                refutations=refutations,
                test_significance=TEST_SIGNIFICANCE,
                confidence_intervals=CONF_INTERVALS,
                std_error=OUTPUT_STDERR,
            )
        # Still save data after every sample is done
        data_io.append_to_jsonl([out_data], causal_out_path)

    print("=" * 30)
    print(f"Done for {MODEL} x {DATASET}.")
    print("=" * 30)
