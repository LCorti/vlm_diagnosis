# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: clone
#     language: python
#     name: python3
# ---

# %%
import copy
import networkx as nx
import pandas as pd
import statsmodels.api as sm
import sys
import warnings

from dowhy import CausalModel
from econml.inference import BootstrapInference
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from word2num import Word2Num

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.dataset_config_loader import DatasetConfig
from config_loaders.ca_config_loader import CAConfig
from config_loaders.rk_handler import RKHandler
from config_loaders.sk_handler import SKHandler
from utils.data_io import load_jsonl, load_pickle, make_dir, save_json
from utils.vis import save_graph_to_img

warnings.filterwarnings("ignore")

# %%
PROMPT_VERSION = 4
MODEL = "internvl2"
DATASET = "vqav2"

# %%
rk_hdl = RKHandler()
sk_hdl = SKHandler()
ds_config = DatasetConfig()

ds_list = ds_config.get_ds_list()
ds_list.remove("vqav2_holdout")

# %% [markdown]
# Load responses_counter
# - These files already include the answer for the "complete RK"

# %%
# TODO: change this to Path(__file__).parent for script
base_dir = Path("..")
rk_hdl.set_curr_model(MODEL)
rk_hdl.set_curr_ds(DATASET)
resps_path = base_dir.joinpath(str(rk_hdl.get_parsed_rk_path()).format(PROMPT_VERSION))
print(resps_path)
resps = load_jsonl(resps_path)

alt_resps_path = base_dir.joinpath(rk_hdl.get_counterfactual_rk_path())
print(alt_resps_path)
alt_resps = load_jsonl(alt_resps_path)

# %%
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

# %% [markdown]
# For each question, add a new entry corresponding to the original response.

# %%
resps[0].keys()

# %%
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
    else:
        print(f"We have an issue with question {q_idx}.")

# %% [markdown]
# Create dataframe out of counterfactual data and responses

# %%
df = pd.DataFrame(all_resps)
df.drop(["path"], axis=1, inplace=True)

# %%
df.head()

# %% [markdown]
# Make ```response``` lowercase for consistency

# %%
df["response"] = df["response"].apply(lambda x: x.lower())

# %%
df[df["size_treatment"] == 0]

# %%
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


# %% [markdown]
# Fix numerical answers given as words (e.g., "Three")


# %%
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


w2n = Word2Num()
df["response"] = df["response"].apply(lambda x: fix_word_numbers(w2n, x))

# %%
df.head()

# %% [markdown]
# Add column ```y``` representing the model answer
# - If response is numeric, just use that
# - If response is text, create lookup for the results of that question

# %%
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

# %%
df.head()

# %%
# Make dictionary of dataframes w.r.t. question id
df_dict = {q_idx: df[df["question_id"] == q_idx] for q_idx in dict_counter_resp}

# %%
df_dict[337814001].head()

# %% [markdown]
# Let us make binary columns indicating whether a concept was occluded or not.
# By default, all concepts are present (1). Then, we zero out the ones occluded.
#
# Note: it is done at this point because each question has a different number of concepts.

# %%
# Adding 1 column (of zeros) to each sub-dataframe for each concept found
for q_idx in df_dict:
    for concept in dict_concepts[q_idx]:
        df_dict[q_idx][concept] = 1

# %%
df_dict[337814001].head()

# %%
# Correcting the dataframes based on occlusions
for q_idx in df_dict:
    for id, row in df_dict[q_idx].iterrows():
        curr_concepts = row["occluded_concepts"]
        for c in curr_concepts:
            df_dict[q_idx].loc[id, c] = 0
    df_dict[q_idx].drop(["occluded_concepts"], axis=1, inplace=True)

# %% [markdown]
# View the different responses from the models

# %%
# for q_idx in df_dict:
#     unique_res = df_dict[q_idx]["response"].unique()
#     print(f"Question {q_idx} -> {len(unique_res)} unique responses {unique_res}")

# %%
df_dict[337814001].head()

# %% [markdown]
# ```df_dict```
# - Dictionary with question_ids as keys for which we have valid RKs
# - Each entry in the dictionary is a dataframe with
#   - The original response (no occlusion; size_treatment=0)
#   - A bunch of rows with the counterfactual responses (combinations of concepts)
#   - A series of binary columns matching 1:1 the concepts present or occluded

# %% [markdown]
# ### Causal Analysis

# %% [markdown]
# First, we load the NX graphs and make the .dot string required by the dowhy package.
#
# Make sure to run ```should_know_processing/pickle_sk.py``` to obtain the pickled files first.

# %%
# Load pickled SK graphs
sk_hdl.set_curr_ds(DATASET)
sk_classes = sk_hdl.get_classes()
sk_graphs = {}
all_img_ids = []
for ds_class in sk_classes:
    sk_hdl.set_curr_class(ds_class)
    curr_graphs = load_pickle(Path("..", sk_hdl.get_pkl_path()))
    all_img_ids.extend(list(curr_graphs.keys()))
    print(f"Loaded {len(curr_graphs)} from {DATASET}+{ds_class}")
    print(f"- Old graph dict: {len(sk_graphs)} entries.")
    sk_graphs.update(curr_graphs)
    print(f"- New graph dict: {len(sk_graphs)} entries.")

# %%
print(f"Duplicate ids: {set([x for x in all_img_ids if all_img_ids.count(x) > 1])}")

# %%
# Add to each graph a node corresponding to the outcome variable 'y'
for sk in sk_graphs.values():
    sk.add_node("y", label="response")

    for node in list(sk.nodes()):
        sk.add_edge(node, "y")

# %%
# Load question data. Needed to match df_dict and sk_graphs
ds_paths = ds_config.get_ds_paths(DATASET)
questions = []
for ds_class in ds_paths:
    questions_path = Path("..").joinpath(ds_paths[ds_class]["sampled_questions"])
    data = load_jsonl(questions_path)
    questions.extend(data)
print(f"Loaded {len(questions)} questions from {DATASET}.")


# %% [markdown]
# Running causal analysis


# %%
def run_causal_inference(data, nx_graph, do_refute=False):
    all_estimates = []
    refutations = []

    # Prepare data for multinomial logistic regressor
    # responses = data["y"]  # 1-d vector of the responses
    cols_to_skip = ["question_id", "size_treatment", "response"]
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
            # missing_nodes_as_confounders=True,
        )
        estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Control and treatment values are correct since we
        # are occluding (removing) elements from images.
        ce_estimate = model.estimate_effect(
            estimand,
            # method_name="backdoor.generalized_linear_model",
            method_name="backdoor.linear_regression",
            control_value=1,
            treatment_value=0,
            test_significance=True,
            confidence_intervals=True,
            fit_estimator=True,
            # method_params={"glm_family": sm.families.Binomial()},
        )

        all_estimates.append(ce_estimate)

        # Run refutation with Placebo
        if do_refute:
            res_refutation = model.refute_estimate(
                estimand,
                ce_estimate,
                method_name="placebo_treatment_refuter",
                show_progress_bar=False,
                placebo_type="permute",
            )
            refutations.append({col: res_refutation})

    # Sort causal estimates before returning
    sorted_estimates = sorted(all_estimates, key=lambda x: x["value"], reverse=True)
    return sorted_estimates, refutations


# %%
def run_causal_inference_DML(
    data, nx_graph, test_significance=False, confidence_intervals=False, do_refute=False
):
    all_estimates = []
    refutations = []

    # Prepare data for multinomial logistic regressor
    # responses = data["y"]  # 1-d vector of the responses
    cols_to_skip = ["question_id", "size_treatment", "response"]
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

        all_estimates.append(ce_estimate)

        # Run refutation with Placebo
        if do_refute:
            res_refutation = model.refute_estimate(
                estimand,
                ce_estimate,
                method_name="placebo_treatment_refuter",
                show_progress_bar=False,
                placebo_type="permute",
            )
            refutations.append({col: res_refutation})

    # Sort causal estimates before returning
    sorted_estimates = sorted(all_estimates, key=lambda x: x.value, reverse=True)
    return sorted_estimates, refutations


# %%
estimates = {}
refutations = {}

for curr_q in questions:
    curr_idx = curr_q["question_id"]
    print(f">> Working with question # {curr_idx}")
    curr_sk_graph = copy.deepcopy(sk_graphs[curr_q["img"]])
    curr_df = df_dict[curr_idx]

    # removing self loops for causal analysis
    curr_sk_graph.remove_edges_from(nx.selfloop_edges(curr_sk_graph))

    # Save graph image
    # curr_out_dir = f"{CE_OUT_DIR}/{MODEL}/{DATASET}"
    # make_dir(curr_out_dir)
    # save_graph_to_img(g_idx, graph, curr_out_dir)

    # Compute causal effects
    estimates[curr_idx], refutations[curr_idx] = run_causal_inference_DML(
        curr_df,
        curr_sk_graph,
        test_significance=False,
        confidence_intervals=False,
        do_refute=False,
    )

    # Do computations only for 1, for now
    break

# %% [markdown]
# Save causal estimates to file

# %%
ca_config = CAConfig()
estimates_out_file = Path(ca_config.get_ca_estimates_paths(MODEL, DATASET))
estimates_out_path = base_dir.joinpath(estimates_out_file).resolve()
make_dir(estimates_out_path.parent)

# %%
import io
from contextlib import redirect_stdout


# %%
# Preparing data to save
def format_estimates(estimates, test_significance=False, std_error=False):
    out_data = {}
    for q_idx in estimates:
        if q_idx not in out_data:
            out_data[q_idx] = {}

        for curr_est in estimates[q_idx]:
            var_name = curr_est._treatment_name[0]
            if var_name not in out_data[q_idx]:
                out_data[q_idx][var_name] = {}

            to_save = {
                "estimate": curr_est.value,
                "estimand_type": curr_est.target_estimand.estimand_type.value,
                "estimand_expr": curr_est.realized_estimand_expr,
            }

            # Catch the output of the interpret() method
            f = io.StringIO()
            with redirect_stdout(f):
                curr_est.interpret()
            to_save["interpretation"] = f.getvalue()

            if test_significance:
                to_save["p-value"] = curr_est.test_stat_significance()
                # TODO: fix this
                to_save["significance"] = None

            if std_error:
                to_save["std_error"] = curr_est.get_standard_error()

            out_data[q_idx][var_name] = to_save
    return out_data


out_data = format_estimates(estimates, test_significance=False, std_error=False)

# %%
# save
save_json(out_data, estimates_out_path)
