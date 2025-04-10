import argparse
import copy
import io
import networkx as nx
import os
import pandas as pd
import sys
import warnings

from contextlib import redirect_stdout
from dowhy import CausalModel
from econml.inference import BootstrapInference
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.dataset_config_loader import DatasetConfig
from config_loaders.ca_config_loader import CAConfig
from config_loaders.sk_config_loader import SKConfig
from utils.data_io import load_json, load_jsonl, load_pickle, make_dir, save_json
from utils.measures import compute_ic_relation
from utils.vis import save_graph_to_img

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


def make_df_dict(ca_data):
    df_dict = {}
    # Get all question_ids
    all_question_ids = list(set(entry["question_id"] for entry in ca_data))
    # Slice ca_data and add these to df_dict
    for q_idx in all_question_ids:
        rel_entries = list(filter(lambda entry: entry["question_id"] == q_idx, ca_data))
        # Convert each value in df_dict to pandas dataframe
        df_dict[q_idx] = pd.DataFrame(rel_entries)
    return df_dict


def handle_cycles(graph, cycles, sg_freqs):
    cycle_edges = {}
    for edge in cycles:
        # Each edge looks like ('9-tree', '23-leaf', 'forward')
        # 'forward' can be ignored, it represents the traversal direction, not the label
        from_concept = edge[0]
        to_concept = edge[1]
        edge_label = graph[from_concept][to_concept]["label"]
        from_concept_name = from_concept.split("-")[1]
        to_concept_name = to_concept.split("-")[1]
        triple = (from_concept_name, edge_label, to_concept_name)
        dict_key = f"{from_concept};{to_concept}"
        cycle_edges[dict_key] = compute_ic_relation(triple, sg_freqs)

    min_ic_edge = min(cycle_edges.values())
    to_remove = next((k for k in cycle_edges if cycle_edges[k] == min_ic_edge), None)

    if to_remove:
        nodes_to_unlink = to_remove.split(";")
        graph.remove_edge(nodes_to_unlink[0], nodes_to_unlink[1])
        return graph
    else:
        print("Error")


if __name__ == "__main__":
    print("Configuring...")
    # Parse args
    args = parse_args()
    prompt_version = args.prompt_version
    dataset = args.dataset
    model = args.model

    # Load config handlers
    ca_config = CAConfig()
    sk_config = SKConfig()
    ds_config = DatasetConfig()

    # Load data
    base_dir = Path(__file__).parent.parent
    ca_data_file = ca_config.get_ca_data_paths(model, dataset)
    ca_data_path = base_dir.joinpath(ca_data_file).resolve()
    ca_data = load_jsonl(ca_data_path)
    # Load scene graph counts
    sgg_freqs_path = base_dir.joinpath("data/common/sgg_freqs.json")
    sg_freqs = load_json(sgg_freqs_path)

    # Build df_dict
    df_dict = make_df_dict(ca_data)
    print("df_dict built.")

    # == == == == == == ==
    # Causal Analysis
    # == == == == == == ==
    # First, we load the NX graphs and make the .dot string required by the dowhy
    # package.Make sure to run `should_know_processing/pickle_sk.py` to obtain the
    # pickled files first.

    # Load pickled SK graphs
    sk_paths = sk_config.get_sk_paths(dataset)
    sk_graphs = {}
    all_img_ids = []
    for ds_class in sk_paths:
        sk_dir = sk_paths[ds_class]["dir"]
        sk_pkl_path = sk_paths[ds_class]["pkl"]
        curr_graphs = load_pickle(Path("..").joinpath(sk_dir).joinpath(sk_pkl_path))
        all_img_ids.extend(list(curr_graphs.keys()))
        print(f"Loaded {len(curr_graphs)} from {dataset}+{ds_class}")
        print(f"- Old graph dict: {len(sk_graphs)} entries.")
        sk_graphs.update(curr_graphs)
        print(f"- New graph dict: {len(sk_graphs)} entries.")

    # Add to each graph a node corresponding to the outcome variable 'y'
    for sk in sk_graphs.values():
        sk.add_node("y", label="y")
        nodes_to_link = [node for node in sk.nodes() if node != "y"]
        for node in nodes_to_link:
            sk.add_edge(node, "y")

    # Load question data. Needed to match df_dict and sk_graphs
    ds_paths = ds_config.get_ds_paths(dataset)
    questions = []
    for ds_class in ds_paths:
        questions_path = Path("..").joinpath(ds_paths[ds_class]["sampled_questions"])
        data = load_jsonl(questions_path)
        questions.extend(data)
    print(f"Loaded {len(questions)} questions from {dataset}.")

    # Running causal analysis
    estimates = {}
    refutations = {}

    for curr_q in questions:
        curr_idx = curr_q["question_id"]
        print(f">> Working with question # {curr_idx}")
        curr_sk_graph = copy.deepcopy(sk_graphs[curr_q["img"]])
        curr_df = df_dict[curr_idx]
        # save_graph_to_img(curr_sk_graph, "test.png")

        if nx.is_directed_acyclic_graph(curr_sk_graph):
            print("DAG found proceeding...")
        else:
            print("Not a DAG, fixing...")
            print("-> Handling selfloops...")
            # removing self loops for causal analysis
            curr_sk_graph.remove_edges_from(nx.selfloop_edges(curr_sk_graph))

            # handling cycles
            has_cycles = True
            while has_cycles:
                try:
                    cycles = nx.find_cycle(curr_sk_graph, orientation="original")
                except nx.exception.NetworkXNoCycle:
                    cycles = []

                if len(cycles) > 0:
                    print("-> Handling cycles...")
                    print(cycles)
                    curr_sk_graph = handle_cycles(curr_sk_graph, cycles, sg_freqs)
                else:
                    has_cycles = False

        # Compute causal effects
        estimates[curr_idx], refutations[curr_idx] = run_causal_inference_DML(
            curr_df,
            curr_sk_graph,
            test_significance=False,
            confidence_intervals=False,
            do_refute=False,
        )

    # Save results to file
    estimates_out_file = Path(ca_config.get_ce_paths(model, dataset))
    estimates_out_path = base_dir.joinpath(estimates_out_file).resolve()
    make_dir(estimates_out_path.parent)

    out_data = format_estimates(estimates, test_significance=False, std_error=False)
    save_json(out_data, estimates_out_path)
