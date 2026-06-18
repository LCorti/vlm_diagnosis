from typing import Tuple

import numpy as np
from networkx import DiGraph
from sklearn.preprocessing import MinMaxScaler

from utils.data_io import np_encoder


# Computation of information content
def compute_ic(elem: str, count_dict: dict, freq_dict: dict) -> float:
    # If we have the frequency for a particular element, we use that.
    # Otherwise, we consider count(element) = 1 and then compute its frequency.
    if elem in freq_dict:
        ic = -np.log2(freq_dict[elem])
    else:
        ic = -np.log2(1 / sum(count_dict.values()))
    return ic


def compute_ic_preds(graph: DiGraph, count_dict: dict, freq_dict: dict) -> np.ndarray:
    preds = [graph[e[0]][e[1]]["label"] for e in graph.edges]
    preds_ic = np.array([compute_ic(p, count_dict, freq_dict) for p in preds]).reshape(
        -1, 1
    )
    return preds_ic


def compute_ic_concepts(
    graph: DiGraph, count_dict: dict, freq_dict: dict
) -> np.ndarray:
    concepts = list(graph.nodes)
    concepts_ic = np.array(
        [compute_ic(c, count_dict, freq_dict) for c in concepts]
    ).reshape(-1, 1)
    return concepts_ic


def compute_ic_graph(
    graph: DiGraph, source_dict: dict
) -> Tuple[np.ndarray, np.ndarray]:
    ic_concepts = compute_ic_concepts(
        graph, source_dict["concept_counts"], source_dict["concept_freqs"]
    )
    ic_preds = compute_ic_preds(
        graph, source_dict["predicate_counts"], source_dict["predicate_freqs"]
    )
    return ic_concepts, ic_preds


def compute_ic_ds(graphs: dict[str, DiGraph], source_dict: dict) -> dict:
    ds_ic = {}
    for g_idx, graph in graphs.items():
        ds_ic[g_idx] = {
            "concepts": compute_ic_concepts(
                graph, source_dict["concept_counts"], source_dict["concept_freqs"]
            ),
            "preds": compute_ic_preds(
                graph, source_dict["predicate_counts"], source_dict["predicate_freqs"]
            ),
        }

    # Fit scalers
    concept_scaler = MinMaxScaler().fit(
        np.concatenate([g["concepts"] for g in ds_ic.values()], axis=0)
    )
    pred_scaler = MinMaxScaler().fit(
        np.concatenate([g["preds"] for g in ds_ic.values()], axis=0)
    )
    for g_idx in ds_ic:
        vals = np.concatenate(
            [
                concept_scaler.transform(ds_ic[g_idx]["concepts"]),
                pred_scaler.transform(ds_ic[g_idx]["preds"]),
            ],
            axis=0,
        )
        ds_ic[g_idx] = np_encoder(np.mean(vals))

    return ds_ic


def compute_ic_relation(relation: tuple, sg_dict: dict) -> float:
    from_concept = relation[0]
    predicate = relation[1]
    to_concept = relation[2]
    ic_from_concept = compute_ic(
        from_concept, sg_dict["count_concepts"], sg_dict["freq_concepts"]
    )
    ic_predicate = compute_ic(predicate, sg_dict["count_preds"], sg_dict["freq_preds"])
    ic_to_concept = compute_ic(
        to_concept, sg_dict["count_concepts"], sg_dict["concept_freqs"]
    )
    return ic_from_concept + ic_predicate + ic_to_concept


def compute_sample_stats(sample_graph: DiGraph, source_dict: dict) -> dict:
    stats = {
        "count_concepts": sample_graph.number_of_nodes(),
        "count_preds": sample_graph.number_of_edges(),
    }
    # Information content (not scaled)
    stats["ic_concepts"], stats["ic_preds"] = compute_ic_graph(
        sample_graph, source_dict
    )
    return stats


def compute_stats(graph_dict: dict, source_dict: dict) -> Tuple[dict, dict]:
    # Compute stats for each sample
    sample_stats = {}
    for g_idx, graph in graph_dict.items():
        sample_stats[g_idx] = compute_sample_stats(graph, source_dict)
    # end up with a dict{g_idx: sample_stats}

    # Fit scalers for information content
    concept_scaler = MinMaxScaler().fit(
        np.concatenate([g["ic_concepts"] for g in sample_stats.values()], axis=0)
    )
    pred_scaler = MinMaxScaler().fit(
        np.concatenate([g["ic_preds"] for g in sample_stats.values()], axis=0)
    )

    # Compute IC for each sample
    for g_idx in sample_stats.keys():
        vals = np.concatenate(
            [
                concept_scaler.transform(sample_stats[g_idx]["ic_concepts"]),
                pred_scaler.transform(sample_stats[g_idx]["ic_preds"]),
            ],
            axis=0,
        )
        sample_stats[g_idx]["ic_graph"] = np_encoder(np.mean(vals))
        # Also fix formatting such that it is JSON-friendly
        sample_stats[g_idx]["ic_concepts"] = (
            sample_stats[g_idx]["ic_concepts"].flatten().tolist()
        )
        sample_stats[g_idx]["ic_preds"] = (
            sample_stats[g_idx]["ic_preds"].flatten().tolist()
        )

    # Compute dataset statistic
    # Concepts
    count_conc_list = [v["count_concepts"] for v in sample_stats.values()]
    # Predicates
    count_preds_list = [v["count_preds"] for v in sample_stats.values()]
    # Graphs
    ic_graphs = [v["ic_graph"] for v in sample_stats.values()]

    stats_summary = {
        "count_concepts": np.sum(count_conc_list),
        "avg_concepts": np.mean(count_conc_list),
        "std_concepts": np.std(count_conc_list) if len(count_conc_list) > 1 else 0,
        "count_preds": np.sum(count_preds_list),
        "avg_preds": np.mean(count_preds_list),
        "std_preds": np.std(count_preds_list) if len(count_preds_list) > 1 else 0,
        "avg_ic": np.mean(ic_graphs),
        "std_ic": np.std(ic_graphs) if len(ic_graphs) > 1 else 0,
    }

    # Fix typing to save to JSON for the summary
    for k, v in stats_summary.items():
        stats_summary[k] = np_encoder(v)
    return sample_stats, stats_summary
