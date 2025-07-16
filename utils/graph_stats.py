import numpy as np

from networkx import DiGraph
from typing import Tuple


# Computation of information content
def compute_ic(elem: str, count_dict: dict, freq_dict: dict) -> float:
    # If we have the frequency for a particular element, we use that.
    # Otherwise, we consider count(element) = 1 and then compute its frequency.
    if elem in freq_dict:
        ic = -np.log2(freq_dict[elem])
    else:
        ic = -np.log2(1 / sum(count_dict.values()))
    return ic


def compute_ic_preds(graph: DiGraph, count_dict, freq_dict) -> float:
    pred_list = [graph[e[0]][e[1]]["label"] for e in graph.edges]
    all_pred_ics = []
    for pred in pred_list:
        all_pred_ics.append(compute_ic(pred, count_dict, freq_dict))
    return np.mean(all_pred_ics)


def compute_ic_concepts(graph: DiGraph, count_dict: dict, freq_dict: dict) -> float:
    concept_list = list(graph.nodes)
    all_concept_ics = []
    for concept in concept_list:
        all_concept_ics.append(compute_ic(concept, count_dict, freq_dict))
    return np.mean(all_concept_ics)


def compute_ic_graph(graph: DiGraph, source_dict: dict) -> Tuple[float, float, float]:
    ic_concepts = compute_ic_concepts(
        graph, source_dict["concept_counts"], source_dict["concept_freqs"]
    )
    ic_preds = compute_ic_preds(
        graph, source_dict["predicate_counts"], source_dict["predicate_freqs"]
    )
    # For now, concepts and predicates have equal weights.
    weights = [1, 1]
    ic_graph = np.average([ic_preds, ic_concepts], weights=weights)
    return ic_concepts, ic_preds, ic_graph


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


# Computation of entropy
def compute_h(elem: str, count_dict: dict, freq_dict: dict) -> float:
    ic = compute_ic(elem, count_dict, freq_dict)
    if elem in freq_dict:
        h = freq_dict[elem] * ic
    else:
        h = 1 / sum(count_dict.values()) * ic
    return h


def compute_h_concepts(graph: DiGraph, count_dict: dict, freq_dict: dict) -> float:
    concept_list = list(graph.nodes)
    all_concepts_h = []
    for concept in concept_list:
        all_concepts_h.append(compute_h(concept, count_dict, freq_dict))
    # Apply Miller-Madow correction
    correction_factor = (len(set(concept_list)) - 1) / (2 * len(concept_list))
    norm_h = (np.sum(all_concepts_h) + correction_factor) / np.log2(len(count_dict))
    return norm_h


def compute_h_preds(graph: DiGraph, count_dict: dict, freq_dict: dict) -> float:
    pred_list = [graph[e[0]][e[1]]["label"] for e in graph.edges]
    all_pred_h = []
    for pred in pred_list:
        all_pred_h.append(compute_h(pred, count_dict, freq_dict))
    # Apply Miller-Madow correction
    correction_factor = (len(set(pred_list)) - 1) / (2 * len(pred_list))
    norm_h = (np.sum(all_pred_h) + correction_factor) / np.log2(len(count_dict))
    return norm_h


def compute_h_graph(graph: DiGraph, source_dict: dict) -> float:
    h_concepts = compute_h_concepts(
        graph, source_dict["concept_counts"], source_dict["concept_freqs"]
    )
    h_preds = compute_h_preds(
        graph, source_dict["predicate_counts"], source_dict["predicate_freqs"]
    )
    # For now, concepts and predicates have equal weights.
    weights = [1, 1]
    h_graph = np.average([h_concepts, h_preds], weights=weights)
    return h_concepts, h_preds, h_graph


def compute_sample_stats(sample_graph: DiGraph, source_dict: dict) -> dict:
    stats = {
        "count_concepts": sample_graph.number_of_nodes(),
        "count_preds": sample_graph.number_of_edges(),
    }
    # Information content
    ic = compute_ic_graph(sample_graph, source_dict)
    stats["ic_concepts"] = ic[0]
    stats["ic_preds"] = ic[1]
    stats["ic_graph"] = ic[2]
    # Entropy
    h = compute_h_graph(sample_graph, source_dict)
    stats["h_concepts"] = h[0]
    stats["h_preds"] = h[1]
    stats["h_graph"] = h[2]
    return stats


def compute_stats(graph_dict: dict, source_dict: dict) -> Tuple[dict, dict]:
    # Compute stats for each sample
    sample_stats = {}
    for idx, graph in graph_dict.items():
        if idx not in sample_stats:
            sample_stats[idx] = {}
        sample_stats[idx] = compute_sample_stats(graph, source_dict)

    # end up with a dict{idx: sample_stats}

    # sample_stats and all the other used to be a list []

    # Compute dataset statistic
    # Concepts
    count_conc_list = [v["count_concepts"] for v in sample_stats.values()]
    ic_concepts = [v["ic_concepts"] for v in sample_stats.values()]
    h_concepts = [v["h_concepts"] for v in sample_stats.values()]

    # Predicates
    count_preds_list = [v["count_preds"] for v in sample_stats.values()]
    ic_preds = [v["ic_preds"] for v in sample_stats.values()]
    h_preds = [v["h_preds"] for v in sample_stats.values()]

    # Graphs
    ic_graphs = [v["ic_graph"] for v in sample_stats.values()]
    h_graphs = [v["h_graph"] for v in sample_stats.values()]

    stats_summary = {
        "count_concepts": np.sum(count_conc_list),
        "avg_concepts": np.mean(count_conc_list),
        "stdev_concepts": np.std(count_conc_list) if len(count_conc_list) > 1 else 0,
        "avg_ic_concepts": np.mean(ic_concepts),
        "stdev_ic_concepts": np.std(ic_concepts) if len(ic_concepts) > 1 else 0,
        "avg_h_concepts": np.mean(h_concepts),
        "std_h_concepts": np.std(h_concepts) if len(h_concepts) > 1 else 0,
        "count_preds": np.sum(count_preds_list),
        "avg_preds": np.mean(count_preds_list),
        "stdev_preds": np.std(count_preds_list) if len(count_preds_list) > 1 else 0,
        "avg_ic_preds": np.mean(ic_preds),
        "stdev_ic_preds": np.std(ic_preds) if len(ic_preds) > 1 else 0,
        "avg_h_preds": np.mean(h_preds),
        "std_h_preds": np.std(h_preds) if len(h_preds) > 1 else 0,
        "avg_ic_graph": np.mean(ic_graphs),
        "stdev_ic_graph": np.std(ic_graphs) if len(ic_graphs) > 1 else 0,
        "avg_h_graph": np.mean(h_graphs),
        "std_h_graph": np.std(h_graphs) if len(h_graphs) > 1 else 0,
    }
    return sample_stats, stats_summary
