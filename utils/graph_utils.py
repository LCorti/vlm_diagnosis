import itertools
import re
from pathlib import Path

import networkx as nx
import numpy as np

from utils.data_io import load_json
from utils.graph_stats import compute_ic


# Loading the dictionary used in IETrans
# https://link.springer.com/chapter/10.1007/978-3-031-19812-0_24
def load_vg1800_dict() -> dict:
    vg1800_path = Path(__file__).parent.parent.joinpath(
        "data", "common", "sgg_dicts.json"
    )
    vg1800_data = load_json(vg1800_path)

    pred_counts = vg1800_data["predicate_count"]
    concept_counts = vg1800_data["object_count"]
    total_pred_occ = sum(pred_counts.values())
    total_concept_occ = sum(concept_counts.values())
    vg1800_dict = {
        "predicate_counts": pred_counts,
        "concept_counts": concept_counts,
        "predicate_freqs": {},
        "concept_freqs": {},
    }
    # Pre-compute frequencies for descriptive statistics
    vg1800_dict["predicate_freqs"] = {
        k: pred_counts[k] / total_pred_occ for k in pred_counts
    }
    vg1800_dict["concept_freqs"] = {
        k: concept_counts[k] / total_concept_occ for k in concept_counts
    }
    return vg1800_dict


# Powerset
def compute_powerset(base_set: list) -> list:
    powerset = itertools.chain.from_iterable(
        itertools.combinations(base_set, r) for r in range(1, len(base_set) + 1)
    )
    return list(powerset)


def compute_concepts_powerset(concepts: list, return_dict: bool = False) -> dict | list:
    powerset_list = compute_powerset(concepts)
    if not return_dict:
        return powerset_list

    powerset_dict = {}
    for subset in powerset_list:
        if len(subset) not in powerset_dict:
            powerset_dict[len(subset)] = [subset]
        else:
            powerset_dict[len(subset)].append(subset)
    return powerset_dict


# Return string version of a graph (SK or RK)
def nx_graph_to_str(graph: nx.DiGraph, template: bool = False) -> list:
    rels = []
    for e in graph.edges:
        from_concept = graph.nodes[e[0]]["label"]
        to_concept = graph.nodes[e[1]]["label"]
        rel = graph[from_concept][to_concept]["label"]
        rels.append(nx_triple_to_str((from_concept, rel, to_concept), template))
    return rels


def nx_triple_to_str(triple: tuple, template: bool = True) -> str:
    from_concept, rel, to_concept = triple
    if template:
        return (
            f"* (Entity 1: {from_concept}, Relationship: {rel}, Entity 2: {to_concept})"
        )
    else:
        return f"{from_concept} {rel} {to_concept}"


def triple_str_to_dict(triple: str) -> dict:
    pattern = r"\* \(Entity 1: (?P<from_concept>.+), Relationship: (?P<relationship>.+), Entity 2: (?P<to_concept>.+)\)"
    triple_match = re.search(pattern, triple)
    triple_match = {k: v.strip() for k, v in triple_match.groupdict().items()}
    return triple_match


# Return NX graph object given strings
def add_to_graph(nx_graph: nx.DiGraph, rel: dict) -> nx.DiGraph:
    from_concept = rel["from_concept"].lower().strip()
    rel_label = rel["relationship"].lower().strip()
    to_concept = rel["to_concept"].lower().strip()
    if from_concept not in nx_graph:
        nx_graph.add_node(from_concept, label=from_concept)
    if to_concept not in nx_graph:
        nx_graph.add_node(to_concept, label=to_concept)
    nx_graph.add_edge(from_concept, to_concept, label=rel_label)
    return nx_graph


def sk_to_nx(raw_sk_rels: list) -> nx.DiGraph:
    # Prepare raw_sk_rels before making nx graph
    nx_graph = nx.DiGraph()
    for sk in raw_sk_rels:
        sk_triple = {
            "from_concept": sk["from_concept"]["bb_label"]["bb_label_text"],
            "relationship": sk["rel_label"]["rel_label_text"],
            "to_concept": sk["to_concept"]["bb_label"]["bb_label_text"],
        }
        nx_graph = add_to_graph(nx_graph, sk_triple)
    return nx_graph


def rk_to_nx(raw_rk_rels: dict, return_full=False) -> nx.DiGraph:
    nx_graph = nx.DiGraph()
    for rk in raw_rk_rels:
        rk_triple = {
            "relationship": rk["rel_label"]["rel_label_text"],
        }
        if return_full:
            rk_triple["from_concept"] = rk["from_concept"]["bb_label"]["bb_label_full"]
            rk_triple["to_concept"] = rk["to_concept"]["bb_label"]["bb_label_full"]
        else:
            rk_triple["from_concept"] = rk["from_concept"]["bb_label"]["bb_label_text"]
            rk_triple["to_concept"] = rk["to_concept"]["bb_label"]["bb_label_text"]
        nx_graph = add_to_graph(nx_graph, rk_triple)

    # Removing self loops for causal analysis
    self_loops = list(nx.selfloop_edges(nx_graph))
    if self_loops:
        print(f"- Found self loops: {self_loops}.")
        nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    else:
        print("- No self-loops found.")

    # Handle cycles for causal analysis
    nx_graph = handle_cycles(nx_graph, load_vg1800_dict())
    return nx_graph


def simple_rk_to_nx(rk_rels: list) -> nx.DiGraph:
    nx_graph = nx.DiGraph()
    for rk in rk_rels:
        nx_graph = add_to_graph(nx_graph, rk)
    return nx_graph


def expand_cycles(cycles: list) -> list:
    new_cycles = []
    for c in cycles:
        if len(c) == 2:
            """
            Base case: we have a cycle between nodes u and v.
            To access the corresponding edges, we need [u,v] and [v,u].
            """
            new_cycles.append([c, c[::-1]])
        else:
            """
            For larger cycles, we need to consider a sliding window of 2 over the list.
            For instance, if the identified cycle looks like [u,v,k], then we need
            to make the following edges: [u,v], [v,k], [k,u]
            """
            subset = []
            for i in range(len(c)):
                subset.append([c[i], c[(i + 1) % len(c)]])
            new_cycles.append(subset)
    return new_cycles


def handle_cycles(graph: nx.DiGraph, sgg_dicts: dict) -> nx.DiGraph:
    # 1. Get first set of expanded cycles
    exp_cycles = expand_cycles(list(nx.simple_cycles(graph)))
    if not exp_cycles:
        print("- No cycles found.")
        return graph

    print(f"- Identified cycles: {exp_cycles}")
    while exp_cycles:
        # 2. Group these cycles based on their size
        idx_to_extract = 0
        exp_cycles_dict = {}
        for c in exp_cycles:
            if len(c) not in exp_cycles_dict:
                exp_cycles_dict[len(c)] = []
            exp_cycles_dict[len(c)].append(c)
        # 3. Consider smallest size and pop one
        min_size = min(list(exp_cycles_dict.keys()))
        curr_cycle = exp_cycles_dict[min_size][idx_to_extract]
        least_inf = {"u": "", "v": "", "ic": np.inf}
        for edge in curr_cycle:
            # 4. Get label for edges in that cycle
            label = graph.edges[edge[0], edge[1]]["label"]
            # 5. Compute information content of label
            ic = compute_ic(
                label, sgg_dicts["predicate_counts"], sgg_dicts["predicate_freqs"]
            )
            if ic < least_inf["ic"]:
                least_inf.update({"u": edge[0], "v": edge[1], "ic": ic})
        # 6. Remove edge corresponding to the min. information content
        graph.remove_edge(least_inf["u"], least_inf["v"])
        # 7. Re-compute expanded cycles within the current graph
        exp_cycles = expand_cycles(list(nx.simple_cycles(graph)))
    return graph
