import itertools

from networkx import DiGraph
from pathlib import Path
from utils.data_io import load_json


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
def nx_graph_to_str(graph: DiGraph, template: bool = False) -> list:
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


# Return NX graph object given strings
def create_nx_graph(rel_list: list) -> DiGraph:
    nx_graph = DiGraph()
    for rel in rel_list:
        from_concept = rel["from_concept"].lower().strip()
        rel_label = rel["relationship"].lower().strip()
        to_concept = rel["to_concept"].lower().strip()
        if from_concept not in nx_graph:
            nx_graph.add_node(from_concept, label=from_concept)
        if to_concept not in nx_graph:
            nx_graph.add_node(to_concept, label=to_concept)
        nx_graph.add_edge(from_concept, to_concept, label=rel_label)
    return nx_graph


def sk_to_nx(raw_sk_rels: dict) -> DiGraph:
    # Prepare raw_sk_rels before making nx graph
    sk_list = [
        {
            "from_concept": sk["from_concept"]["bb_label"]["bb_label_text"],
            "relationship": sk["rel_label"]["rel_label_text"],
            "to_concept": sk["to_concept"]["bb_label"]["bb_label_text"],
        }
        for sk in raw_sk_rels
    ]
    return create_nx_graph(sk_list)


def rk_to_nx(rk_rels: dict) -> DiGraph:
    return create_nx_graph(rk_rels)
