import itertools
import networkx as nx

from utils.data_io import load_json


def load_vg1800_dict(data_path: str) -> dict:
    vg1800_data = load_json(data_path)

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
    vg1800_dict["predicate_freqs"] = {
        k: pred_counts[k] / total_pred_occ for k in pred_counts
    }
    vg1800_dict["concept_freqs"] = {
        k: concept_counts[k] / total_concept_occ for k in concept_counts
    }
    return vg1800_dict


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


def compute_powerset(base_set: list) -> list:
    powerset = itertools.chain.from_iterable(
        itertools.combinations(base_set, r) for r in range(1, len(base_set) + 1)
    )
    return list(powerset)


def stringify_nx_graph(graph: nx.DiGraph) -> list:
    rels = []
    for e in graph.edges:
        from_concept = graph.nodes[e[0]]["label"]
        to_concept = graph.nodes[e[1]]["label"]
        rel = graph[from_concept][to_concept]["label"]
        rels.append((from_concept, rel, to_concept))
    return rels


def stringify_graph_triple(triple: tuple, template: bool = True) -> str:
    from_concept, rel, to_concept = triple
    if template:
        return (
            f"* (Entity 1: {from_concept}, Relationship: {rel}, Entity 2: {to_concept})"
        )
    else:
        return f"{from_concept} {rel} {to_concept}"


def make_nx_graph(raw_sk_rels: dict) -> nx.DiGraph:
    nx_graph = nx.DiGraph()
    for raw_rel in raw_sk_rels:
        from_concept = raw_rel["from_concept"]["bb_label"]["bb_label_full"]
        to_concept = raw_rel["to_concept"]["bb_label"]["bb_label_full"]
        relation = raw_rel["rel_label"]["rel_label_text"]
        nx_graph.add_node(from_concept, label=from_concept)
        nx_graph.add_node(to_concept, label=to_concept)
        nx_graph.add_edge(from_concept, to_concept, label=relation)

    return nx_graph
