import numpy as np


def compute_ic(elem, count_dict, freq_dict):
    # If we have the frequency for a particular element, we use that.
    # Otherwise, we consider count(element) = 1 and then compute its frequency.
    if elem in freq_dict:
        ic = -np.log2(freq_dict[elem])
    else:
        ic = -np.log2(1 / sum(count_dict.values()))
    return ic


def compute_ic_relation(relation, sg_freqs):
    from_concept = relation[0]
    predicate = relation[1]
    to_concept = relation[2]
    ic_from_concept = compute_ic(
        from_concept, sg_freqs["concept_counts"], sg_freqs["concept_freqs"]
    )
    ic_predicate = compute_ic(
        predicate, sg_freqs["predicate_counts"], sg_freqs["predicate_freqs"]
    )
    ic_to_concept = compute_ic(
        to_concept, sg_freqs["concept_counts"], sg_freqs["concept_freqs"]
    )
    return ic_from_concept + ic_predicate + ic_to_concept


def compute_ic_predicates(graph, sg_freqs):
    pred_list = [graph[e[0]][e[1]]["label"] for e in graph.edges]
    all_pred_ics = []
    for pred in pred_list:
        all_pred_ics.append(
            compute_ic(pred, sg_freqs["predicate_counts"], sg_freqs["predicate_freqs"])
        )
    return np.mean(all_pred_ics)


def compute_ic_concepts(graph, sg_freqs):
    concept_list = list(graph.nodes)
    all_concept_ics = []
    for concept in concept_list:
        all_concept_ics.append(
            compute_ic(concept, sg_freqs["concept_counts"], sg_freqs["concept_freqs"])
        )
    return np.mean(all_concept_ics)
