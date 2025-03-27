import itertools

from utils.data_io import load_json


# == == == == == == == == == == == == == == == ==
# Loading data dictionary VG1800
# == == == == == == == == == == == == == == == ==
def load_vg1800_dict(data_path):
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


def compute_concepts_powerset(concepts, return_dict=False):
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


def compute_powerset(base_set):
    powerset = itertools.chain.from_iterable(
        itertools.combinations(base_set, r) for r in range(1, len(base_set) + 1)
    )
    return list(powerset)
