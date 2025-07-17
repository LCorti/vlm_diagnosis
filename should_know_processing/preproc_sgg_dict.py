import sys

from pathlib import Path

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.data_io import load_json, save_json

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    common_path = base_dir.joinpath("data/common")
    sgg_dict_path = common_path.joinpath("sgg_dicts.json")
    sgg_freqs_path = common_path.joinpath("sgg_freqs.json")
    sgg_dict = load_json(sgg_dict_path)

    pred_counts = sgg_dict["predicate_count"]
    concept_counts = sgg_dict["object_count"]
    total_pred_occ = sum(pred_counts.values())
    total_concept_occ = sum(concept_counts.values())

    sgg_freqs = {
        "predicate_counts": pred_counts,
        "concept_counts": concept_counts,
        "predicate_freqs": {},
        "concept_freqs": {},
    }
    sgg_freqs["predicate_freqs"] = {
        k: pred_counts[k] / total_pred_occ for k in pred_counts
    }
    sgg_freqs["concept_freqs"] = {
        k: concept_counts[k] / total_concept_occ for k in concept_counts
    }

    save_json(sgg_freqs, sgg_freqs_path)
