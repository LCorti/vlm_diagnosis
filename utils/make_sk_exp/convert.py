# Script to convert arrays in JSON files into JSON-lines files for SK annotation.

import json

from pathlib import Path


def load_vg1800_dict() -> dict:
    vg1800_data = load_json(
        Path(__file__).parent.parent.joinpath("data", "common", "sgg_dicts.json")
    )
    return vg1800_data


def load_json(file_path: str | Path) -> dict:
    with open(file_path, "r") as fp:
        json_data = json.load(fp)
    return json_data


def save_jsonl(data: list[dict], file_path: str) -> None:
    with open(file_path, "w") as fp:
        fp.write("\n".join(map(json.dumps, data)))


def get_concept_id(concept_label: str, sgg_dict: dict) -> int | str:
    if concept_label in sgg_dict["label_to_idx"]:
        return sgg_dict["label_to_idx"][concept_label]
    else:
        return "new"


def get_relation_id(relation_label: str, sgg_dict: dict) -> int | str:
    if relation_label in sgg_dict["predicate_to_idx"]:
        return sgg_dict["predicate_to_idx"][relation_label]
    else:
        return "new"


if __name__ == "__main__":
    in_file = "sk_final_exp.json"
    out_file = "sk_exp.jsonl"
    sgg_dict = load_vg1800_dict()

    structure = {
        "llava-bench": ["main"],
        "mmbench": ["image_scene", "image_topic"],
        "seed": ["scene_understanding", "visual_reasoning"],
        "vqav2": ["how_many_people_are", "what_is_the_person"],
    }

    for ds in structure:
        for ds_class in structure[ds]:
            # Load data
            data = load_json(Path(ds, ds_class, in_file))
            # Recompute concept and predicate IDs
            for q_data in data:
                for rel in q_data["relations"]:
                    # From concept
                    from_concept_text = rel["from_concept"]["bb_label"]["bb_label_text"]
                    from_concept_idx = get_concept_id(from_concept_text, sgg_dict)
                    from_concept_full = f"{from_concept_idx}-{from_concept_text}"
                    rel["from_concept"]["bb_label"]["bb_label_idx"] = from_concept_idx
                    rel["from_concept"]["bb_label"]["bb_label_full"] = from_concept_full

                    # Predicate
                    rel["rel_label"]["rel_label_idx"] = get_relation_id(
                        rel["rel_label"]["rel_label_text"], sgg_dict
                    )

                    # To concept
                    to_concept_text = rel["to_concept"]["bb_label"]["bb_label_text"]
                    to_concept_idx = get_concept_id(to_concept_text, sgg_dict)
                    to_concept_full = f"{to_concept_idx}-{to_concept_text}"
                    rel["to_concept"]["bb_label"]["bb_label_idx"] = to_concept_idx
                    rel["to_concept"]["bb_label"]["bb_label_full"] = to_concept_full

            # Save data
            save_jsonl(
                data,
                Path(__file__).parent.parent.parent.joinpath(
                    "data", "should_know", ds, ds_class, out_file
                ),
            )
