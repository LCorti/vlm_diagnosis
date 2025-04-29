import os
import sys

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.sk_config_loader import SKConfig
from utils.data_io import load_json, load_jsonl, save_json, save_jsonl


def parse_validation_data(val_data):
    # Check no annotation data is present
    outcome, _ = check_val_data(val_data)
    if not outcome:
        raise

    val_data_dict = {}
    for entry in val_data:
        question_id = entry["id_image"]
        if question_id not in val_data_dict:
            val_data_dict[question_id] = {}
        ann_id = entry["id_annotation"]
        if ann_id not in val_data_dict[question_id]:
            val_data_dict[question_id][ann_id] = {}

        # val_data_dict[question_id][ann_id]["triple_id"] = entry["id_triple"]
        val_data_dict[question_id][ann_id]["original"] = {
            "from_concept": entry["from_concept"].lower().strip(),
            "coords_from_concept": entry["coords_from_concept"],
            "relationship": entry["relationship"].lower().strip(),
            "to_concept": entry["to_concept"].lower().strip(),
            "coords_to_concept": entry["coords_to_concept"],
        }
        val_data_dict[question_id][ann_id]["answers"] = {
            "relevance": entry["answer_question"],
            "concepts_identified": entry["correctly_identified"],
            "concepts_related": entry["entities_in_relationship"],
            "correct_relationship": entry["correct_relationship"],
        }
        val_data_dict[question_id][ann_id]["crowd"] = {
            "new_from_concept": entry["new_label_from_concept"].lower().strip(),
            "new_coords_from_concept": entry["new_coords_from_concept"],
            "new_relationship": entry["new_relationship"].lower().strip(),
            "new_to_concept": entry["new_label_to_concept"].lower().strip(),
            "new_coords_to_concept": entry["new_coords_to_concept"],
        }
    return val_data_dict


def parse_annotation_data(ann_data):
    # Check no validation data is present
    outcome, _ = check_ann_data(ann_data)
    if not outcome:
        raise

    ann_data_dict = {}
    for entry in ann_data:
        question_id = entry["id_image"]
        if question_id not in ann_data_dict:
            ann_data_dict[question_id] = {}
        ann_id = entry["id"]
        if ann_id not in ann_data_dict[question_id]:
            ann_data_dict[question_id][ann_id] = {}

        ann_data_dict[question_id][ann_id]["crowd"] = {
            "from_concept": entry["entity1"].lower().strip(),
            "coords_from_concept": entry["coords_entity1"],
            "relationship": entry["relationship"].lower().strip(),
            "to_concept": entry["entity2"].lower().strip(),
            "coords_to_concept": entry["coords_entity2"],
            "id_json": entry["id_json"],
        }
    return ann_data_dict


def check_data(data, step):
    new_ann_value = 0 if step == "val" else 1
    print(f">> Checking data from {step} step:")
    check_entries = []
    for entry in data:
        # Check that all "new_user_annotation" = 0 and the consent was given
        if entry["new_user_annotation"] != new_ann_value or entry["optout"] == 1:
            check_entries.append(entry)
    if len(check_entries) == 0:
        print("... Everything ok.")
        return True, check_entries
    else:
        print("... Found the following problems.")
        no_consent_users = [e["id_user"] for e in check_entries if e["optout"] == 1]
        print(f"-- Users no consent: {no_consent_users}")
        problem_data = [
            e["id_annotation"]
            for e in check_entries
            if e["new_user_annotation"] == new_ann_value
        ]
        print(f"-- Entries from annotation: {problem_data}")
        return False, check_entries


def check_val_data(val_data):
    return check_data(val_data, "val")


def check_ann_data(ann_data):
    return check_data(ann_data, "ann")


def fix_bbox(bbox_to_fix):
    # bbox_to_fix: string formatted like "[start_x, start_y, width, height]"
    start_x, start_y, width, height = (
        int(elem) for elem in bbox_to_fix.replace("[", "").replace("]", "").split(",")
    )
    fixed_bbox = {}
    if width > 0:
        if height > 0:
            # w & h positive -> bbox drawn from top-left
            fixed_bbox["top_left_x"] = start_x
            fixed_bbox["top_left_y"] = start_y
            fixed_bbox["bottom_right_x"] = start_x + width
            fixed_bbox["bottom_right_y"] = start_y + height
            fixed_bbox["width"] = width
            fixed_bbox["height"] = height
        else:
            # w positive & h negative -> bbox drawn from bottom-left
            fixed_bbox["top_left_x"] = start_x
            fixed_bbox["top_left_y"] = start_y - abs(height)
            fixed_bbox["bottom_right_x"] = start_x + width
            fixed_bbox["bottom_right_y"] = start_y
            fixed_bbox["width"] = width
            fixed_bbox["height"] = abs(height)
    else:
        if height > 0:
            # w negative & h positive -> bbox drawn from top-right
            fixed_bbox["top_left_x"] = start_x - abs(width)
            fixed_bbox["top_left_y"] = start_y
            fixed_bbox["bottom_right_x"] = start_x
            fixed_bbox["bottom_right_y"] = start_y + height
            fixed_bbox["width"] = abs(width)
            fixed_bbox["height"] = height
        else:
            # w & h negative -> bbox drawn from bottom-right
            fixed_bbox["top_left_x"] = start_x - abs(width)
            fixed_bbox["top_left_y"] = start_y - abs(height)
            fixed_bbox["bottom_right_x"] = start_x
            fixed_bbox["bottom_right_y"] = start_y
            fixed_bbox["width"] = abs(width)
            fixed_bbox["height"] = abs(height)
    return fixed_bbox


def get_concept_id(concept_label, sgg_dict):
    if concept_label in sgg_dict["label_to_idx"]:
        return sgg_dict["label_to_idx"][concept_label]
    else:
        return "new"


def get_relation_id(relation_label, sgg_dict):
    if relation_label in sgg_dict["predicate_to_idx"]:
        return sgg_dict["predicate_to_idx"][relation_label]
    else:
        return "new"


def resolve_val_data(sg_data, parsed_crowd_val, sgg_dict, question):
    res_val_entry = {
        "question_id": question["question_id"],
        "img": question["img"],
        "img_path": question["img_path"],
        "relations": [],
    }
    stats = {"initial": 0, "after_val": 0}
    print(
        f"Checking q. {res_val_entry['question_id']} and image {res_val_entry['img']}"
    )
    # print(
    #     f"Looking for {res_val_entry['img']} among : {[sg['img_id'] for sg in sg_data]}"
    # )
    rels_to_check = next(
        sg["rel_clusters_unique"]
        for sg in sg_data
        if sg["img_id"] == res_val_entry["img"]
    )
    # Add +1 to match with ids from DB. Only added for this purpose.
    question_key = int(res_val_entry["question_id"]) + 1
    # print(f"Looking for {question_key} in {parsed_crowd_val.keys()}")
    all_crowd_val = list(parsed_crowd_val[question_key].values())
    # Update stats
    stats["initial"] = len(rels_to_check)

    for rel in rels_to_check:
        from_concept = rel["from_concept"]["bb_label"]["bb_label_text"]
        relationship = rel["rel_label"]["rel_label_text"]
        to_concept = rel["to_concept"]["bb_label"]["bb_label_text"]
        # Retrieve corresponding annotation
        # print(f"Searching for: {from_concept} - {relationship} - {to_concept}")
        crowd_rel = next(
            (
                rel_val
                for rel_val in all_crowd_val
                if rel_val["original"]["from_concept"] == from_concept
                and rel_val["original"]["relationship"] == relationship
                and rel_val["original"]["to_concept"] == to_concept
            ),
            None,
        )

        if not crowd_rel:
            print(f"Annotation not found! (img_ig: {res_val_entry['img']})")
            res_val_entry["relations"].append(rel)  # Consider them anyway
            # print(f"{rel}")
            continue

        if crowd_rel["answers"]["relevance"] == 1:
            # Fixing bboxes for 'from_concept' and 'to_concept'
            # from_concept
            fixed_bbox = fix_bbox(crowd_rel["crowd"]["new_coords_from_concept"])
            rel["from_concept"].update(fixed_bbox)
            # to_concept
            fixed_bbox = fix_bbox(crowd_rel["crowd"]["new_coords_to_concept"])
            rel["to_concept"].update(fixed_bbox)

            # Fixing concept labels if not identified correctly
            if crowd_rel["answers"]["concepts_identified"] == 0:
                # from_concept
                from_concept_idx = get_concept_id(
                    crowd_rel["crowd"]["new_from_concept"], sgg_dict
                )
                from_concept_label = crowd_rel["crowd"]["new_from_concept"]
                rel["from_concept"]["bb_label"]["bb_label_idx"] = from_concept_idx
                rel["from_concept"]["bb_label"]["bb_label_text"] = from_concept_label
                rel["from_concept"]["bb_label"]["bb_label_full"] = (
                    f"0-{from_concept_label}"
                )
                # to_concept
                to_concept_idx = get_concept_id(
                    crowd_rel["crowd"]["new_to_concept"], sgg_dict
                )
                to_concept_label = crowd_rel["crowd"]["new_to_concept"]
                rel["to_concept"]["bb_label"]["bb_label_idx"] = to_concept_idx
                rel["to_concept"]["bb_label"]["bb_label_text"] = to_concept_label
                rel["to_concept"]["bb_label"]["bb_label_full"] = f"0-{to_concept_label}"

            # Fixing relationships if incorrect label
            if crowd_rel["answers"]["correct_relationship"] == 0:
                rel_label_idx = get_relation_id(
                    crowd_rel["crowd"]["new_relationship"], sgg_dict
                )
                rel["rel_label"]["rel_label_idx"] = rel_label_idx
                rel["rel_label"]["rel_label_text"] = crowd_rel["crowd"][
                    "new_relationship"
                ]
            # Recompute rel_id after checks and corrections
            from_concept_idx = rel["from_concept"]["bb_label"]["bb_label_idx"]
            to_concept_idx = rel["to_concept"]["bb_label"]["bb_label_idx"]
            rel_idx = rel["rel_label"]["rel_label_idx"]
            rel["rel_id"] = f"{from_concept_idx}-{rel_idx}-{to_concept_idx}"
            # print(rel["rel_id"])

            # Add to list of resolved relations
            res_val_entry["relations"].append(rel)
        else:
            print("-- Relationship marked as irrelevant.")

    stats["after_val"] = len(res_val_entry["relations"])
    return res_val_entry, stats


def resolve_ann_data(val_entry, parsed_crowd_ann, sgg_dict, question, stats):
    res_ann_entry = val_entry
    # Add new relationships
    question_key = int(res_ann_entry["question_id"]) + 1
    all_crowd_ann = [elem["crowd"] for elem in parsed_crowd_ann[question_key].values()]
    for raw_crowd_ann in all_crowd_ann:
        # TODO: fix bb_label_full of from_concept and to_concept
        crowd_ann = {
            "from_concept": {
                "bb_label": {
                    "bb_label_idx": get_concept_id(
                        raw_crowd_ann["from_concept"], sgg_dict
                    ),
                    "bb_label_text": raw_crowd_ann["from_concept"],
                    "bb_label_full": raw_crowd_ann["from_concept"],
                }
            },
            "to_concept": {
                "bb_label": {
                    "bb_label_idx": get_concept_id(
                        raw_crowd_ann["to_concept"], sgg_dict
                    ),
                    "bb_label_text": raw_crowd_ann["to_concept"],
                    "bb_label_full": raw_crowd_ann["to_concept"],
                }
            },
            "rel_label": {
                "rel_label_idx": get_relation_id(
                    raw_crowd_ann["relationship"], sgg_dict
                ),
                "rel_label_text": raw_crowd_ann["relationship"],
            },
        }
        crowd_ann["from_concept"].update(fix_bbox(raw_crowd_ann["coords_from_concept"]))
        crowd_ann["to_concept"].update(fix_bbox(raw_crowd_ann["coords_to_concept"]))
        # Update rel_id field
        from_concept_idx = crowd_ann["from_concept"]["bb_label"]["bb_label_idx"]
        to_concept_idx = crowd_ann["to_concept"]["bb_label"]["bb_label_idx"]
        relationship_idx = crowd_ann["rel_label"]["rel_label_idx"]
        crowd_ann["rel_id"] = f"{from_concept_idx}-{relationship_idx}-{to_concept_idx}"
        res_ann_entry["relations"].append(crowd_ann)

    # Update stats by adding a new key
    stats["after_ann"] = len(res_ann_entry["relations"])
    return res_ann_entry, stats


def resolve_crowd_data(
    sg_data, parsed_crowd_val, parsed_crowd_ann, sgg_dict, ds_questions, stats
):
    resolved_data = []
    stats = {}
    for question in ds_questions:
        # Resolve validation data
        resolved_val_entry, curr_stats = resolve_val_data(
            sg_data, parsed_crowd_val, sgg_dict, question
        )
        # print(resolved_val_entry)
        # Resolve annotation data
        resolved_ann_entry, curr_stats = resolve_ann_data(
            resolved_val_entry, parsed_crowd_ann, sgg_dict, question, curr_stats
        )
        resolved_data.append(resolved_val_entry)
        stats[question["question_id"]] = curr_stats
        # Return the data annotation data has been merged
    return resolved_data, stats


if __name__ == "__main__":
    # Load SK config
    sk_config = SKConfig()
    ds_list = sk_config.get_sk_list()
    ds_list.remove("vqav2_holdout")
    # Load SGG dict to match concept and relation labels
    sgg_dict = load_json("../data/common/sgg_dicts.json")
    stats = {}

    for dataset in ds_list:
        print(f"Looking at {dataset}")
        # Get current sk paths
        sk_paths = sk_config.get_sk_paths(dataset)
        # Load questions for this dataset
        q_path = f"../data/datasets/{dataset}/q_crowd.json"
        ds_questions = load_json(q_path)
        # Add entry for statistics
        if dataset not in stats:
            stats[dataset] = {}

        for ds_class in sk_paths:
            print(f"- Looking at {ds_class}")
            # Add entry for statistics
            if ds_class not in stats[dataset]:
                stats[dataset][ds_class] = {"samples": {}, "summary": {}}

            # Load scene graphs showed to crowd workers for a given class
            base_dir = sk_paths[ds_class]["dir"]
            crowd_sg_file = sk_paths[ds_class]["sg_crowd"]
            crowd_sg = load_jsonl(f"../{base_dir}/{crowd_sg_file}")
            # Load data from validation step
            crowd_val_file = sk_paths[ds_class]["val_step"]
            crowd_val = load_jsonl(f"../{base_dir}/{crowd_val_file}")
            # Load data from annotation step
            crowd_ann_file = sk_paths[ds_class]["ann_exp"]
            crowd_ann = load_jsonl(f"../{base_dir}/{crowd_ann_file}")

            # Parse validation and annotation data
            parsed_crowd_val = parse_validation_data(crowd_val)
            parsed_crowd_ann = parse_annotation_data(crowd_ann)
            # print(parsed_crowd_val)
            # print("=" * 20)
            # print(parsed_crowd_ann)

            # Filter dataset questions for the given class
            qs_to_use = [q for q in ds_questions if q["class"] == ds_class]
            sg_final, stats[dataset][ds_class]["samples"] = resolve_crowd_data(
                crowd_sg,
                parsed_crowd_val,
                parsed_crowd_ann,
                sgg_dict,
                qs_to_use,
                stats[dataset][ds_class]["samples"],
            )

            stats[dataset][ds_class]["summary"]["initial"] = sum(
                entry["initial"]
                for entry in stats[dataset][ds_class]["samples"].values()
            )
            stats[dataset][ds_class]["summary"]["after_val"] = sum(
                entry["after_val"]
                for entry in stats[dataset][ds_class]["samples"].values()
            )
            stats[dataset][ds_class]["summary"]["after_ann"] = sum(
                entry["after_ann"]
                for entry in stats[dataset][ds_class]["samples"].values()
            )

            # Save to disk
            print("... Saving reconciled data to file...")
            sg_final_file = sk_paths[ds_class]["sk_final"]
            out_file_path = f"../{base_dir}/{sg_final_file}"
            save_jsonl(sg_final, out_file_path)
            print("... Saving stats to file...")
            out_file_path = f"../{base_dir}/stats.json"
            save_json(stats[dataset][ds_class], out_file_path)
            print("Saved.")
