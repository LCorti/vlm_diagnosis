import cv2
import numpy as np
import os
import sys
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer
from ultralytics import YOLOE

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.rk_config_loader import RKConfig
from config_loaders.sk_config_loader import SKConfig
from utils.graph_utils import stringify_graph_triple
from utils.data_io import make_dir, load_json, load_jsonl, save_json


def free_gpu(var):
    del var
    torch.cuda.empty_cache()


# Compute semantic similarity
def compute_similarity(model, text_a, text_b):
    text_a_emb = torch.FloatTensor(
        model.encode(text_a, normalize_embeddings=True)
    ).unsqueeze(0)
    text_b_emb = torch.FloatTensor(
        model.encode(text_b, normalize_embeddings=True)
    ).unsqueeze(0)
    return model.similarity_pairwise(text_a_emb, text_b_emb)


def get_similar_concept(emb_model, concept_to_match, res_dicts):
    max_sim = -1
    curr_concept = {}
    for res in res_dicts:
        sim = compute_similarity(
            emb_model,
            concept_to_match,
            res["box"]["bb_label"]["bb_label_text"],
        )
        # print(f"Sim val: {sim}")
        # Keep track of max sim and corresponding data
        if sim > 0 and sim > max_sim:
            max_sim = sim
            curr_concept = res

    return curr_concept


def load_merge_sk_data(sk_paths):
    sk_data = []
    for ds_class in sk_paths:
        curr_base_dir = sk_paths[ds_class]["dir"]
        curr_sk_file = sk_paths[ds_class]["sk_final"]
        curr_sk_path = Path("..", curr_base_dir, curr_sk_file)
        sk_data.extend(load_jsonl(curr_sk_path))
    return sk_data


def get_sk_rels(sk):
    rels = []
    for rel in sk["relations"]:
        rels.append(
            {
                "from_concept": rel["from_concept"]["bb_label"]["bb_label_text"],
                "relationship": rel["rel_label"]["rel_label_text"],
                "to_concept": rel["to_concept"]["bb_label"]["bb_label_text"],
            }
        )
    return rels


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


def make_pred_dict(box, label, conf):
    pred_dict = {
        "conf": conf,
        "box": {
            "top_left_x": round(box[0]),
            "top_left_y": round(box[1]),
            "bottom_right_x": round(box[2]),
            "bottom_right_y": round(box[3]),
            "bb_label": {
                "bb_label_idx": 0,  # Updated later
                "bb_label_text": label,
                "bb_label_full": label,  # TODO: fix
            },
        },
    }
    pred_dict["box"]["width"] = np.abs(
        pred_dict["box"]["top_left_x"] - pred_dict["box"]["bottom_right_x"]
    ).item()
    pred_dict["box"]["height"] = np.abs(
        pred_dict["box"]["top_left_y"] - pred_dict["box"]["bottom_right_y"]
    ).item()
    return pred_dict


def parse_yoloe_res(yoloe_res, prompt_free=False):
    label_dict = yoloe_res.names
    all_pred_idx = yoloe_res.boxes.cls.detach().cpu().numpy()
    all_conf = yoloe_res.boxes.conf.detach().cpu().numpy()
    all_boxes = yoloe_res.boxes.xyxy.detach().cpu().numpy()
    predictions = []

    if prompt_free:
        for pred_idx, conf, box in zip(all_pred_idx, all_conf, all_boxes):
            curr_pred = make_pred_dict(box, label_dict[pred_idx], np.array(conf).item())
            # Fix bb_label_idx field
            curr_pred["box"]["bb_label"]["bb_label_idx"] = get_concept_id(
                curr_pred["box"]["bb_label"]["bb_label_text"], sgg_dict
            )
            predictions.append(curr_pred)
    else:
        checked_pred_idx = []
        for pred_idx in all_pred_idx:
            if pred_idx in checked_pred_idx:
                continue

            # get positions of data for current class
            curr_ids = [e[0] for e in enumerate(all_pred_idx) if e[1] == pred_idx]
            # get confidence values for current class
            curr_conf = all_conf[curr_ids]
            # compute the max confidence values for current class
            max_conf = np.max(curr_conf)
            # get index of max confidence in original array
            max_conf_idx = np.where(all_conf == max_conf)[0][0]
            # extract bboxes data corresponding to max confidence
            box = all_boxes[max_conf_idx]
            curr_pred = make_pred_dict(
                box, label_dict[pred_idx], np.array(max_conf).item()
            )
            # update lists
            checked_pred_idx.append(pred_idx)
            predictions.append(curr_pred)
    return predictions


if __name__ == "__main__":
    PROMPT_VERSION = 4

    # Loading config files
    sk_config = SKConfig()
    rk_config = RKConfig()
    model_list = rk_config.get_model_list() * 4
    ds_list = sk_config.get_sk_list()
    ds_list.remove("vqav2_holdout")
    ds_list = ds_list * 4
    model_list.sort()

    # Load sgg dict
    sgg_dict = load_json(Path("..", "data", "common", "sgg_dicts.json"))

    # Variables to check what is happening
    counts = {}

    # Load sentence-transformers for computing embeddings
    emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    for model, ds in zip(model_list, ds_list):
        print(f"{model} -- {ds}")
        # Initialise counters
        if model not in counts:
            counts[model] = {}
        if ds not in counts[model]:
            counts[model][ds] = {
                "initial": 0,
                "after_exact_match": 0,
                "after_sim_match": 0,
                "after_cv_match": 0,
            }

        # Load SK final
        sk_final_paths = sk_config.get_sk_paths(ds)
        # Merge batches from classes
        sk_final = load_merge_sk_data(sk_final_paths)

        # Load parsed RK
        rk_parsed_path = rk_config.get_parsed_rk_paths(model, ds)
        rk_parsed_path = rk_parsed_path.format(PROMPT_VERSION)
        rk_parsed_path = Path("..", rk_parsed_path)
        rk_parsed = load_jsonl(rk_parsed_path)

        matched_rk = {}

        for rk in rk_parsed:
            print(f">> Checking question {rk['question_id']}")
            if len(rk["triple_objs"]) == 0:
                print(f"No RKs found for question {rk['question_id']}")
                continue

            # Get SK data for current question
            curr_sk = next(
                (sk for sk in sk_final if sk["question_id"] == rk["question_id"]), None
            )
            if not curr_sk:
                print(f"Big error: SK missing for question {rk['question_id']}")
                break

            # Load image for 3rd matching step
            img_path = Path("..", curr_sk["img_path"])
            img = cv2.imread(img_path)

            # Make SK into a format that can be easily compared with SK
            # RK format: {'from_concept': ..., 'relationship': ..., 'to_concept': ...}
            curr_sk_rels = get_sk_rels(curr_sk)
            # Go relation by relation and try to match
            curr_rk_matches = []
            for rk_rel in rk["triple_objs"]:
                print(f">> >> Checking relation: {rk_rel}")
                # Update count
                counts[model][ds]["initial"] += 1

                # First try: exact match -- look for the RK in the list of SK
                # The matching is done at the triple-level to void the "wrong" concepts
                # to be matched by mistake
                match_sk_rel = next(
                    (
                        sk_rel
                        for sk_rel in curr_sk_rels
                        if sk_rel["from_concept"] == rk_rel["from_concept"]
                        and sk_rel["relationship"] == rk_rel["relationship"]
                        and sk_rel["to_concept"] == rk_rel["to_concept"]
                    ),
                    None,
                )
                # If a match is found, just use that (and update count)
                # and go to next iteration
                if match_sk_rel:
                    counts[model][ds]["after_exact_match"] += 1
                    curr_rk_matches.append(match_sk_rel)
                    continue

                # Second try: similarity current RK rel vs. all SK rels
                # Run similarity but only keep the largest one
                max_sim = -1
                sk_max_sim = {}
                rk_rel_text = stringify_graph_triple(rk_rel.values(), template=False)
                for sk_rel in curr_sk_rels:
                    sk_rel_text = stringify_graph_triple(
                        sk_rel.values(), template=False
                    )
                    sim = compute_similarity(emb_model, rk_rel_text, sk_rel_text)
                    if sim > max_sim:
                        max_sim = sim
                        sk_max_sim = sk_rel
                # Check if the matched triple is similar enough
                if max_sim > 0.5:
                    counts[model][ds]["after_sim_match"] += 1
                    curr_rk_matches.append(sk_max_sim)
                    continue

                # Third try: run object detection on the fly to get positions
                yoloe = YOLOE("yoloe-11l-seg.pt")  # this version expects a prompt
                yoloe_pf = YOLOE("yoloe-11l-seg-pf.pt")
                from_concept = rk_rel["from_concept"]  # this will be class 0
                to_concept = rk_rel["to_concept"]  # this will be class 1
                print(f"CV matching for {from_concept} - {to_concept}")
                prompt = [from_concept, to_concept]
                yoloe.set_classes(prompt, yoloe.get_text_pe(prompt))
                yoloe_res = yoloe.predict(img)[0]
                preds = yoloe_res.boxes.cls  # .detach().cpu().tolist()
                # print(preds)
                if len(preds) > 0:
                    print(f"Found {len(preds)} objects!")
                    res_dicts = parse_yoloe_res(yoloe_res, prompt_free=False)
                    free_gpu(yoloe_res)
                    print(res_dicts)
                    # Since we have from_concept and to_concept, merge data
                    matched_rk = {
                        "rel_label": {
                            "rel_label_idx": get_relation_id(
                                rk_rel["relationship"], sgg_dict
                            ),
                            "rel_label_text": rk_rel["relationship"],
                        }
                    }
                    for pred_idx, res in zip(preds, res_dicts):
                        if int(pred_idx.item()) == 0:
                            matched_rk["from_concept"] = next(
                                res["box"]
                                for res in res_dicts
                                if res["box"]["bb_label"]["bb_label_text"]
                                == from_concept
                            )
                        elif int(pred_idx.item()) == 1:
                            matched_rk["to_concept"] = next(
                                res["box"]
                                for res in res_dicts
                                if res["box"]["bb_label"]["bb_label_text"] == to_concept
                            )
                        else:
                            print(f"Unexpected pred_idx in results: {pred_idx}")

                    print(matched_rk)

                    # If only 1 concept is found, run sem similarity to try to match it.
                    if "from_concept" not in matched_rk:
                        sim_concept = get_similar_concept(
                            emb_model, from_concept, res_dicts
                        )
                        if "box" in sim_concept:
                            matched_rk["from_concept"] = sim_concept["box"]
                            matched_rk["from_concept"]["bb_label"]["bb_label_idx"] = (
                                get_concept_id(from_concept, sgg_dict)
                            )
                            matched_rk["from_concept"]["bb_label"]["bb_label_text"] = (
                                from_concept
                            )
                            matched_rk["from_concept"]["bb_label"]["bb_label_full"] = (
                                from_concept  # TODO: fix
                            )
                        else:
                            matched_rk["from_concept"] = rk_rel
                            matched_rk["from_concept"]["missing"] = True

                    if "to_concept" not in matched_rk:
                        sim_concept = get_similar_concept(
                            emb_model, to_concept, res_dicts
                        )
                        if "box" in sim_concept:
                            matched_rk["to_concept"] = sim_concept["box"]
                            matched_rk["to_concept"]["bb_label"]["bb_label_idx"] = (
                                get_concept_id(to_concept, sgg_dict)
                            )
                            matched_rk["to_concept"]["bb_label"]["bb_label_text"] = (
                                to_concept
                            )
                            matched_rk["to_concept"]["bb_label"]["bb_label_full"] = (
                                to_concept  # TODO: fix
                            )
                        else:
                            matched_rk["to_concept"] = rk_rel
                            matched_rk["to_concept"]["missing"] = True

                    counts[model][ds]["after_cv_match"] += 1
                    curr_rk_matches.append(matched_rk)
                    free_gpu(yoloe_res)
                else:
                    print("Need to go look for concepts...")
                    # We did not find anything, try to use the prompt-free yoloe
                    free_gpu(yoloe_res)
                    yoloe_pf_res = yoloe_pf.predict(img)[0]
                    res_dicts = parse_yoloe_res(yoloe_pf_res, prompt_free=True)
                    free_gpu(yoloe_pf_res)
                    # We can not do exact matches here, try again with cosine similarity
                    # print(res_dicts)
                    max_sim_from_concept = get_similar_concept(
                        emb_model, from_concept, res_dicts
                    )
                    max_sim_to_concept = get_similar_concept(
                        emb_model, to_concept, res_dicts
                    )

                    # Merge bbox data with correct labels
                    # print(max_sim_from_concept)
                    # print(max_sim_to_concept)
                    matched_rk = {
                        "rel_label": {
                            "rel_label_idx": get_relation_id(
                                rk_rel["relationship"], sgg_dict
                            ),
                            "rel_label_text": rk_rel["relationship"],
                        },
                    }
                    # Add detection data
                    if "box" in max_sim_from_concept:
                        matched_rk["from_concept"] = max_sim_from_concept["box"]
                        matched_rk["from_concept"]["bb_label"]["bb_label_idx"] = (
                            get_concept_id(from_concept, sgg_dict)
                        )
                        matched_rk["from_concept"]["bb_label"]["bb_label_text"] = (
                            from_concept
                        )
                        matched_rk["from_concept"]["bb_label"]["bb_label_full"] = (
                            from_concept  # TODO: fix
                        )
                    else:
                        matched_rk["from_concept"] = rk_rel
                        matched_rk["from_concept"]["missing"] = True

                    if "box" in max_sim_to_concept:
                        matched_rk["to_concept"] = max_sim_to_concept["box"]
                        matched_rk["to_concept"]["bb_label"]["bb_label_idx"] = (
                            get_concept_id(to_concept, sgg_dict)
                        )
                        matched_rk["to_concept"]["bb_label"]["bb_label_text"] = (
                            to_concept
                        )
                        matched_rk["to_concept"]["bb_label"]["bb_label_full"] = (
                            to_concept  # TODO: fix
                        )
                    else:
                        matched_rk["to_concept"] = rk_rel
                        matched_rk["to_concept"]["missing"] = True

                    counts[model][ds]["after_cv_match"] += 1
                    curr_rk_matches.append(matched_rk)

        matched_rk[rk["question_id"]] = curr_rk_matches

        print(counts)

        # Save data to file
        out_path = Path("..", rk_config.get_final_rk_paths(model, ds))
        make_dir(out_path.parent)
        save_json(matched_rk, out_path)
