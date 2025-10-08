import numpy as np
import sys
import torch

from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import Owlv2Processor, Owlv2ForObjectDetection

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.rk_handler import RKHandler
from config_handlers.sk_handler import SKHandler
from utils.data_io import make_dir, load_json, load_jsonl, save_jsonl


# Data utils
def load_merge_sk_data(sk_hdl: SKHandler) -> list:
    sk_data = []
    for ds_class in sk_hdl.get_classes():
        sk_hdl.set_curr_class(ds_class)
        sk_data.extend(load_jsonl(Path("..", sk_hdl.get_sk_exp_path())))
    return sk_data


def exact_match_sk(rk_rel: dict, sk_list: list[dict]) -> dict:
    return next(
        (
            (
                sk_rel
                for sk_rel in sk_list
                if sk_rel["from_concept"]["bb_label"]["bb_label_text"]
                == rk_rel["from_concept"]
                and sk_rel["rel_label"]["rel_label_text"] == rk_rel["relationship"]
                and sk_rel["to_concept"]["bb_label"]["bb_label_text"]
                == rk_rel["to_concept"]
            )
        ),
        None,
    )


# OVD utils
def filter_ovd_res(
    texts: list[str], boxes: list[float], scores: list[float], labels: list[int]
) -> dict:
    filtered_res = {}
    for box, score, label in zip(boxes, scores, labels):
        # Reducing to an integer only to match the output from IETrans
        box = [int(i) for i in box.tolist()]
        # For each label, take the detection with the max score
        if texts[label] not in filtered_res:
            filtered_res[texts[label]] = {"score": score, "box": box}
        elif filtered_res[texts[label]]["score"] < score:
            filtered_res[texts[label]]["score"] = score
            filtered_res[texts[label]]["box"] = box
    return filtered_res


# Cosine similarity utils
def compute_similarity(model: SentenceTransformer, text_a: str, text_b: str) -> float:
    text_a_emb = torch.FloatTensor(
        model.encode(text_a, normalize_embeddings=True)
    ).unsqueeze(0)
    text_b_emb = torch.FloatTensor(
        model.encode(text_b, normalize_embeddings=True)
    ).unsqueeze(0)
    return model.similarity(text_a_emb, text_b_emb)


def get_similar_concept(
    emb_model: SentenceTransformer, concept: str, list_search: list[dict]
) -> dict:
    max_sim = -1
    sim_th = 0.7
    curr_concept = {}
    for elem in list_search:
        sim_from_concept = compute_similarity(
            emb_model,
            concept,
            elem["from_concept"]["bb_label"]["bb_label_text"],
        )
        sim_to_concept = compute_similarity(
            emb_model, concept, elem["from_concept"]["bb_label"]["bb_label_text"]
        )
        # Keep track of max sim and corresponding data
        if sim_from_concept > sim_th and sim_from_concept > max_sim:
            max_sim = sim_from_concept
            curr_concept = elem["from_concept"]
        if sim_to_concept > sim_th and sim_to_concept > max_sim:
            max_sim = sim_to_concept
            curr_concept = elem["to_concept"]

    return curr_concept


def get_concept_id(concept_label: str, sgg_dict: dict) -> int | str:
    if concept_label in sgg_dict["label_to_idx"]:
        return sgg_dict["label_to_idx"][concept_label]
    else:
        return "new"


def format_concept(label: str, data: dict) -> dict:
    concept_id = get_concept_id(label, sgg_dict)
    out = {
        "top_left_x": data["box"][0],
        "top_left_y": data["box"][1],
        "bottom_right_x": data["box"][2],
        "bottom_right_y": data["box"][3],
        "width": np.abs(data["box"][0] - data["box"][2]).item(),
        "height": np.abs(data["box"][1] - data["box"][3]).item(),
        "bb_label": {
            "bb_label_idx": concept_id,
            "bb_label_text": label,
            "bb_label_full": f"{concept_id}-{label}",
        },
    }
    return out


def get_relation_id(relation_label: str, sgg_dict: dict) -> int | str:
    if relation_label in sgg_dict["predicate_to_idx"]:
        return sgg_dict["predicate_to_idx"][relation_label]
    else:
        return "new"


if __name__ == "__main__":
    PROMPT_VERSION = 4

    # Loading config files
    sk_hdl = SKHandler()
    rk_hdl = RKHandler()
    model_list = rk_hdl.get_model_list() * 4
    ds_list = sk_hdl.get_ds_list()
    ds_list = ds_list * 4
    model_list.sort()
    # Load sgg dict
    sgg_dict = load_json(Path("..", "data", "common", "sgg_dicts.json"))
    # Variables to check what is happening
    counts = {}

    # Load sentence-transformers for computing embeddings
    # emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    emb_model = SentenceTransformer("ibm-granite/granite-embedding-english-r2")
    ovd_model_name = "google/owlv2-large-patch14-ensemble"
    ovd_processor = Owlv2Processor.from_pretrained(ovd_model_name, device_map="auto")
    ovd_model = Owlv2ForObjectDetection.from_pretrained(
        ovd_model_name, device_map="auto"
    )
    ovd_model.eval()
    ovd_th = 0.1

    for model, ds in zip(model_list, ds_list):
        print(f"{model} -- {ds}")
        # Initialise counters
        if model not in counts:
            counts[model] = {}
        if ds not in counts[model]:
            counts[model][ds] = {
                "initial": 0,
                "with_exact_match": 0,
                "with_ovd_match": 0,
                "with_sim_match": 0,
            }

        # Load SK final
        sk_hdl.set_curr_ds(ds)
        # Merge batches from classes
        sk_final = load_merge_sk_data(sk_hdl)

        # Load parsed RK
        rk_hdl.set_curr_model(model)
        rk_hdl.set_curr_ds(ds)
        rk_parsed_path = Path(
            "..", str(rk_hdl.get_rk_parsed_path()).format(PROMPT_VERSION)
        )
        rk_parsed = load_jsonl(rk_parsed_path)

        all_matches = []

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

            # Load (resized) image for the OVD matching step
            img_path = curr_sk["img_path"].replace("/imgs/", "/imgs_resized/")
            img_path = Path("..", img_path)
            img = Image.open(img_path)

            # Make SK into a format that can be easily compared with SK
            # RK format: {'from_concept': ..., 'relationship': ..., 'to_concept': ...}
            curr_sk_rels = curr_sk["relations"]
            # Go relation by relation and try to match
            curr_rk_matches = []
            for rk_rel in rk["triple_objs"]:
                print(f">> >> Checking relation: {rk_rel}")
                # Update count
                counts[model][ds]["initial"] += 1

                # First try: exact match -- look for the RK in the list of SK
                # The matching is done at the triple-level to void the "wrong" concepts
                # to be matched by mistake
                match_sk_rel = exact_match_sk(rk_rel, curr_sk_rels)
                # If a match is found, just use that (and update count)
                # and go to next iteration
                if match_sk_rel:
                    counts[model][ds]["with_exact_match"] += 1
                    curr_rk_matches.append(match_sk_rel)
                    continue

                # Second try: Zero-shot OVD with OWLv2
                # Prepare output object
                matched_rk = {
                    "rel_label": {
                        "rel_label_idx": get_relation_id(
                            rk_rel["relationship"], sgg_dict
                        ),
                        "rel_label_text": rk_rel["relationship"],
                    }
                }
                texts = [rk_rel["from_concept"], rk_rel["to_concept"]]
                inputs = ovd_processor(text=[texts], images=img, return_tensors="pt")
                with torch.no_grad():
                    outputs = ovd_model(**inputs)
                target_sizes = torch.Tensor([img.size[::-1]])
                res = ovd_processor.post_process_grounded_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=ovd_th
                )
                boxes, scores, labels = (
                    res[0]["boxes"],
                    res[0]["scores"],
                    res[0]["labels"],
                )
                filtered_res = filter_ovd_res(texts, boxes, scores, labels)
                doing_cosine = False
                # Just to update the counts correctly
                if (
                    rk_rel["from_concept"] in filtered_res
                    and rk_rel["to_concept"] in filtered_res
                ):
                    counts[model][ds]["with_ovd_match"] += 1
                else:
                    doing_cosine = True

                # Handle 'from_concept'
                if rk_rel["from_concept"] in filtered_res:
                    # Present, just need to format data
                    print("from_concept found")
                    matched_rk["from_concept"] = format_concept(
                        rk_rel["from_concept"], filtered_res[rk_rel["from_concept"]]
                    )
                else:
                    # Missing, need to run cosine sim
                    print(
                        f"Searching from_concept ({rk_rel['to_concept']}) with cosine sim."
                    )
                    sim_concept = get_similar_concept(
                        emb_model, rk_rel["from_concept"], curr_sk["relations"]
                    )
                    print(sim_concept)
                    if not sim_concept:
                        print("cosine sim. did not find anything")
                        continue

                    matched_rk["from_concept"] = sim_concept
                    label = rk_rel["from_concept"]
                    concept_id = get_concept_id(label, sgg_dict)
                    matched_rk["from_concept"]["bb_label"]["bb_label_idx"] = concept_id
                    matched_rk["from_concept"]["bb_label"]["bb_label_text"] = label
                    matched_rk["from_concept"]["bb_label"]["bb_label_full"] = (
                        f"{concept_id}-{label}"
                    )
                # Handle 'to_concept'
                if rk_rel["to_concept"] in filtered_res:
                    # Present, just need to format data
                    print("to_concept found")
                    matched_rk["to_concept"] = format_concept(
                        rk_rel["to_concept"], filtered_res[rk_rel["to_concept"]]
                    )
                else:
                    # Missing, need to run cosine sim
                    print(
                        f"Searching to_concept ({rk_rel['to_concept']}) with cosine sim."
                    )
                    sim_concept = get_similar_concept(
                        emb_model, rk_rel["to_concept"], curr_sk["relations"]
                    )
                    print(sim_concept)
                    if not sim_concept:
                        print("cosine sim. did not find anything")
                        continue

                    matched_rk["to_concept"] = sim_concept
                    label = rk_rel["to_concept"]
                    concept_id = get_concept_id(label, sgg_dict)
                    matched_rk["to_concept"]["bb_label"]["bb_label_idx"] = concept_id
                    matched_rk["to_concept"]["bb_label"]["bb_label_text"] = label
                    matched_rk["to_concept"]["bb_label"]["bb_label_full"] = (
                        f"{concept_id}-{label}"
                    )

                if (
                    doing_cosine
                    and "from_concept" in matched_rk
                    and "to_concept" in matched_rk
                ):
                    counts[model][ds]["with_sim_match"] += 1
                curr_rk_matches.append(matched_rk)

            all_matches.append(
                {
                    "question_id": rk["question_id"],
                    "response": rk["response_clean"],
                    "triple_objs": curr_rk_matches,
                }
            )

        # Save data to file
        out_path = Path("..", rk_hdl.get_rk_final_path())
        make_dir(out_path.parent)
        save_jsonl(all_matches, out_path)

    print(counts)
