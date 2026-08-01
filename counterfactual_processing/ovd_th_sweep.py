import sys
from pathlib import Path

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import Owlv2ForObjectDetection, Owlv2Processor

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.rk_handler import RKHandler  # noqa: E402
from config_handlers.sk_handler import SKHandler  # noqa: E402
from utils.data_io import append_to_jsonl, load_jsonl  # noqa: E402


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
    texts: list[str], boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> dict:
    filtered_res = {}
    for box, score, label in zip(boxes, scores, labels):
        label_idx = int(label.item())
        score_value = float(score.item())
        # Reducing to an integer only to match the output from IETrans
        box_values = [int(i) for i in box.detach().cpu().tolist()]
        # For each label, take the detection with the max score
        if texts[label_idx] not in filtered_res:
            filtered_res[texts[label_idx]] = {"score": score_value, "box": box_values}
        elif filtered_res[texts[label_idx]]["score"] < score_value:
            filtered_res[texts[label_idx]]["score"] = score_value
            filtered_res[texts[label_idx]]["box"] = box_values
    return filtered_res


# Cosine similarity utils
def compute_similarity(model: SentenceTransformer, text_a: str, text_b: str) -> float:
    text_a_emb = torch.FloatTensor(
        model.encode(text_a, normalize_embeddings=True)
    ).unsqueeze(0)
    text_b_emb = torch.FloatTensor(
        model.encode(text_b, normalize_embeddings=True)
    ).unsqueeze(0)
    return model.similarity(text_a_emb, text_b_emb).item()


def has_similar_concept(
    emb_model: SentenceTransformer, concept: str, list_search: list[dict]
) -> bool:
    sim_th = 0.8
    for elem in list_search:
        sim_from_concept = compute_similarity(
            emb_model,
            concept,
            elem["from_concept"]["bb_label"]["bb_label_text"],
        )
        sim_to_concept = compute_similarity(
            emb_model, concept, elem["to_concept"]["bb_label"]["bb_label_text"]
        )
        if sim_from_concept > sim_th or sim_to_concept > sim_th:
            return True

    return False


def has_semantic_match(
    emb_model: SentenceTransformer, rk_rel: dict, sk_relations: list[dict]
) -> bool:
    return has_similar_concept(
        emb_model, rk_rel["from_concept"], sk_relations
    ) and has_similar_concept(emb_model, rk_rel["to_concept"], sk_relations)


if __name__ == "__main__":
    PROMPT_VERSION = 4

    # Loading config files
    sk_hdl = SKHandler()
    rk_hdl = RKHandler()
    ds_list = sk_hdl.get_ds_list()
    model_list = rk_hdl.get_model_list()
    # Skip MiniGPT-4
    model_list.remove("minigpt4")
    all_ds = ds_list * len(model_list)
    all_models = model_list * len(ds_list)
    all_models.sort()
    counts = {}

    # Load sentence-transformers for computing embeddings
    # emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # Use granite with eager attention.
    emb_model = SentenceTransformer(
        "ibm-granite/granite-embedding-english-r2",
        model_kwargs={"attn_implementation": "eager"},
    )
    ovd_model_name = "google/owlv2-large-patch14-ensemble"
    ovd_processor = Owlv2Processor.from_pretrained(ovd_model_name)
    ovd_model = Owlv2ForObjectDetection.from_pretrained(
        ovd_model_name, device_map="auto"
    )
    ovd_model.eval()
    ovd_device = next(ovd_model.parameters()).device
    all_ovd_ths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    out_file = "ovd_th_sweep.jsonl"

    print(f"OWLv2 device: {ovd_device}")
    print(f"Embedding model device: {emb_model.device}")

    for ovd_th in all_ovd_ths:
        print(f">> Evaluating OWLv2 th: {ovd_th}")

        for model, ds in tqdm(zip(all_models, all_ds), total=len(all_ds)):
            tqdm.write(f"{model} -- {ds}")
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

            for rk in tqdm(rk_parsed, desc=f"{model} -- {ds}", leave=False):
                # print(f">> Checking question {rk['question_id']}")
                if len(rk["triple_objs"]) == 0:
                    tqdm.write(f"No RKs found for question {rk['question_id']}")
                    continue

                # Get SK data for current question
                curr_sk = next(
                    (sk for sk in sk_final if sk["question_id"] == rk["question_id"]),
                    None,
                )
                if not curr_sk:
                    tqdm.write(
                        f"Big error: SK missing for question {rk['question_id']}"
                    )
                    break

                # Load (resized) image for the OVD matching step
                img_path = curr_sk["img_path"].replace("/imgs/", "/imgs_resized/")
                img_path = Path("..", img_path)
                img = Image.open(img_path)

                # RK format: {'from_concept': ..., 'relationship': ..., 'to_concept': ...}
                # Count each relation once, using the first successful matching stage.
                for rk_rel in rk["triple_objs"]:
                    # print(f">> >> Checking relation: {rk_rel}")
                    # Update count
                    counts[model][ds]["initial"] += 1

                    # First try: exact match -- look for the RK in the list of SK
                    # The matching is done at the triple-level to void the "wrong" concepts
                    # to be matched by mistake
                    match_sk_rel = exact_match_sk(rk_rel, curr_sk["relations"])
                    if match_sk_rel:
                        counts[model][ds]["with_exact_match"] += 1
                        continue

                    # Second try: zero-shot OVD with OWLv2
                    texts = [rk_rel["from_concept"], rk_rel["to_concept"]]
                    inputs = ovd_processor(
                        text=[texts], images=img, return_tensors="pt"
                    ).to(ovd_device)
                    with torch.no_grad():
                        outputs = ovd_model(**inputs)
                    target_sizes = torch.tensor(
                        [img.size[::-1]], device=outputs.pred_boxes.device
                    )
                    res = ovd_processor.post_process_grounded_object_detection(
                        outputs=outputs, target_sizes=target_sizes, threshold=ovd_th
                    )
                    boxes, scores, labels = (
                        res[0]["boxes"],
                        res[0]["scores"],
                        res[0]["labels"],
                    )
                    filtered_res = filter_ovd_res(texts, boxes, scores, labels)
                    if all(text in filtered_res for text in texts):
                        counts[model][ds]["with_ovd_match"] += 1
                        continue

                    # Third try: semantic similarity against SK concepts
                    if has_semantic_match(emb_model, rk_rel, curr_sk["relations"]):
                        counts[model][ds]["with_sim_match"] += 1

        print(counts)

        # Save to file
        out_data = {"th": int(ovd_th * 10), "data": counts}
        append_to_jsonl(out_data, out_file)
