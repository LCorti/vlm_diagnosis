import hashlib
import json
import sys
from pathlib import Path
from statistics import median

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision.ops import nms
from tqdm import tqdm
from transformers import Owlv2ForObjectDetection, Owlv2Processor

module_path = str(Path("..", "..").resolve())
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
        sk_data.extend(load_jsonl(Path("..", "..", sk_hdl.get_sk_exp_path())))
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


def make_ovd_cache_key(model_name: str, img_path: Path, texts: list[str]) -> str:
    img_stat = img_path.stat()
    cache_identity = {
        "model": model_name,
        "image": str(img_path.resolve()),
        "image_size": img_stat.st_size,
        "image_mtime_ns": img_stat.st_mtime_ns,
        "texts": texts,
    }
    serialized = json.dumps(cache_identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_ovd_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    return {
        cache_entry["key"]: cache_entry["raw_predictions"]
        for cache_entry in load_jsonl(cache_path)
    }


def get_raw_ovd_predictions(
    model_name: str,
    model: Owlv2ForObjectDetection,
    processor: Owlv2Processor,
    device: torch.device,
    img_path: Path,
    texts: list[str],
    cache: dict[str, dict],
    cache_path: Path,
) -> dict:
    cache_key = make_ovd_cache_key(model_name, img_path, texts)
    if cache_key in cache:
        return cache[cache_key]

    with Image.open(img_path) as img:
        inputs = processor(text=[texts], images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]], device=outputs.pred_boxes.device)
        # Confidence filtering and NMS are deliberately deferred until the sweep.
        res = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.0
        )[0]

    raw_predictions = {
        "boxes": res["boxes"].detach().cpu().tolist(),
        "scores": res["scores"].detach().cpu().tolist(),
        "labels": res["labels"].detach().cpu().tolist(),
    }
    cache[cache_key] = raw_predictions
    append_to_jsonl(
        [{"key": cache_key, "raw_predictions": raw_predictions}], str(cache_path)
    )
    return raw_predictions


def apply_class_agnostic_nms(
    texts: list[str],
    raw_predictions: dict,
    score_threshold: float,
    nms_threshold: float,
) -> dict:
    boxes = torch.tensor(raw_predictions["boxes"], dtype=torch.float32)
    scores = torch.tensor(raw_predictions["scores"], dtype=torch.float32)
    labels = torch.tensor(raw_predictions["labels"], dtype=torch.long)

    above_threshold = scores > score_threshold
    boxes = boxes[above_threshold]
    scores = scores[above_threshold]
    labels = labels[above_threshold]
    if boxes.numel() == 0:
        return {}

    # Deliberately omit labels here: detections for different text queries compete.
    keep = nms(boxes, scores, nms_threshold)
    return filter_ovd_res(texts, boxes[keep], scores[keep], labels[keep])


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
) -> tuple[bool, float | None]:
    sim_th = 0.7
    for elem in list_search:
        sim_from_concept = compute_similarity(
            emb_model,
            concept,
            elem["from_concept"]["bb_label"]["bb_label_text"],
        )
        sim_to_concept = compute_similarity(
            emb_model, concept, elem["to_concept"]["bb_label"]["bb_label_text"]
        )
        max_similarity = max(sim_from_concept, sim_to_concept)
        if max_similarity > sim_th:
            return True, max_similarity

    return False, None


def has_semantic_match(
    emb_model: SentenceTransformer, rk_rel: dict, sk_relations: list[dict]
) -> tuple[bool, list[float]]:
    from_match, from_similarity = has_similar_concept(
        emb_model, rk_rel["from_concept"], sk_relations
    )
    similarities = [from_similarity] if from_similarity is not None else []
    if not from_match:
        return False, similarities

    to_match, to_similarity = has_similar_concept(
        emb_model, rk_rel["to_concept"], sk_relations
    )
    if to_similarity is not None:
        similarities.append(to_similarity)
    return to_match, similarities


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
    OVD_SCORE_THRESHOLD = 0.1
    all_nms_ths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    out_file = "nms_th_sweep.jsonl"
    ovd_cache_path = Path("ovd_raw_predictions_cache.jsonl")
    ovd_cache = load_ovd_cache(ovd_cache_path)
    evaluations = []

    print(f"OWLv2 device: {ovd_device}")
    print(f"Embedding model device: {emb_model.device}")
    print(f"Loaded {len(ovd_cache)} cached OWLv2 predictions")

    # Build threshold-independent evaluation records. OWLv2 and semantic matching are each evaluated only once instead of once per threshold.
    for model, ds in tqdm(zip(all_models, all_ds), total=len(all_ds)):
        tqdm.write(f"{model} -- {ds}")

        sk_hdl.set_curr_ds(ds)
        sk_final = load_merge_sk_data(sk_hdl)
        sk_by_question = {sk["question_id"]: sk for sk in sk_final}

        rk_hdl.set_curr_model(model)
        rk_hdl.set_curr_ds(ds)
        rk_parsed_path = Path(
            "..", "..", str(rk_hdl.get_rk_parsed_path()).format(PROMPT_VERSION)
        )
        rk_parsed = load_jsonl(rk_parsed_path)

        for rk in tqdm(rk_parsed, desc=f"{model} -- {ds}", leave=False):
            if len(rk["triple_objs"]) == 0:
                tqdm.write(f"No RKs found for question {rk['question_id']}")
                continue

            curr_sk = sk_by_question.get(rk["question_id"])
            if not curr_sk:
                tqdm.write(f"Big error: SK missing for question {rk['question_id']}")
                break

            img_path = Path(
                "..", "..", curr_sk["img_path"].replace("/imgs/", "/imgs_resized/")
            )

            for rk_rel in rk["triple_objs"]:
                evaluation = {"model": model, "ds": ds, "exact_match": False}
                if exact_match_sk(rk_rel, curr_sk["relations"]):
                    evaluation["exact_match"] = True
                    evaluations.append(evaluation)
                    continue

                texts = [rk_rel["from_concept"], rk_rel["to_concept"]]
                raw_predictions = get_raw_ovd_predictions(
                    ovd_model_name,
                    ovd_model,
                    ovd_processor,
                    ovd_device,
                    img_path,
                    texts,
                    ovd_cache,
                    ovd_cache_path,
                )
                evaluation["texts"] = texts
                evaluation["raw_ovd_predictions"] = raw_predictions
                (
                    evaluation["semantic_match"],
                    evaluation["semantic_similarities"],
                ) = has_semantic_match(emb_model, rk_rel, curr_sk["relations"])
                evaluations.append(evaluation)

    for nms_th in all_nms_ths:
        print(
            f">> Evaluating class-agnostic NMS th: {nms_th} "
            f"(OWLv2 score th: {OVD_SCORE_THRESHOLD})"
        )
        counts = {
            model: {
                ds: {
                    "initial": 0,
                    "with_exact_match": 0,
                    "with_ovd_match": 0,
                    "with_sim_match": 0,
                }
                for ds in ds_list
            }
            for model in model_list
        }
        semantic_similarities = {
            model: {ds: [] for ds in ds_list} for model in model_list
        }
        for evaluation in evaluations:
            model = evaluation["model"]
            ds = evaluation["ds"]
            counters = counts.setdefault(model, {}).setdefault(
                ds,
                {
                    "initial": 0,
                    "with_exact_match": 0,
                    "with_ovd_match": 0,
                    "with_sim_match": 0,
                },
            )
            counters["initial"] += 1

            if evaluation["exact_match"]:
                counters["with_exact_match"] += 1
            else:
                filtered_predictions = apply_class_agnostic_nms(
                    evaluation["texts"],
                    evaluation["raw_ovd_predictions"],
                    OVD_SCORE_THRESHOLD,
                    nms_th,
                )
                if all(text in filtered_predictions for text in evaluation["texts"]):
                    counters["with_ovd_match"] += 1
                    continue

                semantic_similarities[model][ds].extend(
                    evaluation["semantic_similarities"]
                )
                if evaluation["semantic_match"]:
                    counters["with_sim_match"] += 1

        for model, datasets in counts.items():
            for ds, counters in datasets.items():
                similarity_values = semantic_similarities[model][ds]
                counters["semantic_similarity_median"] = (
                    median(similarity_values) if similarity_values else None
                )

        print(counts)
        out_data = {"th": int(nms_th * 10), "data": counts}
        append_to_jsonl([out_data], out_file)
