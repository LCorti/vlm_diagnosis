import os
import sys

from evaluate import load
from pathlib import Path
from eval_utils import (
    cleanup_resp,
    get_eval_dict,
    prep_for_split,
    prep_resp,
    save_bertscore_hash,
)

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.dataset_config_loader import DatasetConfig
from config_loaders.eval_config_loader import EvalConfigLoader
from config_loaders.rk_config_loader import RKConfig
from utils.data_io import load_jsonl, make_dir, save_json

PROMPT_VERSION = 4
REF_MC_ANSWERS = ["A", "B", "C", "D"]

START = 0
END = 100
STEP = 20

if __name__ == "__main__":
    rk_config = RKConfig()
    ds_config = DatasetConfig()
    eval_config = EvalConfigLoader()

    ds_list = ds_config.get_ds_list()
    ds_list.remove("vqav2_holdout")
    model_list = rk_config.get_model_list()

    all_models = model_list * 4
    all_models.sort()
    all_ds = ds_list * 4

    eval_dict = get_eval_dict(all_models, all_ds, start=START, end=END, step=STEP)

    for model, ds in zip(all_models, all_ds):
        print(f"Evaluating {model} on {ds}...")
        # Load responses for current model + dataset combo
        resp_path = Path("..", rk_config.get_parsed_rk_paths(model, ds))
        resp_path = resp_path.format(PROMPT_VERSION)
        resp_data = load_jsonl(resp_path)
        count_ok = 0

        # Iterate through datasets' paths
        ds_paths = ds_config.get_ds_paths(ds)
        for ds_class in ds_paths:
            # Load questions
            questions_path = Path("..", ds_paths[ds_class]["sampled_questions"])
            questions = load_jsonl(questions_path)

            if ds in ["llava-bench", "mmbench"]:
                # For open-ended VQA datasets, we compute BERTScore-F1
                bertscore = load("bertscore")

                for q in questions:
                    resp = next(
                        (r for r in resp_data if r["question_id"] == q["question_id"]),
                        None,
                    )

                    if not resp:
                        eval_dict[model][ds]["invalid_resp"].append(
                            prep_resp(q, "<NOT_FOUND>")
                        )
                        continue

                    # Compute accuracy on single data instance
                    # Using microsoft/deberta-xlarge-mnli suggested by authors
                    # https://github.com/Tiiiger/bert_score
                    scores = bertscore.compute(
                        predictions=[resp["response"]],
                        references=[q["answer"]],
                        lang="en",
                        model_type="microsoft/deberta-xlarge-mnli",
                    )
                    # Update measures
                    measures = {
                        "precision": scores["precision"][0],
                        "recall": scores["recall"][0],
                        "f1": scores["f1"][0],
                    }
                    eval_dict[model][ds]["measures"]["precision"] += measures[
                        "precision"
                    ]
                    eval_dict[model][ds]["measures"]["recall"] += measures["recall"]
                    eval_dict[model][ds]["measures"]["f1"] += measures["f1"]

                    # Update splits over F1
                    for split in range(START, END, STEP):
                        if split <= 100 * measures["f1"] < split + STEP:
                            split_key = f"{split}_{split + STEP}"
                            eval_dict[model][ds]["ranges"][split_key].append(
                                prep_for_split(q, resp["response"], measures)
                            )

                    # Handle case where F1 is 1
                    if float(measures["f1"]) == 1.0:
                        eval_dict[model][ds]["ranges"]["80_100"].append(
                            prep_for_split(q, resp["response"], measures)
                        )

                # Save bertscore info
                bertscore_hash_path = "../data/eval_performance/bertscore_hash.txt"
                save_bertscore_hash(scores["hashcode"], bertscore_hash_path)
            elif ds in ["seed", "vqav2"]:
                for q in questions:
                    resp = next(
                        (r for r in resp_data if r["question_id"] == q["question_id"]),
                        None,
                    )
                    if not resp:
                        eval_dict[model][ds]["invalid_resp"].append(
                            prep_resp(q, "<NOT_FOUND>")
                        )
                        continue

                    # Clean up text
                    clean_resp = cleanup_resp(resp["response"])
                    # If nothing is left after this simple cleaining, save as invalid
                    if len(clean_resp) == 0:
                        eval_dict[model][ds]["invalid_resp"].append(
                            prep_resp(q, "<INV>")
                        )
                        continue

                    if ds == "seed":
                        # If option letter matches directly
                        if q["answer"] == clean_resp:
                            eval_dict[model][ds]["correct_resp"].append(
                                prep_resp(q, clean_resp)
                            )
                            count_ok += 1
                        # If model answer is present as-is in the list of alternatives even with no letter
                        elif q["options"] is not None and clean_resp in list(
                            q["options"].values()
                        ):
                            # Get corresponding letter and save that
                            ref_letter = next(
                                k for k, v in q["options"] if v == clean_resp
                            )
                            eval_dict[model][ds]["correct_resp"].append(
                                prep_resp(q, ref_letter)
                            )
                            count_ok += 1
                        # If either ':' or '.' is present try to parse as
                        # '<letter>: <explanation>'. They seem to be used frequently
                        # by the MLLMs tested.
                        elif clean_resp.find(":") != -1 or clean_resp.find(".") != -1:
                            # Figure out which character it is
                            if clean_resp.find(":") != -1:
                                char = ":"
                            else:
                                char = "."

                            ref_letter = clean_resp.split(char)[0]
                            if (
                                len(ref_letter) == 1
                                and ref_letter in REF_MC_ANSWERS
                                and ref_letter == q["answer"]
                            ):
                                eval_dict[model][ds]["correct_resp"].append(
                                    prep_resp(q, ref_letter)
                                )
                                count_ok += 1
                            else:
                                eval_dict[model][ds]["wrong_resp"].append(
                                    prep_resp(q, ref_letter)
                                )
                        # For some responses there is a preamble, e.g., 'The correct answer is'.
                        # The letter indicating the reponse appears right before.
                        elif clean_resp[-1] == q["answer"]:
                            eval_dict[model][ds]["correct_resp"].append(
                                prep_resp(q, clean_resp)
                            )
                            count_ok += 1
                        else:
                            eval_dict[model][ds]["wrong_resp"].append(
                                prep_resp(q, clean_resp)
                            )
                    elif ds == "vqav2":
                        # For VQA v2, consider as correct
                        # - exact matches
                        # - reponses that contain the ground truth answer
                        if q["answer"].lower() in clean_resp.lower():
                            eval_dict[model][ds]["correct_resp"].append(
                                prep_resp(q, clean_resp)
                            )
                            count_ok += 1
                        else:
                            eval_dict[model][ds]["wrong_resp"].append(
                                prep_resp(q, clean_resp)
                            )
            else:
                print("Something went very wrong.")

        if ds in ["llava-bench", "mmbench"]:
            # Compute average scores for open-ended datasets
            eval_dict[model][ds]["measures"]["precision"] /= len(resp_data)
            eval_dict[model][ds]["measures"]["recall"] /= len(resp_data)
            eval_dict[model][ds]["measures"]["f1"] /= len(resp_data)
        else:
            # Compute average scores for close-ended datasets
            eval_dict[model][ds]["measures"]["accuracy"] = float(
                count_ok / len(resp_data)
            )

        # Print and save scores
        if ds in ["llava-bench", "mmbench"]:
            print(f"- F1 = {eval_dict[model][ds]['measures']['f1']}")
        elif ds in ["seed", "vqav2"]:
            print(f"- accuracy = {eval_dict[model][ds]['measures']['accuracy']}")
        else:
            print("Something went very wrong.")

        # Save measures to file
        # Make dirs, if necessary
        out_path = Path("..", eval_config.get_eval_paths(model, ds))
        make_dir(Path(out_path).parent)
        save_json(eval_dict[model][ds], out_path)
