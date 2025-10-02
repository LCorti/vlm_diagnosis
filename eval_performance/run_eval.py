import sys

from evaluate import load
from pathlib import Path
from eval_utils import (
    get_eval_dict,
    prep_for_split,
    prep_resp,
    save_bertscore_hash,
)

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.dataset_handler import DatasetHandler
from config_handlers.eval_handler import EvalHandler
from config_handlers.rk_handler import RKHandler
from utils.data_io import load_jsonl, make_dir, save_json

REF_MC_ANSWERS = ["A", "B", "C", "D"]

START = 0
END = 100
STEP = 20

if __name__ == "__main__":
    rk_hdl = RKHandler()
    ds_hdl = DatasetHandler()
    eval_hdl = EvalHandler()

    ds_list = ds_hdl.get_ds_list()
    model_list = rk_hdl.get_model_list()

    all_models = model_list * 4
    all_models.sort()
    all_ds = ds_list * 4

    eval_dict = get_eval_dict(all_models, all_ds, start=START, end=END, step=STEP)

    for model, ds in zip(all_models, all_ds):
        print(f"Evaluating {model} on {ds}...")
        # Load responses for current model + dataset combo
        rk_hdl.set_curr_model(model)
        rk_hdl.set_curr_ds(ds)
        resp_path = Path("..", rk_hdl.get_rk_final_path())
        resp_data = load_jsonl(resp_path)
        count_ok = 0

        # For open-ended VQA datasets, we compute BERTScore-F1
        if ds in ["llava-bench", "mmbench"]:
            bertscore = load("bertscore")

        # Iterate through datasets' paths
        ds_hdl.set_curr_ds(ds)
        for ds_class in ds_hdl.get_classes():
            # Load questions
            ds_hdl.set_curr_class(ds_class)
            q_path = ds_hdl.get_sampled_questions_path()
            questions = load_jsonl(Path("..", q_path))

            for q in questions:
                print(f"Processing question {q['question_id']}")
                resp = next(
                    (r for r in resp_data if r["question_id"] == q["question_id"]),
                    None,
                )
                if not resp:
                    print("> Not found.")
                    eval_dict[model][ds]["invalid_resp"].append(
                        prep_resp(q, "<NOT_FOUND>")
                    )
                    continue

                if ds in ["llava-bench", "mmbench"]:
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

                elif ds in ["seed", "vqav2"]:
                    # If nothing is left after this simple cleaining, save as invalid
                    if resp["response"] is None or len(resp["response"]) == 0:
                        eval_dict[model][ds]["invalid_resp"].append(
                            prep_resp(q, "<INV>")
                        )
                        continue
                    if ds == "seed":
                        # If option letter matches directly
                        if (
                            "options" in q
                            and resp["response"] in q["options"]
                            and q["answer"] == resp["response"]
                        ):
                            eval_dict[model][ds]["correct_resp"].append(
                                prep_resp(q, resp["response"])
                            )
                            count_ok += 1
                        else:
                            eval_dict[model][ds]["wrong_resp"].append(
                                prep_resp(q, resp["response"])
                            )
                    elif ds == "vqav2":
                        # For VQA v2, consider as correct
                        # - exact matches
                        # - reponses that contain the ground truth answer
                        if q["answer"].lower() in resp["response"].lower():
                            eval_dict[model][ds]["correct_resp"].append(
                                prep_resp(q, resp["response"])
                            )
                            count_ok += 1
                        else:
                            eval_dict[model][ds]["wrong_resp"].append(
                                prep_resp(q, resp["response"])
                            )
                else:
                    print("Something went very wrong.")

        if ds in ["llava-bench", "mmbench"]:
            # Save bertscore info
            bertscore_hash_path = Path(
                "..", "data", "eval_performance", "bertscore_hash.txt"
            )
            save_bertscore_hash(scores["hashcode"], bertscore_hash_path)
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
        eval_hdl.set_curr_model(model)
        eval_hdl.set_curr_ds(ds)
        out_path = Path("..", eval_hdl.get_eval_path())
        make_dir(Path(out_path).parent)
        save_json(eval_dict[model][ds], out_path)
