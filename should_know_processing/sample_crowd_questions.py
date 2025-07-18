import sys

from pathlib import Path

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.dataset_handler import DatasetHandler
from config_loaders.sk_handler import SKHandler
from utils.data_io import load_jsonl, save_jsonl, save_json


if __name__ == "__main__":
    # Load DS config
    ds_hdl = DatasetHandler()
    ds_list = ds_hdl.get_sk_list()
    if "vqav2_holdout" in ds_list:
        ds_list.remove("vqav2_holdout")

    # Load SK config
    sk_hdl = SKHandler()

    for ds in ds_list:
        ds_hdl.set_curr_ds(ds)
        sk_hdl.set_curr_ds(ds)
        ds_crowd_questions = []

        for ds_class in sk_hdl.get_classes():
            print(f"Dealing with {ds} + {ds_class}")
            ds_hdl.set_curr_class(ds_class)
            questions = load_jsonl(Path("..", ds_hdl.get_questions_path()))
            print(f"... Loaded {len(questions)} questions")

            sk_hdl.set_curr_class(ds_class)
            crowd_sg = load_jsonl(Path("..", sk_hdl.get_sg_crowd_path()))
            print(f"... Loaded {len(crowd_sg)} scene graphs")

            sampled_questions = []

            if ds == "llava-bench":
                # llava-bench is used as-is
                sampled_questions = questions
            else:
                for img in crowd_sg:
                    q_obj = next(
                        (q for q in questions if q["img"] == img["img_id"]),
                        None,
                    )
                    if q_obj:
                        sampled_questions.append(q_obj)
                    else:
                        print(img["img_id"])

                    # for q in questions:
                    #     if q["question_id"] == img["img_id"]:
                    #         sampled_questions.append(q)
                    # if not (ds == "mmbench" and ds_class == "image_topic"):
                    #     break

            # Save individual files
            out_file_path = Path("..", ds_hdl.get_sampled_questions_path())
            ds_crowd_questions.extend(sampled_questions)
            save_jsonl(sampled_questions, out_file_path)

        # Save single file for dataset
        out_file_path = Path("..", "data", "datasets", ds, "q_crowd.json")
        save_json(ds_crowd_questions, out_file_path)
