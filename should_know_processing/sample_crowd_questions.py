import os
import sys

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.dataset_config_loader import DatasetConfig
from config_loaders.sk_config_loader import SKConfig
from utils.data_io import load_jsonl, save_jsonl, save_json


if __name__ == "__main__":
    # Load dataset config
    ds_config = DatasetConfig()
    ds_list = ds_config.get_ds_list()
    # Skip holdout set from VQA v2
    ds_list.remove("vqav2_holdout")

    # Load SK config
    sk_config = SKConfig()

    for ds in ds_list:
        ds_paths = ds_config.get_ds_paths(ds)
        sk_paths = sk_config.get_sk_paths(ds)
        ds_crowd_questions = []

        for ds_class in sk_paths:
            print(f"Dealing with {ds} + {ds_class}")
            questions = load_jsonl(f"../{ds_paths[ds_class]['questions']}")
            print(f"... Loaded {len(questions)} questions")

            base_sk_dir = sk_paths[ds_class]["dir"]
            crowd_sg_file = sk_paths[ds_class]["sg_crowd"]
            crowd_sg = load_jsonl(f"../{base_sk_dir}/{crowd_sg_file}")
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
            out_file_path = f"../{ds_paths[ds_class]['sampled_questions']}"
            ds_crowd_questions.extend(sampled_questions)
            save_jsonl(sampled_questions, out_file_path)

        # Save single file for dataset
        out_file_path = f"../data/datasets/{ds}/q_crowd.json"
        save_json(ds_crowd_questions, out_file_path)
