import copy

from config_loaders.dataset_config_loader import DatasetConfig
from utils.data_io import load_jsonl
from pathlib import Path


class QuestionFormatter:
    def __init__(self, dataset):
        self.QUESTION_TEMPLATE = {
            "question_id": "",
            "question": "",
            "img": "",
            "img_path": "",
            "is_open": True,  # 0: multiple-choice, 1: open-ended
            "options": {"A": "", "B": "", "C": "", "D": ""},
            "answer": "",
            "class": "",
        }
        self.DATASET_CONFIG = DatasetConfig()
        self.DATASET = self.validate_dataset_name(dataset)
        self.DATASET_PATHS = self.DATASET_CONFIG.get_ds_paths(self.DATASET)

        if self.DATASET == "llava-bench":
            self.LLAVA_ANS = self.load_llavabench_ans()
        else:
            self.LLAVA_ANS = None

    def validate_dataset_name(self, dataset):
        if dataset in self.DATASET_CONFIG.get_ds_list():
            return dataset
        else:
            raise

    def format(self, raw_q, ds_class):
        new_q = copy.deepcopy(self.QUESTION_TEMPLATE)

        if self.DATASET == "llava-bench":
            new_q["question_id"] = raw_q["question_id"]
            new_q["question"] = raw_q["text"]
            new_q["img"] = Path(raw_q["image"]).stem
            new_q["img_path"] = str(
                Path(".", self.DATASET_PATHS[ds_class]["imgs"], raw_q["image"])
            )
            new_q["is_open"] = True
            new_q["options"] = None
            new_q["answer"] = self.get_llava_answer(raw_q["question_id"])
            new_q["class"] = "main"
        elif self.DATASET == "mmbench":
            # MMBench is used in a generation setting. Only for this dataset,
            # we change the question and ask the model to generate a caption
            # instead of picking one from the alternatives.
            new_q["question_id"] = raw_q["index"]
            new_q["question"] = (
                "Provide a concise and descriptive caption for the given image."
            )
            new_q["class"] = raw_q["category"]
            new_q["img"] = Path(raw_q["img_path"]).stem
            new_q["img_path"] = str(
                Path(".", self.DATASET_PATHS[ds_class]["imgs"], f"{new_q['img']}.jpg")
            )
            new_q["is_open"] = False
            new_q["options"] = None
            new_q["answer"] = raw_q[raw_q["answer"]]
        elif self.DATASET == "seed":
            new_q["question_id"] = raw_q["question_id"].split("_")[-1]
            new_q["question"] = raw_q["question"]
            new_q["class"] = ds_class
            new_q["img"] = Path(raw_q["img_path"]).stem
            new_q["img_path"] = str(
                Path(".", self.DATASET_PATHS[ds_class]["imgs"], f"{new_q['img']}.jpg")
            )
            new_q["is_open"] = False
            new_q["options"] = {
                "A": raw_q["choice_a"],
                "B": raw_q["choice_b"],
                "C": raw_q["choice_c"],
                "D": raw_q["choice_d"],
            }
            new_q["answer"] = raw_q["answer"]
        elif self.DATASET == "vqav2" or self.DATASET == "vqav2_holdout":
            new_q["question_id"] = raw_q["question_id"]
            new_q["question"] = raw_q["question"]
            new_q["class"] = ds_class
            new_q["img"] = Path(raw_q["img_path"]).stem
            new_q["img_path"] = str(
                Path(".", self.DATASET_PATHS[ds_class]["imgs"], f"{new_q['img']}.jpg")
            )
            new_q["is_open"] = True
            new_q["options"] = None
            new_q["answer"] = raw_q["multiple_choice_answer"]
        return new_q

    def load_llavabench_ans(self):
        ans_file_path = Path(
            "..", "data", "datasets", "llava-bench", "answers_gpt4.jsonl"
        )
        return load_jsonl(ans_file_path)

    def get_llava_answer(self, q_idx):
        ans = next(a["text"] for a in self.LLAVA_ANS if a["question_id"] == q_idx)
        return ans
