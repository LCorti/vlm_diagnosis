import json
import numpy as np
import pandas as pd
import pickle

from pathlib import Path
from typing import Any


# == == == == == == == == == == == == == == == ==
# Handling CSV files
# == == == == == == == == == == == == == == == ==
def read_csv(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path, sep=";")
    data.drop("Unnamed: 0", axis=1, inplace=True)
    return data


# == == == == == == == == == == == == == == == ==
# Handling JSON files
# == == == == == == == == == == == == == == == ==
def load_json(file_path: str | Path) -> dict:
    with open(file_path, "r") as fp:
        json_data = json.load(fp)
    return json_data


def save_json(data: dict, file_path: str | Path) -> None:
    with open(file_path, "w") as fp:
        json.dump(data, fp, indent=2)


def load_jsonl(file_path: str | Path) -> list[dict]:
    with open(file_path, "r") as fp:
        data = [json.loads(line) for line in fp]
    return data


def save_jsonl(data: list[dict], file_path: str) -> None:
    with open(file_path, "w") as fp:
        fp.write("\n".join(map(json.dumps, data)))


def append_to_jsonl(data: dict, file_path: str) -> None:
    with open(file_path, "a") as fp:
        fp.write("\n".join(map(json.dumps, data)))
        fp.write("\n")


def np_encoder(obj: Any) -> dict:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()


# == == == == == == == == == == == == == == == ==
# Directory management
# == == == == == == == == == == == == == == == ==
def make_dir(dir_path: str | Path) -> None:
    try:
        Path(dir_path).mkdir(parents=True)
    except FileExistsError as f_exception:
        print(f"{f_exception}")


# == == == == == == == == == == == == == == == ==
# Handling NX graphs
# == == == == == == == == == == == == == == == ==
def save_pickle(data: Any, file_path: str) -> None:
    with open(file_path, "wb") as fp:
        pickle.dump(data, fp)


def load_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as fp:
        data = pickle.load(fp, encoding="utf-8")
    return data


# == == == == == == == == == == == == == == == ==
# Handling Model Responses
# == == == == == == == == == == == == == == == ==
def load_eval_responses(model: str, ds: str, prompt_version: int) -> list[dict]:
    file_path = Path(
        "..", "really_knows", model, f"results_v{prompt_version}", f"exp_{ds}_rk.jsonl"
    )
    return load_jsonl(file_path)


def load_counter_responses(model: str, ds: str) -> list[dict]:
    file_path = Path("..", "really_knows", model, "res_counter", f"res_{ds}_c_rk.jsonl")
    return load_jsonl(file_path)


# == == == == == == == == == == == == == == == ==
# Handling Statistics Results
# == == == == == == == == == == == == == == == ==
def save_stats(file_path: str, stats_obj: dict) -> None:
    with open(file_path, "w") as fp:
        json.dump(stats_obj, fp, indent=2)
