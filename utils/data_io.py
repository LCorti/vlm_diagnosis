import json
import pandas as pd
import pickle

from pathlib import Path


# == == == == == == == == == == == == == == == ==
# Handling CSV files
# == == == == == == == == == == == == == == == ==
def read_csv(file_path):
    data = pd.read_csv(file_path, sep=";")
    data.drop("Unnamed: 0", axis=1, inplace=True)
    return data


# == == == == == == == == == == == == == == == ==
# Handling JSON files
# == == == == == == == == == == == == == == == ==
def load_json(file_path):
    with open(file_path, "r") as fp:
        json_data = json.load(fp)
    return json_data


def save_json(data, file_path):
    with open(file_path, "w") as fp:
        json.dump(data, fp, indent=2)


def load_jsonl(file_path):
    with open(file_path, "r") as fp:
        data = [json.loads(line) for line in fp]
    return data


def save_jsonl(data, file_path):
    with open(file_path, "w") as fp:
        fp.write("\n".join(map(json.dumps, data)))


def append_to_jsonl(data, file_path):
    with open(file_path, "a") as fp:
        fp.write("\n".join(map(json.dumps, data)))
        fp.write("\n")


# == == == == == == == == == == == == == == == ==
# Directory management
# == == == == == == == == == == == == == == == ==
def make_dir(dir_path):
    try:
        Path(dir_path).mkdir(parents=True)
    except FileExistsError as f_exception:
        print(f"{f_exception}")


# == == == == == == == == == == == == == == == ==
# Handling NX graphs
# == == == == == == == == == == == == == == == ==
def save_pickle(data, file_path):
    with open(file_path, "wb") as fp:
        pickle.dump(data, fp)


def load_pickle(file_path):
    with open(file_path, "rb") as fp:
        data = pickle.load(fp, encoding="utf-8")
    return data


# == == == == == == == == == == == == == == == ==
# Handling Model Responses
# == == == == == == == == == == == == == == == ==
def load_eval_responses(model, ds, prompt_version):
    file_path = Path(
        "..", "really_knows", model, f"results_v{prompt_version}", f"exp_{ds}_rk.jsonl"
    )
    return load_jsonl(file_path)


def load_counter_responses(model, ds):
    file_path = Path("..", "really_knows", model, "res_counter", f"res_{ds}_c_rk.jsonl")
    return load_jsonl(file_path)


# == == == == == == == == == == == == == == == ==
# Handling Statistics Results
# == == == == == == == == == == == == == == == ==
def save_stats(file_path, stats_obj):
    with open(file_path, "w") as fp:
        json.dump(stats_obj, fp, indent=2)
