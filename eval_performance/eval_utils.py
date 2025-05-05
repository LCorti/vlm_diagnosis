# Prepare the dictionary to store results
def get_eval_dict(model_list, ds_list, start=0, end=100, step=20):
    eval_dict = {}
    for model, ds in zip(model_list, ds_list):
        # Create structure
        if model not in eval_dict:
            eval_dict[model] = {}
        if ds not in eval_dict[model]:
            eval_dict[model][ds] = {}

        if ds in ["llava-bench", "mmbench"]:
            # For open-ended VQA datasets, create entries for BERTScore measures
            eval_dict[model][ds]["measures"] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
            # Prepare splits based on F1 score
            eval_dict[model][ds]["ranges"] = {}
            for r in range(start, end, step):
                eval_dict[model][ds]["ranges"][f"{r}_{r + step}"] = []
        elif ds in ["seed", "vqav2"]:
            # For open-ended VQA datasets, create entry for classic QA accuracy
            eval_dict[model][ds]["measures"] = {"accuracy": 0.0}
            # Split correct and incorrect answers
            eval_dict[model][ds]["correct_resp"] = []
            eval_dict[model][ds]["wrong_resp"] = []
        else:
            print("Something went very wrong.")

        # Add field for when the model refuses to answer or other problems emerge
        eval_dict[model][ds]["invalid_resp"] = []
    return eval_dict


# Storing response data for multiple-choice VQA
def prep_resp(q, resp):
    new_r = q.copy()
    new_r["response"] = resp
    return new_r


def cleanup_resp(resp):
    # Clear preceding and trailing spaces, new lines, etc.
    clean_resp = "".join(resp.splitlines())
    # Remove unicode characters (e.g., zero-width spaces and emojis)
    clean_resp = (clean_resp.encode("ascii", "ignore")).decode("utf-8")
    clean_resp = clean_resp.rstrip().lstrip()
    return clean_resp


# Storing response data for open-ended VQA
def prep_for_split(q, resp, measures):
    to_add = q.copy()
    to_add["response"] = resp
    to_add["measures"] = measures
    return to_add


# For reproducibility
def save_bertscore_hash(hash, path):
    with open(path, "w") as fp:
        fp.write(hash)
