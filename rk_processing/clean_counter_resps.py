import sys

from pathlib import Path

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.causal_handler import CausalHandler
from config_handlers.dataset_handler import DatasetHandler
from utils.data_io import load_jsonl, save_jsonl
from utils.text_utils import clean_response


def get_options(question_id, ds_hdl):
    for ds_class in ds_hdl.get_classes():
        ds_hdl.set_curr_class(ds_class)
        q_path = ds_hdl.get_sampled_questions_path()
        questions = load_jsonl(Path("..", q_path))
        match_q = next((q for q in questions if q["question_id"] == question_id), None)
        if match_q:
            return match_q["options"] if "options" in match_q else None


if __name__ == "__main__":
    ds_hdl = DatasetHandler()
    causal_hdl = CausalHandler()
    model_list = causal_hdl.get_model_list()
    ds_list = ds_hdl.get_ds_list()
    all_models = model_list * 4
    all_models.sort()
    all_ds = ds_list * 4

    for model, ds in zip(all_models, all_ds):
        ds_hdl.set_curr_ds(ds)
        causal_hdl.set_curr_model(model)
        causal_hdl.set_curr_ds(ds)
        counter_resp_path = Path("..", causal_hdl.get_counter_resps_path())
        try:
            counter_resps = load_jsonl(counter_resp_path)
            print(f"Processing data for {model} + {ds}.")
        except Exception as _:
            print(f"Data not found for {model} + {ds}.")
            continue
        oe = ds in ["llava-bench", "mmbench"]

        new_cr = []
        for cr in counter_resps:
            options = get_options(cr["question_id"], ds_hdl)
            cr["response"] = clean_response(
                cr["response"], ds, open_ended=oe, mc_options=options
            )
            new_cr.append(cr)

        # Save after fixing all responses
        save_jsonl(new_cr, counter_resp_path)
