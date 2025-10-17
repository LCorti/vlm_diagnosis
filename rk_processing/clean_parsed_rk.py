import sys

from pathlib import Path

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.dataset_handler import DatasetHandler
from config_handlers.rk_handler import RKHandler
from utils.data_io import load_jsonl, save_jsonl
from utils.text_utils import clean_response

PROMPT_VERSION = 4

if __name__ == "__main__":
    rk_hdl = RKHandler()
    ds_hdl = DatasetHandler()

    ds_list = ds_hdl.get_ds_list()
    model_list = rk_hdl.get_model_list()

    all_models = model_list * len(ds_list)
    all_models.sort()
    all_ds = ds_list * len(model_list)

    for model, ds in zip(all_models, all_ds):
        print(f"Processing {model} on {ds}...")
        # Load responses for current model + dataset combo
        rk_hdl.set_curr_model(model)
        rk_hdl.set_curr_ds(ds)
        rk_path = Path("..", str(rk_hdl.get_rk_parsed_path()).format(PROMPT_VERSION))
        rk_data = load_jsonl(rk_path)
        oe = ds in ["llava-bench", "mmbench"]
        # Iterate through datasets' paths
        ds_hdl.set_curr_ds(ds)
        new_rk = []
        for ds_class in ds_hdl.get_classes():
            # Load questions
            ds_hdl.set_curr_class(ds_class)
            q_path = ds_hdl.get_sampled_questions_path()
            questions = load_jsonl(Path("..", q_path))

            for q in questions:
                options = q["options"] if "options" in q else None
                curr_rk = next(
                    (rk for rk in rk_data if rk["question_id"] == q["question_id"]),
                    None,
                )

                if not curr_rk:
                    continue

                curr_rk["response_clean"] = clean_response(
                    curr_rk["response"], ds, open_ended=oe, mc_options=options
                )
                new_rk.append(curr_rk)

        # Save after fixing all responses in rk
        save_jsonl(new_rk, rk_path)
