import sys

from pathlib import Path

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.rk_handler import RKHandler
from utils.data_io import load_jsonl, save_jsonl

PROMPT_VERSION = 4

if __name__ == "__main__":
    rk_hdl = RKHandler()
    model_list = rk_hdl.get_model_list()
    rk_hdl.set_curr_model(model_list[0])
    ds_list = rk_hdl.get_ds_list()

    all_models = model_list * len(ds_list)
    all_models.sort()
    all_ds = ds_list * len(model_list)

    for model, ds in zip(all_models, all_ds):
        print(f"Processing {model} on {ds}...")
        # Load responses for current model + dataset combo
        rk_hdl.set_curr_model(model)
        rk_hdl.set_curr_ds(ds)
        rk_parsed_path = Path(
            "..", str(rk_hdl.get_rk_parsed_path()).format(PROMPT_VERSION)
        )
        rk_final_path = Path("..", rk_hdl.get_rk_final_path())
        rk_parsed = load_jsonl(rk_parsed_path)
        rk_final = load_jsonl(rk_final_path)

        for rk in rk_final:
            corr_rk = next(
                p for p in rk_parsed if p["question_id"] == rk["question_id"]
            )
            rk["response"] = corr_rk["response_clean"]
        save_jsonl(rk_final, rk_final_path)
