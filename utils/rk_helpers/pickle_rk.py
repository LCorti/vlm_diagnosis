import sys

from pathlib import Path

module_path = str(Path("..", "..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.rk_handler import RKHandler
from utils import data_io, graph_utils

if __name__ == "__main__":
    rk_hdl = RKHandler()
    model_list = rk_hdl.get_model_list()
    prompt_version = 4

    for model in model_list:
        rk_hdl.set_curr_model(model)
        for ds in rk_hdl.get_ds_list():
            rk_hdl.set_curr_ds(ds)
            input_path = str(rk_hdl.get_parsed_rk_path()).format(prompt_version)
            parsed_rk = data_io.load_jsonl(Path("..", "..", input_path))
            out_file = Path("..", "..", rk_hdl.get_parsed_pkl_path())
            data_io.make_dir(out_file.parent)

            graphs = {}
            for rk in parsed_rk:
                graphs[rk["question_id"]] = graph_utils.rk_to_nx(rk["triple_objs"])

            data_io.save_pickle(graphs, out_file)
