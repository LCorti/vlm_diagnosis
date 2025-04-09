import os
import sys

from pathlib import Path

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.sk_config_loader import SKConfig
from utils.data_io import load_pickle, make_dir
from utils.vis import save_graph_to_img

if __name__ == "__main__":
    sk_config = SKConfig()
    ds_list = sk_config.get_sk_list()
    ds_list.remove("vqav2_holdout")
    base_dir = Path(__file__).parent.parent

    for ds_name in ds_list:
        ds_paths = sk_config.get_sk_paths(ds_name)
        for ds_class in ds_paths:
            # Loading graphs
            main_dir = ds_paths[ds_class]["dir"]
            graphs_pkl = ds_paths[ds_class]["pkl"]
            path_pkl = base_dir.joinpath(main_dir).joinpath(graphs_pkl).resolve()
            graphs = load_pickle(path_pkl)

            # Make output folder
            out_dir = base_dir.joinpath(main_dir).joinpath("./graph_plots").resolve()
            make_dir(out_dir)

            # Plot
            for g in graphs:
                out_file = out_dir.joinpath(f"{g}.png")
                save_graph_to_img(graphs[g], out_file)
