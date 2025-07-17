import sys

from pathlib import Path

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.sk_handler import SKHandler
from utils.data_io import load_pickle, make_dir
from utils.vis import save_graph_to_img

if __name__ == "__main__":
    sk_hdl = SKHandler()
    ds_list = sk_hdl.get_sk_list()
    ds_list.remove("vqav2_holdout")
    base_dir = Path(__file__).parent.parent

    for ds_name in ds_list:
        sk_hdl.set_curr_ds(ds_name)
        for ds_class in sk_hdl.get_classes():
            # Loading graphs
            sk_hdl.set_curr_class(ds_class)
            pkl_file = sk_hdl.get_pkl_path()
            graphs = load_pickle(Path("..", pkl_file))

            # Make output folder
            out_dir = base_dir.joinpath(pkl_file.parent, "graph_plots")
            make_dir(out_dir)

            # Plot
            for g in graphs:
                out_file = out_dir.joinpath(f"{g}.png")
                save_graph_to_img(graphs[g], out_file)
