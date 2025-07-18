import sys

from pathlib import Path

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.sk_handler import SKHandler
from utils.data_io import load_jsonl, make_dir, save_pickle
from utils.graph_utils import raw_sk_to_nx

if __name__ == "__main__":
    # Load SK config
    sk_hdl = SKHandler()
    ds_list = sk_hdl.get_ds_list()
    if "vqav2_holdout" in ds_list:
        ds_list.remove("vqav2_holdout")

    # Open jsonl from crowdsourcing
    # Convert to NX graph (probably add graph utils)
    # Save to pickle from path

    for dataset in ds_list:
        print(f"Looking at {dataset}")
        sk_hdl.set_curr_ds(dataset)

        for ds_class in sk_hdl.get_classes():
            print(f"- Looking at {ds_class}")
            # Load merged scene graphs
            sk_hdl.set_curr_class(ds_class)
            crowd_sg = load_jsonl(Path("..", sk_hdl.get_sg_crowd_path()))
            graph_dict = {}

            # Go through current batch of SKs and make NX graphs for thema
            for sk in crowd_sg:
                sk_graph = raw_sk_to_nx(sk["rel_clusters_unique"])
                graph_dict[sk["img_id"]] = sk_graph

            # Make directory if needed and save graph to it
            out_file = sk_hdl.get_pkl_path()
            out_folder = Path("..", out_file.parent)
            make_dir(out_folder)
            save_pickle(graph_dict, out_file)
            print("Graphs saved.")
