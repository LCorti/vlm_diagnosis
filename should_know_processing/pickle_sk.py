import os
import sys

from pathlib import Path

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.sk_config_loader import SKConfig
from utils.data_io import load_jsonl, make_dir, save_pickle
from utils.graph_utils import make_nx_graph

if __name__ == "__main__":
    # Load SK config
    sk_config = SKConfig()
    ds_list = sk_config.get_sk_list()
    ds_list.remove("vqav2_holdout")

    # Open jsonl from crowdsourcing
    # Convert to NX graph (probably add graph utils)
    # Save to pickle from path

    for dataset in ds_list:
        print(f"Looking at {dataset}")
        # Get current paths
        sk_paths = sk_config.get_sk_paths(dataset)

        for ds_class in sk_paths:
            print(f"- Looking at {ds_class}")
            # Load merged scene graphs
            base_dir = sk_paths[ds_class]["dir"]
            crowd_sg_file = sk_paths[ds_class]["sg_crowd"]
            crowd_sg = load_jsonl(Path("..").joinpath(base_dir).joinpath(crowd_sg_file))
            graph_dict = {}

            # Go through current batch of SKs and make NX graphs for thema
            for sk in crowd_sg:
                sk_graph = make_nx_graph(sk["rel_clusters_unique"])
                graph_dict[sk["img_id"]] = sk_graph

            # Make directory if needed and save graph to it
            out_file = Path("..").joinpath(base_dir).joinpath(sk_paths[ds_class]["pkl"])
            out_folder = out_file.parent
            make_dir(out_folder)
            save_pickle(graph_dict, out_file)
            print("Graphs saved.")
