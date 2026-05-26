import sys

from pathlib import Path

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.sk_handler import SKHandler
from utils.data_io import load_jsonl, make_dir, save_pickle
from utils.graph_utils import sk_to_nx

if __name__ == "__main__":
    # Load SK config
    sk_hdl = SKHandler()
    ds_list = sk_hdl.get_ds_list()

    # Open jsonl from crowdsourcing
    # Convert to NX graph (probably add graph utils)
    # Save to pickle from path

    for dataset in ds_list:
        print(f"Looking at {dataset}")
        sk_hdl.set_curr_ds(dataset)

        for ds_class in sk_hdl.get_classes():
            print(f"- Looking at {ds_class}")
            sk_hdl.set_curr_class(ds_class)

            # Load merged scene graphs
            crowd_sg_path = Path("..", sk_hdl.get_sg_crowd_path())
            crowd_sg = None
            print(f"Attempting loading merged scene graphs from: {str(crowd_sg_path)}")
            if crowd_sg_path.exists():
                crowd_sg = load_jsonl(crowd_sg_path)
                crowd_sg_dict = {}
                # Go through current batch of SKs and make NX graphs for them
                for sk in crowd_sg:
                    sk_graph = sk_to_nx(sk["rel_clusters_unique"])
                    crowd_sg_dict[sk["img_id"]] = sk_graph

                # Make directory if needed and save graph to it
                out_file = sk_hdl.get_crowd_pkl_path()
                make_dir(out_file.parent)
                save_pickle(crowd_sg_dict, out_file)
                print(f"Crowd SKs saved to {str(out_file)}")
            else:
                print(f"... Unable to find {str(crowd_sg_path)}")

            # Load final SKs
            exp_sk_path = Path("..", sk_hdl.get_sk_exp_path())
            exp_sk = None
            if exp_sk_path.exists():
                exp_sk = load_jsonl(exp_sk_path)
                exp_sk_dict = {}
                for sk in exp_sk:
                    sk_graph = sk_to_nx(sk["relations"])
                    exp_sk_dict[sk["question_id"]] = sk_graph

                # Save
                out_file = Path("..", sk_hdl.get_exp_pkl_path())
                make_dir(out_file.parent)
                save_pickle(exp_sk_dict, out_file)
                print(f"Expert SKs saved to {str(out_file)}")
            else:
                print(f"... Unable to find {str(exp_sk_path)}")
