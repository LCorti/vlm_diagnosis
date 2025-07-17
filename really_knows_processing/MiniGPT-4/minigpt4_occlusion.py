import argparse
import sys
import torch

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from pathlib import Path

module_path = str(Path("..", "..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.dataset_config_loader import DatasetConfig
from config_loaders.rk_handler import RKHandler
from really_knows_processing.common_utils.gen_utils import GenUtils
from really_knows_processing.common_utils.path_utils import merge_path
from utils.data_io import make_dir, load_json, load_jsonl, append_to_jsonl
from utils.model_utils import (
    ask,
    get_chat_state,
    get_response,
    load_image,
    make_message,
)

PROMPT_VERSION = 4
MODEL_NAME = "minigpt4"
SURF_DRIVE = Path("data", "storage")

# ================================================================== #
# The original code can be found in demo.py.                         #
# This is a quick edit to run inference directly from command line.  #
# ================================================================== #


def parse_args():
    parser = argparse.ArgumentParser(
        description="Counterfacual generation for MiniGPT4"
    )
    parser.add_argument("--cfg-path", required=True, help="Path to configuration file.")
    parser.add_argument("--ds_name", required=True, help="Name of the dataset.")
    parser.add_argument(
        "--questions_file", required=True, help="Path to the file with questions."
    )
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=int, default=0.1)
    parser.add_argument(
        "--max_new_tokens", help="Maximum number of tokens to generate", default=256
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="Specify the gpu to load the model."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse args
    args = parse_args()
    ds_name = args.ds_name
    questions_file = args.questions_file

    # Load dataset config
    ds_config = DatasetConfig()
    ds_paths = ds_config.get_ds_paths(ds_name)

    # Load RK Config
    rk_hdl = RKHandler()
    rk_hdl.set_curr_model(MODEL_NAME)
    rk_hdl.set_curr_ds(ds_name)
    counter_out_f = rk_hdl.get_counterfactual_rk_path()
    # Make directory if missing
    base_dir = Path(__file__).parent.parent.parent
    full_counter_out_f = base_dir.joinpath(counter_out_f)
    full_counter_out_dir = base_dir.joinpath(Path(full_counter_out_f).parent)
    make_dir(full_counter_out_dir)

    # Load generation config and prompt templates for generation
    gen_utils = GenUtils(MODEL_NAME, prompt_version=PROMPT_VERSION)
    question_template = gen_utils.get_question_template(ds_name)

    # Load model
    print("Loading model...")
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to("cuda:{}".format(args.gpu_id))
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
        vis_processor_cfg
    )
    chat = Chat(model, vis_processor, device="cuda:{}".format(args.gpu_id))
    print("Model loaded.")

    # Load data
    print("Loading data...")
    print("Loading questions from {}".format(questions_file))
    questions = load_json(args.questions_file)

    # == == == == Get counterfactual responses on occluded images == == == ==
    print("Running inference on occluded images...")

    # Check if some data is already present, if so load it.
    # Only keep the question ids for later checks.
    if full_counter_out_f.exists():
        all_resp_counter = load_jsonl(full_counter_out_f)
        all_resp_counter = [curr_q["question_id"] for curr_q in all_resp_counter]
    else:
        all_resp_counter = []

    for curr_q in questions:
        # If data for curr_q is already present, skip it.
        if curr_q["question_id"] in all_resp_counter:
            continue

        print("Current question ID: {}".format(curr_q["question_id"]))
        print("-- Text: {}".format(curr_q["question"]))

        ds_class = curr_q["class"]
        curr_dir = SURF_DRIVE.joinpath(ds_paths[ds_class]["imgs_occluded"])
        path_to_summary = f"{curr_dir.parent}/summary.json"
        summary = load_json(path_to_summary)
        occlusion_data = summary[ds_class][str(curr_q["question_id"])]["occlusion"]
        # curr_dir_imgs = curr_dir.joinpath(curr_q["question_id"])
        print(f">> Found {len(occlusion_data)} occluded images to process.")
        curr_subset = []

        for curr_occ in occlusion_data:
            # Prepare object to store info
            curr_occ_res = {
                "question_id": curr_q["question_id"],
                "size_treatment": len(curr_occ["occluded_concepts"]),
                "occluded_concepts": curr_occ["occluded_concepts"],
                "path": curr_occ["path"],
            }

            # Load image
            image_path = merge_path(str(curr_dir), curr_occ["path"])
            message = make_message(question_template, curr_q)
            conv = get_chat_state()
            conv, img_list = load_image(chat, image_path, conv)

            # First generation step: get answer from the model
            conv = ask(chat, message, conv)
            curr_occ_res["response"] = get_response(
                chat,
                img_list,
                conv,
                num_repeats=1,
                num_beams=args.num_beams,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )

            all_resp_counter.append(curr_occ_res)
            curr_subset.append(curr_occ_res)

            # Free up memory
            del img_list
            torch.cuda.empty_cache()
            # == == == == == == == == == == == == == == == == == == ==

        # Saving results to file once computations for each 'size' are done
        print("... Saving data ...")
        append_to_jsonl(curr_subset, full_counter_out_f)
        print("All data saved.")
