import argparse
import sys
import torch

from pathlib import Path

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from utils.model_utils import (
    ask,
    get_chat_state,
    get_response,
    load_image,
    make_message,
)

module_path = str(Path("..", "..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

import utils.data_io as data_io

from config_handlers.causal_handler import CausalHandler
from config_handlers.dataset_handler import DatasetHandler
from rk_processing.utils.gen_utils import GenUtils
from rk_processing.utils.path_utils import merge_path

PROMPT_VERSION = 4
MODEL_NAME = "minigpt4"

# ================================================================== #
# The original code can be found in demo.py.                         #
# This is a quick edit to run inference directly from command line.  #
# ================================================================== #


def parse_args():
    parser = argparse.ArgumentParser(
        description="Counterfactual output generation for MiniGPT4"
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
    base_dir = Path(__file__).parent.parent.parent

    # Load dataset config
    ds_hdl = DatasetHandler()
    ds_hdl.set_curr_ds(ds_name)

    # Load handler for causal analysis
    causal_hdl = CausalHandler()
    causal_hdl.set_curr_model(MODEL_NAME)
    causal_hdl.set_curr_ds(ds_name)
    imgs_occ_paths = causal_hdl.get_imgs_occluded_path()
    imgs_occ_paths = data_io.add_path_to_surf_storage(base_dir, imgs_occ_paths)
    counter_resps_path = base_dir.joinpath(causal_hdl.get_counter_resps_path())
    data_io.make_dir(counter_resps_path.parent)

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
    questions = data_io.load_json(args.questions_file)

    # == == == == Get counterfactual responses on occluded images == == == ==
    print("Running inference on occluded images...")

    # Check if some data is already present, if so load it.
    # Only keep the question ids for later checks.
    if counter_resps_path.exists():
        all_counter_resps = data_io.load_jsonl(counter_resps_path)
        all_counter_resps = list(set([cq["question_id"] for cq in all_counter_resps]))
    else:
        all_counter_resps = []

    det = data_io.load_jsonl(imgs_occ_paths.joinpath("det_paths.jsonl"))
    for q_det in det:
        # If data for curr_q is already present, skip it.
        if q_det["question_id"] in all_counter_resps:
            continue

        # Retrieve question info
        question = next(
            (q for q in questions if q["question_id"] == q_det["question_id"]), None
        )
        if question:
            print(f"Current question ID: {q_det['question_id']}")
            print(f"-- Text: {question['question']}")
        else:
            print(f"Problem with question with ID: {q_det['question_id']}")

        curr_cr_list = []
        for subset in q_det["powerset"].values():
            for _, data in subset.items():
                # Prepare object to store info
                curr_cr = {
                    "question_id": q_det["question_id"],
                    "occluded": data["combination"],
                    "size": len(data["combination"]),
                    "path": data["path"],
                }

                # Run inference
                # Prepare message and image
                message = make_message(question_template, question)
                conv = get_chat_state()
                conv, img_list = load_image(
                    chat, merge_path(imgs_occ_paths, data["path"]), conv
                )
                # Get response
                conv = ask(chat, message, conv)
                curr_cr["response"] = get_response(
                    chat,
                    img_list,
                    conv,
                    num_repeats=1,
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )
                curr_cr_list.append(curr_cr)

                # Free up memory
                del img_list
                torch.cuda.empty_cache()

        # Saving results to file once computations for each 'size' are done
        print("... Saving data ...")
        data_io.append_to_jsonl(curr_cr_list, counter_resps_path)
        print("All data saved.")
