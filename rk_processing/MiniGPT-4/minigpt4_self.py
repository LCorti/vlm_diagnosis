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

from config_handlers.rk_handler import RKHandler
from rk_processing.common_utils.gen_utils import GenUtils
from utils.data_io import make_dir, load_json, save_jsonl
from utils.model_utils import (
    ask,
    get_chat_state,
    get_response,
    load_image,
    make_message,
)

PROMPT_VERSION = 5
MODEL_NAME = "minigpt4"

# ================================================================== #
# The original code can be found in demo.py.                         #
# This is a quick edit to run inference directly from command line.  #
# ================================================================== #


def parse_args():
    parser = argparse.ArgumentParser(description="RK Generation for MiniGPT4")
    parser.add_argument("--cfg-path", required=True, help="Path to configuration file.")
    parser.add_argument("--ds_name", required=True, help="Name of the dataset.")
    parser.add_argument(
        "--questions_file", required=True, help="Path to the file with questions."
    )
    parser.add_argument(
        "--do_sample", type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=int, default=0.1)
    parser.add_argument("--top_p", help="Top-P", default=0.95)
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

    # Load RK Config
    rk_hdl = RKHandler()
    rk_hdl.set_curr_model(MODEL_NAME)
    rk_hdl.set_curr_ds(ds_name)
    # Get RK output paths and
    # (1) complete file name with prompt version
    raw_out_f = str(rk_hdl.get_raw_rk_path()).format(PROMPT_VERSION)
    parsed_out_f = str(rk_hdl.get_parsed_rk_path()).format(PROMPT_VERSION)
    # (2) make directory if missing
    base_dir = Path(__file__).parent.parent.parent
    full_raw_out_f = base_dir.joinpath(raw_out_f)
    full_raw_out_dir = base_dir.joinpath(Path(raw_out_f).parent)
    make_dir(full_raw_out_dir)
    full_parsed_out_f = base_dir.joinpath(parsed_out_f)
    full_parsed_out_dir = base_dir.joinpath(Path(parsed_out_f).parent)
    make_dir(full_parsed_out_dir)

    # Load generation config and prompt templates for generation
    gen_utils = GenUtils(MODEL_NAME, prompt_version=PROMPT_VERSION)
    question_template = gen_utils.get_question_template(ds_name)
    rationale_template = gen_utils.get_rationale_template()
    out_format_template = gen_utils.get_out_format_template()

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

    # == == == == Get responses and self-explanations == == == ==
    print("Running inference and computing explanations...")
    all_rk = []
    all_parsed_rk = []

    for curr_q in questions:
        print("Current question ID: {}".format(curr_q["question_id"]))
        print("-- Text: {}".format(curr_q["question"]))

        # Prepare object to store info
        curr_rk = {"question_id": curr_q["question_id"]}

        # Loading image and conversation
        img_path = base_dir.joinpath(curr_q["img_path"])
        print("-- Loading image {}".format(img_path))
        message = make_message(question_template, curr_q)
        conv = get_chat_state()
        conv, img_list = load_image(chat, img_path, conv)

        # First generation step: get answer from the model
        conv = ask(chat, message, conv)
        curr_rk["response"] = get_response(
            chat,
            img_list,
            conv,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
        )
        print("Response: {}".format(curr_rk["response"]))
        print("=" * 25)

        # Second generation step: get unstructured rationales for model output
        conv = ask(chat, rationale_template, conv)
        curr_rk["rationales"] = get_response(
            chat,
            img_list,
            conv,
            num_beams=args.num_beams,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
        )
        # print(curr_rk["rationales"])
        # print("=" * 25)

        # Third step: triple extraction and structuring from rationales
        conv = ask(chat, out_format_template, conv)
        curr_rk["triples"] = get_response(
            chat,
            img_list,
            conv,
            num_beams=args.num_beams,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
        )
        print(curr_rk["triples"])
        print("=" * 25)

        all_rk.append(curr_rk)
        all_parsed_rk.append(gen_utils.parse_raw_rk(curr_rk))

        # Free up memory
        del img_list
        torch.cuda.empty_cache()
        # == == == == == == == == == == == == == == == == == == ==

        # Save results to file
        print("... Saving raw Really Knows ...")
        save_jsonl(all_rk, full_raw_out_f)
        print("... Saving parsed Really Knows ...")
        save_jsonl(all_parsed_rk, full_parsed_out_f)
        print("All data saved.")
