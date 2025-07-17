import json
import torch
import argparse

from pathlib import Path

from minigpt4.common.config import Config

# from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# ================================================================== #
# The original code can be found in demo.py.                         #
# This is a quick edit to run inference directly from command line.  #
# ================================================================== #


def parse_args():
    parser = argparse.ArgumentParser(description="minigpt4_inference")
    parser.add_argument("--cfg-path", required=True, help="Path to configuration file.")
    parser.add_argument("--ds_name", required=True, help="Name of the experiment.")
    parser.add_argument(
        "--ds_folder", required=True, help="Path to the dataset folder."
    )
    parser.add_argument(
        "--questions_file", required=True, help="Path to the file with questions."
    )
    parser.add_argument(
        "--out_dir",
        help="Name of the output file with responses from the LLM.",
        default="results",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="Specify the gpu to load the model."
    )
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=int, default=0.2)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def get_chat_state():
    return CONV_VISION_Vicuna0.copy()


def load_img(chat, img_path, chat_state):
    if img_path is None:
        print("Image not found")
        return

    img_list = []
    _ = chat.upload_img(img_path, chat_state, img_list)
    chat.encode_img(img_list)
    return chat_state, img_list


def ask(chat, user_message, chat_state):
    if len(user_message) == 0:
        print("No message received!")
        return
    chat.ask(user_message, chat_state)
    return chat_state


def get_answers(chat, img_list, chat_state, num_repeats=1, num_beams=1, temperature=1):
    all_responses = []
    for _ in range(num_repeats):
        response = chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=300,
            max_length=2000,
        )[0]
        all_responses.append(response)
    return all_responses


def main():
    print("Initializing model...")
    args = parse_args()
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
    print("Initialization finished")

    print("Loading data...")

    print("Loading questions from {}".format(args.questions_file))
    questions = []
    with open(args.questions_file, "r") as f:
        questions = json.load(f)

    print("Running inference...")

    all_responses = []

    for q in questions:
        print("Current question ID: {}".format(q["question_id"]))
        print("-- Text: {}".format(q["question"]))
        user_message = q["question"]
        img_path = Path(args.ds_folder, q["img_path"])
        print("-- Loading image {}".format(img_path))

        chat_state = get_chat_state()
        chat_state, img_list = load_img(chat, img_path, chat_state)
        chat_state = ask(chat, user_message, chat_state)

        response = get_answers(
            chat,
            img_list,
            chat_state,
            num_repeats=1,
            num_beams=args.num_beams,
            temperature=args.temperature,
        )

        all_responses.append(
            {
                "question_id": q["question_id"],
                "question": q["question"],
                "response": response,
            }
        )

        # Free up memory
        del img_list
        torch.cuda.empty_cache()

    # Save responses to file

    res_folder = args.out_dir
    Path(res_folder).mkdir(parents=True)

    with open(Path(res_folder, f"exp_{args.ds_name}_responses.jsonl"), "w") as f:
        for r in all_responses:
            json.dump(r, f)
            f.write("\n")


if __name__ == "__main__":
    main()
