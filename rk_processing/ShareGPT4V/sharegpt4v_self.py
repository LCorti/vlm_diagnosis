import argparse
import sys
import torch

from pathlib import Path

from share4v.constants import IMAGE_TOKEN_INDEX
from share4v.mm_utils import get_model_name_from_path, tokenizer_image_token
from share4v.model.builder import load_pretrained_model
from share4v.utils import disable_torch_init
from utils.model_utils import (
    add_conv_step,
    get_conv_template,
    get_stop_str,
    get_stopping_criteria,
    make_message,
    set_conv_mode,
)

module_path = str(Path("..", "..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

import utils.data_io as data_io
import utils.image_utils as image_utils

from config_handlers.rk_handler import RKHandler
from rk_processing.utils.gen_utils import GenUtils


PROMPT_VERSION = 4
MODEL_NAME = "sharegpt4v"
HF_MODEL_NAME = "Lin-Chen/ShareGPT4V-7B"


# Kwargs parser
def parse_args():
    parser = argparse.ArgumentParser(description="RK Generation for ShareGPT4V")
    parser.add_argument("--ds_name", required=True, help="Name of the dataset.")
    parser.add_argument(
        "--questions_file", required=True, help="Path to the file with questions."
    )
    parser.add_argument(
        "--do_sample", type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", help="Model temperature.", default=0.1)
    parser.add_argument("--top_k", help="Top-K", default=10)
    parser.add_argument("--top_p", help="Top-P", default=0.95)
    parser.add_argument(
        "--max_new_tokens", help="Maximum number of tokens to generate", default=256
    )
    parser.add_argument("--use_cache", default=True)
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
    raw_out_f = str(rk_hdl.get_rk_raw_path()).format(PROMPT_VERSION)
    parsed_out_f = str(rk_hdl.get_rk_parsed_path()).format(PROMPT_VERSION)
    # (2) make directory if missing
    base_dir = Path(__file__).parent.parent.parent
    full_raw_out_f = base_dir.joinpath(raw_out_f)
    full_raw_out_dir = base_dir.joinpath(Path(raw_out_f).parent)
    data_io.make_dir(full_raw_out_dir)
    full_parsed_out_f = base_dir.joinpath(parsed_out_f)
    full_parsed_out_dir = base_dir.joinpath(Path(parsed_out_f).parent)
    data_io.make_dir(full_parsed_out_dir)

    # Load generation config
    gen_utils = GenUtils(MODEL_NAME, prompt_version=PROMPT_VERSION)
    gen_config = gen_utils.get_gen_config(
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        temperature=args.temperature,
        use_cache=args.use_cache,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    # Get prompt templates for generation
    question_template = gen_utils.get_question_template(ds_name)
    rationale_template = gen_utils.get_rationale_template()
    out_format_template = gen_utils.get_out_format_template()

    # Load model
    print("Loading model...")
    # Kept from original implementation
    disable_torch_init()
    model_base = None
    model_name = get_model_name_from_path(HF_MODEL_NAME)
    conv_mode = set_conv_mode(model_name)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=HF_MODEL_NAME, model_base=model_base, model_name=model_name
    )
    model.half()
    # print(getattr(model.config, "image_aspect_ratio", None))
    print("Model loaded.")

    # Load data
    print("Loading data...")
    print("Loading questions from {}".format(questions_file))
    questions = data_io.load_json(questions_file)

    # == == == == Get responses and self-explanations == == == ==
    print("Running inference and computing explanations...")
    all_rk = []
    all_parsed_rk = []

    for curr_q in questions:
        print("Current question ID: {}".format(curr_q["question_id"]))
        print("-- Text: {}".format(curr_q["question"]))

        # Prepare object to store info
        curr_rk = {"question_id": curr_q["question_id"]}

        # Loading image
        img_path = base_dir.joinpath(curr_q["img_path"])
        print("-- Loading image {}".format(img_path))
        image = image_utils.load_img(img_path)
        image_tensor = (
            image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            .half()
            .to("cuda:0")
        )

        # Format text input
        message = make_message(model, question_template, curr_q)
        conv = get_conv_template(conv_mode)
        conv = add_conv_step(conv, message)
        prompt = conv.get_prompt()
        stop_str = get_stop_str(conv)
        # Prepare input token ids
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stopping_criteria = get_stopping_criteria(input_ids, stop_str, tokenizer)

        # First generation step: get answer from the model
        curr_rk["response"] = gen_utils.generate_sharegpt4v(
            model, tokenizer, image_tensor, input_ids, stopping_criteria, stop_str
        )
        print("Response: {}".format(curr_rk["response"]))
        print("=" * 25)

        # Second generation step: get unstructured rationales for model output
        conv = add_conv_step(conv, rationale_template, prev_resp=curr_rk["response"])
        prompt = conv.get_prompt()
        stop_str = get_stop_str(conv)
        # Prepare input token ids
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stopping_criteria = get_stopping_criteria(input_ids, stop_str, tokenizer)
        curr_rk["rationales"] = gen_utils.generate_sharegpt4v(
            model, tokenizer, image_tensor, input_ids, stopping_criteria, stop_str
        )
        # print(model_rk['rationales'])
        # print('='*25)

        # Third step: triple extraction and structuring from rationales
        conv = add_conv_step(conv, out_format_template, prev_resp=curr_rk["rationales"])
        prompt = conv.get_prompt()
        stop_str = get_stop_str(conv)
        # Prepare input token ids
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stopping_criteria = get_stopping_criteria(input_ids, stop_str, tokenizer)
        curr_rk["triples"] = gen_utils.generate_sharegpt4v(
            model, tokenizer, image_tensor, input_ids, stopping_criteria, stop_str
        )
        print(curr_rk["triples"])
        print("=" * 25)

        all_rk.append(curr_rk)
        all_parsed_rk.append(gen_utils.parse_raw_rk(curr_rk))

        # Free up memory
        del image_tensor
        del input_ids
        torch.cuda.empty_cache()
        # == == == == == == == == == == == == == == == == == == ==

        # Saving results to file
        print("... Saving raw Really Knows ...")
        data_io.save_jsonl(all_rk, full_raw_out_f)
        print("... Saving parsed Really Knows ...")
        data_io.save_jsonl(all_parsed_rk, full_parsed_out_f)
        print("All data saved.")
