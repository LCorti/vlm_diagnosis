import argparse
import os
import sys
import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from pathlib import Path

module_path = os.path.abspath(os.path.join("../../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.rk_config_loader import RKConfig
from utils.data_io import make_dir, load_json, save_jsonl
from utils.image_utils import load_image
from utils.model_utils import (
    set_conv_mode,
    make_message,
    get_conv_template,
    add_conv_step,
)
from really_knows_processing.common_utils.gen_utils import GenUtils

PROMPT_VERSION = 4
MODEL_NAME = "llava-1.6"


# Kwargs parser
def parse_args():
    parser = argparse.ArgumentParser(description="RK Generation for LLaVa-1.6.")
    parser.add_argument("--ds_name", required=True, help="Name of the dataset.")
    parser.add_argument(
        "--questions_file", required=True, help="Path to the file with questions."
    )
    parser.add_argument("--do_sample", default=False)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", help="Model temperature.", default=0.1)
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
    rk_config = RKConfig()
    # Get RK output paths and
    # (1) complete file name with prompt version
    out_file = rk_config.get_rk_paths(MODEL_NAME, ds_name)
    out_file = out_file.format(PROMPT_VERSION)
    # (2) make directory if missing
    base_dir = Path(__file__).parent.parent.parent
    full_out_file = base_dir.joinpath(out_file)
    full_out_dir = base_dir.joinpath(Path(out_file).parent)
    make_dir(full_out_dir)

    # Load generation config
    gen_utils = GenUtils(MODEL_NAME, prompt_version=PROMPT_VERSION)
    gen_config = gen_utils.get_gen_config(
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        temperature=args.temperature,
        use_cache=args.use_cache,
        max_new_tokens=args.max_new_tokens,
    )
    # Get prompt templates for generation
    question_template = gen_utils.get_question_template(ds_name)
    rationale_template = gen_utils.get_rationale_template()
    out_format_template = gen_utils.get_out_format_template()

    # Load model
    print("Loading model...")
    # Kept from original implementation
    disable_torch_init()
    hf_model_path = "liuhaotian/llava-v1.6-vicuna-7b"
    model_base = None
    model_name = get_model_name_from_path(hf_model_path)
    conv_mode = set_conv_mode(model_name)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=hf_model_path, model_base=model_base, model_name=model_name
    )
    model.half()
    # print(getattr(model.config, "image_aspect_ratio", None))
    print("Model loaded.")

    # Load data
    print("Loading data...")
    print("Loading questions from {}".format(questions_file))
    questions = load_json(questions_file)

    # == == == == Get attributions over input image == == == ==
    print("Running inference and computing explanations...")
    all_model_rk = []

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    for curr_q in questions:
        print("Current question ID: {}".format(curr_q["question_id"]))
        print("-- Text: {}".format(curr_q["question"]))

        # Prepare object to store info
        model_rk = {"question_id": curr_q["question_id"]}

        # Loading image
        img_path = base_dir.joinpath(curr_q["img_path"])
        print("-- Loading image {}".format(img_path))
        image = load_image(img_path)
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
        # Prepare input token ids
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        # First generation step: get answer from the model
        model_rk["response"] = gen_utils.gen_response_llava_next(
            model, tokenizer, image_tensor, input_ids
        )
        print("Response: {}".format(model_rk["response"]))
        print("=" * 25)

        # Second generation step: get unstructured rationales for model output
        conv = add_conv_step(conv, rationale_template, prev_resp=model_rk["response"])
        prompt = conv.get_prompt()
        # Prepare input token ids
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        model_rk["rationales"] = gen_utils.gen_response_llava_next(
            model, tokenizer, image_tensor, input_ids
        )
        # print(model_rk['rationales'])
        # print('='*25)

        # Third step: triple extraction and structuring from rationales
        conv = add_conv_step(
            conv, out_format_template, prev_resp=model_rk["rationales"]
        )
        prompt = conv.get_prompt()
        # Prepare input token ids
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        model_rk["triples"] = gen_utils.gen_response_llava_next(
            model, tokenizer, image_tensor, input_ids
        )
        print(model_rk["triples"])
        print("=" * 25)

        all_model_rk.append(model_rk)

        # Free up memory
        del image_tensor
        del input_ids
        torch.cuda.empty_cache()
        # == == == == == == == == == == == == == == == == == == ==

        # Saving results to file
        print("... Saving data ...")
        save_jsonl(all_model_rk, full_out_file)
        print("Data saved.")
