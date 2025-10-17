import argparse
import sys
import torch

from pathlib import Path
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from utils.model_utils import make_message, add_conv_step

module_path = str(Path("..", "..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

import utils.data_io as data_io
import utils.image_utils as image_utils

from config_handlers.rk_handler import RKHandler
from rk_processing.utils.gen_utils import GenUtils

PROMPT_VERSION = 4
MODEL_NAME = "qwen2_5_vl"
HF_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description="RK Generation for InternVL2.")
    parser.add_argument("--ds_name", required=True, help="Name of the dataset.")
    parser.add_argument(
        "--questions_file", required=True, help="Path to the file with questions."
    )
    parser.add_argument(
        "--do_sample", type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--temperature", help="Model temperature.", default=0.1)
    parser.add_argument("--top_k", help="Top-K", default=10)
    parser.add_argument("--top_p", help="Top-P", default=0.95)
    parser.add_argument(
        "--max_new_tokens", help="Maximum tokens to generate.", default=256
    )
    parser.add_argument("--use_cache", default=True)
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

    # Load RK config
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
        temperature=args.temperature,
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        HF_MODEL_NAME, torch_dtype="auto", device_map="auto"
    )
    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token
    # range of 256-1280, to balance performance and cost.
    min_pixels = 4 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        HF_MODEL_NAME, min_pixels=min_pixels, max_pixels=max_pixels
    )
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
        print(f"Current question ID: {curr_q['question_id']}")
        print(f"-- Text: {curr_q['question']}")

        # Prepare object to store info
        curr_rk = {"question_id": curr_q["question_id"]}

        # Loading image
        img_path = base_dir.joinpath(curr_q["img_path"])
        print("-- Loading image {}".format(img_path))
        image = image_utils.load_img(img_path)

        # Process inputs
        conv = []
        # Prepare actual question
        curr_message_text = make_message(question_template, curr_q)
        # Add start message
        conv = add_conv_step(conv, curr_message_text, image)
        text = processor.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(conv)  # Ignore video_inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # First generation step: get answer from the model
        curr_rk["response"] = gen_utils.generate_qwen2_5_vl(model, processor, inputs)
        print(curr_rk["response"])
        print("=" * 25)

        # Second generation step: get unstructured rationales for model output
        conv = add_conv_step(
            conv, rationale_template, image, prev_resp=curr_rk["response"]
        )
        text = processor.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        curr_rk["rationales"] = gen_utils.generate_qwen2_5_vl(model, processor, inputs)
        # print(model_rk['rationales'])
        # print('='*25)

        # Third step: triple extraction and structuring from rationales
        conv = add_conv_step(
            conv, out_format_template, image, prev_resp=curr_rk["response"]
        )
        text = processor.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        curr_rk["triples"] = gen_utils.generate_qwen2_5_vl(model, processor, inputs)
        print(curr_rk["triples"])
        print("=" * 25)

        all_rk.append(curr_rk)
        all_parsed_rk.append(gen_utils.parse_raw_rk(curr_rk))

        # Free up memory
        del inputs
        torch.cuda.empty_cache()
        # == == == == == == == == == == == == == == == == == == ==

        # Saving results to file
        print("... Saving raw Really Knows ...")
        data_io.save_jsonl(all_rk, full_raw_out_f)
        print("... Saving parsed Really Knows ...")
        data_io.save_jsonl(all_parsed_rk, full_parsed_out_f)
        print("All data saved.")
