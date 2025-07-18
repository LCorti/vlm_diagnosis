import argparse
import sys
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModel

module_path = str(Path("..", "..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.rk_handler import RKHandler
from really_knows_processing.common_utils.gen_utils import GenUtils
from really_knows_processing.common_utils.image_utils import (
    load_image,
    get_image_tensor,
)
from utils.data_io import make_dir, load_json, save_jsonl
from utils.model_utils import split_model, make_message

PROMPT_VERSION = 5
MODEL_NAME = "internvl2"
HF_MODEL_NAME = "OpenGVLab/InternVL2-8B"


# Kwargs parser
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
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU to load model.")
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
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
    # If more GPUs are available split, use authors' function to split the model
    if torch.cuda.device_count() > 1:
        print("Found {} GPUs; splitting model...".format(torch.cuda.device_count()))
        device_map = split_model(Path(HF_MODEL_NAME).stem)
        model = AutoModel.from_pretrained(
            HF_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
    else:
        print("Found 1 GPU.")
        model = (
            AutoModel.from_pretrained(
                HF_MODEL_NAME,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )
    print("Model loaded.")

    # Load data
    print("Loading data...")
    print("Loading questions from {}".format(questions_file))
    questions = load_json(questions_file)

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
        image = load_image(img_path)
        image_tensor = get_image_tensor(image, max_num=12).to(torch.bfloat16).cuda()

        # Format text input
        message = make_message(question_template, curr_q)

        # First generation step: get answer from the model
        curr_rk["response"], history = gen_utils.gen_response_internvl2(
            model, tokenizer, image_tensor, message
        )
        print(curr_rk["response"])
        print("=" * 25)

        # Second generation step: get unstructured rationales for model output
        curr_rk["rationales"], history = gen_utils.gen_response_internvl2(
            model,
            tokenizer,
            image_tensor,
            rationale_template,
            history=history,
        )
        print(curr_rk["rationales"])
        print("=" * 25)

        # Third step: triple extraction and structuring from rationales
        curr_rk["triples"], history = gen_utils.gen_response_internvl2(
            model,
            tokenizer,
            image_tensor,
            out_format_template,
            history=history,
        )
        print(curr_rk["triples"])
        print("=" * 25)

        all_rk.append(curr_rk)
        all_parsed_rk.append(gen_utils.parse_raw_rk(curr_rk))

        # Free up memory
        del image_tensor
        torch.cuda.empty_cache()

        # == == == == == == == == == == == == == == == == == == ==

        # Saving results to file
        print("... Saving raw Really Knows ...")
        save_jsonl(all_rk, full_raw_out_f)
        print("... Saving parsed Really Knows ...")
        save_jsonl(all_parsed_rk, full_parsed_out_f)
        print("All data saved.")
