import argparse
import os
import sys
import torch

module_path = os.path.abspath(os.path.join("../../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.rk_config_loader import RKConfig
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from utils.data_io import make_dir, load_json, save_jsonl
from utils.image_utils import load_image
from utils.model_utils import split_model
from really_knows_processing.common_utils.gen_utils import GenUtils

PROMPT_VERSION = 4
MODEL_NAME = "internvl2"


# Kwargs parser
def parse_args():
    parser = argparse.ArgumentParser(description="RK Generation for InternVL2.")
    parser.add_argument("--ds_name", required=True, help="Name of the dataset.")
    parser.add_argument(
        "--questions_file", required=True, help="Path to the file with questions."
    )
    parser.add_argument("--do_sample", default=False)
    parser.add_argument("--temperature", help="Model temperature.", default=0.1)
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
        args.do_sample, args.temperature, args.max_new_tokens
    )
    # Get prompt templates for generation
    question_template = gen_utils.get_question_template(ds_name)
    rationale_template = gen_utils.get_rationale_template()
    out_format_template = gen_utils.get_out_format_template()

    # Load model
    print("Loading model...")
    hf_model_name = "OpenGVLab/InternVL2-8B"
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    # If more GPUs are available split, use authors' function to split the model
    if torch.cuda.device_count() > 1:
        print("Found {} GPUs; splitting model...".format(torch.cuda.device_count()))
        device_map = split_model(Path(hf_model_name).stem)
        model = AutoModel.from_pretrained(
            hf_model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
    else:
        print("Found 1 GPU.")
        model = (
            AutoModel.from_pretrained(
                hf_model_name,
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
    all_model_rk = []

    for curr_q in questions:
        print(f"Current question ID: {curr_q['question_id']}")
        print(f"-- Text: {curr_q['question']}")

        # Prepare object to store info
        model_rk = {"question_id": curr_q["question_id"]}

        # Loading image
        img_path = base_dir.joinpath(curr_q["img_path"])
        print("-- Loading image {}".format(img_path))
        image_tensor = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()

        # Format text input
        message = gen_utils.make_message(question_template, curr_q)

        # First generation step: get answer from the model
        model_rk["response"], history = gen_utils.gen_response_internvl2(
            model, tokenizer, image_tensor, message
        )
        print(model_rk["response"])
        print("=" * 25)

        # Second generation step: get unstructured rationales for model output
        model_rk["rationales"], history = gen_utils.gen_response_internvl2(
            model,
            tokenizer,
            image_tensor,
            rationale_template,
            history=history,
        )
        print(model_rk["rationales"])
        print("=" * 25)

        # Third step: triple extraction and structuring from rationales
        model_rk["triples"], history = gen_utils.gen_response_internvl2(
            model,
            tokenizer,
            image_tensor,
            out_format_template,
            history=history,
        )
        print(model_rk["triples"])
        print("=" * 25)

        all_model_rk.append(model_rk)

        # Free up memory
        del image_tensor
        torch.cuda.empty_cache()

        # == == == == == == == == == == == == == == == == == == ==

        # Saving results to file
        print("... Saving data ...")
        save_jsonl(all_model_rk, full_out_file)
        print("Data saved.")
