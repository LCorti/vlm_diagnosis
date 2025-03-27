import argparse
import os
import sys
import torch

module_path = os.path.abspath(os.path.join("../../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from pathlib import Path
from really_knows_processing.common_utils.gen_utils import GenUtils
from transformers import AutoTokenizer, AutoModel
from utils.data_io import load_json, save_jsonl
from utils.image_utils import load_image
from utils.model_utils import split_model, make_message

PROMPT_VERSION = 4
MODEL_NAME = "internvl2"
HF_MODEL_NAME = "OpenGVLab/InternVL2-8B"


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for InternVL2.")
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
    base_dir = Path(__file__).parent.parent.parent

    # Load generation config
    gen_utils = GenUtils(MODEL_NAME, prompt_version=PROMPT_VERSION)
    gen_config = gen_utils.get_gen_config(
        do_sample=args.do_sample,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    # Get prompt templates for generation
    question_template = gen_utils.get_question_template(ds_name)

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

    # == == == == Get responses == == == ==
    print("Running inference...")
    all_responses = []

    for curr_q in questions:
        print("Current question ID: {}".format(curr_q["question_id"]))
        print("-- Text: {}".format(curr_q["question"]))

        # Loading image
        img_path = base_dir.joinpath(curr_q["img_path"])
        print("-- Loading image {}".format(img_path))
        image_tensor = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()

        # Format text input
        message = make_message(question_template, curr_q)

        response = gen_utils.gen_response_internvl2(
            model, tokenizer, image_tensor, message
        )

        all_responses.append(
            {
                "question_id": curr_q["question_id"],
                "question": curr_q["question"],
                "response": response,
            }
        )

        # Free up memory because the model is large
        del image_tensor
        torch.cuda.empty_cache()

        # == == == == == == == == == == == == == == == == == == ==

        # Save responses to file
        # TODO: update saving directory.
        print("... Saving data ...")
        save_jsonl(all_responses, "./responses.jsonl")
        print("Data saved.")
