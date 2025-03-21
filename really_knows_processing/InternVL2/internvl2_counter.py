import argparse
import os
import sys
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModel

module_path = os.path.abspath(os.path.join("../../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.rk_config_loader import RKConfig
from utils.data_io import make_dir, load_json, save_jsonl
from utils.image_utils import load_image
from utils.model_utils import split_model
from really_knows_processing.common_utils.gen_utils import GenUtils

PROMPT_VERSION = 4
MODEL_NAME = "internvl2"


# Kwargs parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Counterfacual RK generation for InternVL2."
    )
    parser.add_argument("--ds_name", required=True, help="Name of the dataset.")
    parser.add_argument(
        "--questions_file", required=True, help="Path to the file with questions."
    )
    parser.add_argument(
        "--constraint_folder", required=True, help="Path to folder with RK constraints."
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
    constraint_folder = args.constraint_folder

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

    question_template = gen_utils.get_question_template(ds_name)

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

    # == == == == Get responses on restricted information == == == ==
    print("Running inference on restricted information...")
    all_model_resp_counter = []

    for curr_q in questions:
        print(f"Current question ID: {curr_q['question_id']}")
        print(f"-- Text: {curr_q['question']}")

        # Load RK constraints
        file_path = f"{constraint_folder}/graph_alt_text_{curr_q['question_id']}.json"
        rk_constraints = load_json(file_path)

        # Get image path
        img_path = base_dir.joinpath(curr_q["img_path"])

        # Format text input
        message = gen_utils.make_message(question_template, curr_q)

        # Iterate over different sizes of constraints
        for size in rk_constraints:
            print(f"-- Working on constraints of size {size}...")

            # For each 'size' iterate over all constraints settings
            for rk_const in rk_constraints[size]:
                # Prepare object to store info
                model_rk = {
                    "question_id": curr_q["question_id"],
                    "size_const": size,
                    "response": "",
                    "constraints": "",
                }
                model_rk["constraints"] = rk_const

                # Loading image
                image_tensor = (
                    load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                )

                # Include contraint instruction and data
                constr_message = f"{message}\n"
                constr_message += (
                    "When answering the question, only consider the following concepts:"
                )
                constr_message += "\n".join(rk_const)

                # Get response
                model_rk["response"], _ = gen_utils.gen_response_internvl2(
                    model, tokenizer, image_tensor, constr_message
                )

                all_model_resp_counter.append(model_rk)

                # Free up memory
                del image_tensor
                torch.cuda.empty_cache()

                # == == == == == == == == == == == == == == == == == == ==

            # Saving results to file once computations for each 'size' are done
            print("... Saving data ...")
            save_jsonl(all_model_resp_counter, "./counter_resp.jsonl")
            print("Data saved.")
