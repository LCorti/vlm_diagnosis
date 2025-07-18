import argparse
import sys
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModel

module_path = str(Path("..", "..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.dataset_handler import DatasetHandler
from config_handlers.rk_handler import RKHandler
from rk_processing.common_utils.gen_utils import GenUtils
from rk_processing.common_utils.image_utils import (
    load_image,
    get_image_tensor,
)
from rk_processing.common_utils.path_utils import merge_path
from utils.data_io import make_dir, load_json, load_jsonl, append_to_jsonl
from utils.model_utils import split_model, make_message

PROMPT_VERSION = 4
MODEL_NAME = "internvl2"
HF_MODEL_NAME = "OpenGVLab/InternVL2-8B"
SURF_DRIVE = Path("data", "storage")


# Kwargs parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Counterfacual generation for InternVL2."
    )
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

    # Load dataset config
    ds_hdl = DatasetHandler()
    ds_hdl.set_curr_ds(ds_name)

    # Load rk config to save responses
    rk_hdl = RKHandler()
    rk_hdl.set_curr_model(MODEL_NAME)
    rk_hdl.set_curr_ds(ds_name)
    counter_out_f = rk_hdl.get_counterfactual_rk_path()
    # Make directory if missing
    base_dir = Path(__file__).parent.parent.parent
    full_counter_out_f = base_dir.joinpath(counter_out_f)
    full_counter_out_dir = base_dir.joinpath(Path(full_counter_out_f).parent)
    make_dir(full_counter_out_dir)

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

    # == == == == Get counterfactual responses on occluded images == == == ==
    print("Running inference on occluded images...")

    # Check if some data is already present, if so load it.
    # Only keep the question ids for later checks.
    if full_counter_out_f.exists():
        all_resp_counter = load_jsonl(full_counter_out_f)
        all_resp_counter = [curr_q["question_id"] for curr_q in all_resp_counter]
    else:
        all_resp_counter = []

    for curr_q in questions:
        # If data for curr_q is already present, skip it.
        if curr_q["question_id"] in all_resp_counter:
            continue

        print(f"Current question ID: {curr_q['question_id']}")
        print(f"-- Text: {curr_q['question']}")

        ds_class = curr_q["class"]
        ds_hdl.set_curr_class(ds_class)
        curr_dir = SURF_DRIVE.joinpath(ds_hdl.get_imgs_occluded_path())
        path_to_summary = f"{curr_dir.parent}/summary.json"
        summary = load_json(path_to_summary)
        occlusion_data = summary[ds_class][str(curr_q["question_id"])]["occlusion"]
        # curr_dir_imgs = curr_dir.joinpath(curr_q["question_id"])
        print(f">> Found {len(occlusion_data)} occluded images to process.")
        curr_subset = []

        for curr_occ in occlusion_data:
            # Prepare object to store info
            curr_occ_res = {
                "question_id": curr_q["question_id"],
                "size_treatment": len(curr_occ["occluded_concepts"]),
                "occluded_concepts": curr_occ["occluded_concepts"],
                "path": curr_occ["path"],
            }

            # Load image
            image_path = merge_path(str(curr_dir), curr_occ["path"])
            image = load_image(image_path)
            image_tensor = get_image_tensor(image, max_num=12).to(torch.bfloat16).cuda()

            # Format text input
            message = make_message(question_template, curr_q)

            # Get response
            curr_occ_res["response"], _ = gen_utils.gen_response_internvl2(
                model, tokenizer, image_tensor, message
            )

            all_resp_counter.append(curr_q["question_id"])
            curr_subset.append(curr_occ_res)

            # Free up memory
            del image_tensor
            torch.cuda.empty_cache()

            # == == == == == == == == == == == == == == == == == == ==

        # Saving results to file once computations for each 'size' are done
        print("... Saving data ...")
        append_to_jsonl(curr_subset, full_counter_out_f)
        print("Data saved.")
