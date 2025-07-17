import argparse
import sys
import torch

from share4v.constants import IMAGE_TOKEN_INDEX
from share4v.mm_utils import get_model_name_from_path, tokenizer_image_token
from share4v.model.builder import load_pretrained_model
from share4v.utils import disable_torch_init
from pathlib import Path

module_path = str(Path("..", "..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.dataset_config_loader import DatasetConfig
from config_loaders.rk_handler import RKHandler
from really_knows_processing.common_utils.gen_utils import GenUtils
from really_knows_processing.common_utils.image_utils import load_image
from really_knows_processing.common_utils.path_utils import merge_path
from utils.data_io import make_dir, load_json, load_jsonl, append_to_jsonl
from utils.model_utils import (
    add_conv_step,
    get_conv_template,
    get_stop_str,
    get_stopping_criteria,
    make_message,
    set_conv_mode,
)


PROMPT_VERSION = 4
MODEL_NAME = "sharegpt4v"
HF_MODEL_NAME = "Lin-Chen/ShareGPT4V-7B"
SURF_DRIVE = Path("data", "storage")


# Kwargs parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Counterfacual generation for ShareGPT4V"
    )
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

    # Load dataset config
    ds_config = DatasetConfig()
    ds_paths = ds_config.get_ds_paths(ds_name)

    # Load RK Config
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
        num_beams=args.num_beams,
        temperature=args.temperature,
        use_cache=args.use_cache,
        max_new_tokens=args.max_new_tokens,
    )
    # Get prompt templates for generation
    question_template = gen_utils.get_question_template(ds_name)

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

        print("Current question ID: {}".format(curr_q["question_id"]))
        print("-- Text: {}".format(curr_q["question"]))

        ds_class = curr_q["class"]
        curr_dir = SURF_DRIVE.joinpath(ds_paths[ds_class]["imgs_occluded"])
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

            # Loading image
            image_path = merge_path(str(curr_dir), curr_occ["path"])
            image = load_image(image_path)
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

            # Get response
            curr_occ_res["response"] = gen_utils.gen_response_sharegpt4v(
                model, tokenizer, image_tensor, input_ids, stopping_criteria, stop_str
            )

            all_resp_counter.append(curr_q["question_id"])
            curr_subset.append(curr_occ_res)

            # Free up memory
            del image_tensor
            del input_ids
            torch.cuda.empty_cache()
            # == == == == == == == == == == == == == == == == == == ==

        # Saving results to file once computations for each 'size' are done
        print("... Saving data ...")
        append_to_jsonl(curr_subset, full_counter_out_f)
        print("Data saved.")
