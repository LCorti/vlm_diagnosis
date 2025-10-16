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

from config_handlers.causal_handler import CausalHandler
from config_handlers.dataset_handler import DatasetHandler
from rk_processing.utils.gen_utils import GenUtils
from rk_processing.utils.path_utils import merge_path


PROMPT_VERSION = 4
MODEL_NAME = "sharegpt4v"
HF_MODEL_NAME = "Lin-Chen/ShareGPT4V-7B"


# Kwargs parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Counterfactual output generation for ShareGPT4V"
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
    base_dir = Path(__file__).parent.parent.parent

    # Load dataset config
    ds_hdl = DatasetHandler()
    ds_hdl.set_curr_ds(ds_name)

    # Load handler for causal analysis
    causal_hdl = CausalHandler()
    causal_hdl.set_curr_model(MODEL_NAME)
    causal_hdl.set_curr_ds(ds_name)
    imgs_occ_paths = causal_hdl.get_imgs_occluded_path()
    imgs_occ_paths = data_io.add_path_to_surf_storage(base_dir, imgs_occ_paths)
    counter_resps_path = base_dir.joinpath(causal_hdl.get_counter_resps_path())
    data_io.make_dir(counter_resps_path.parent)

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
    disable_torch_init()  # Kept from original implementation
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

    # == == == == Get counterfactual responses on occluded images == == == ==
    print("Running inference on occluded images...")

    # Check if some data is already present, if so load it.
    # Only keep the question ids for later checks.
    if counter_resps_path.exists():
        all_counter_resps = data_io.load_jsonl(counter_resps_path)
        all_counter_resps = list(set([cq["question_id"] for cq in all_counter_resps]))
    else:
        all_counter_resps = []

    det = data_io.load_jsonl(imgs_occ_paths.joinpath("det_paths.jsonl"))
    for q_det in det:
        # If data for curr_q is already present, skip it.
        if q_det["question_id"] in all_counter_resps:
            continue

        # Retrieve question info
        question = next(
            (q for q in questions if q["question_id"] == q_det["question_id"]), None
        )
        if question:
            print(f"Current question ID: {q_det['question_id']}")
            print(f"-- Text: {question['question']}")
        else:
            print(f"Problem with question with ID: {q_det['question_id']}")

        curr_cr_list = []
        for subset in q_det["powerset"].values():
            for _, data in subset.items():
                # Prepare object to store info
                curr_cr = {
                    "question_id": q_det["question_id"],
                    "occluded": data["combination"],
                    "size": len(data["combination"]),
                    "path": data["path"],
                }

                # Run inference
                # Loading image
                img_path = merge_path(imgs_occ_paths, data["path"])
                image = image_utils.load_img(img_path)
                image_tensor = (
                    image_processor.preprocess(image, return_tensors="pt")[
                        "pixel_values"
                    ]
                    .half()
                    .to("cuda:0")
                )
                # Format text input
                message = make_message(model, question_template, question)
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
                stopping_criteria = get_stopping_criteria(
                    input_ids, stop_str, tokenizer
                )
                # Get response
                curr_cr["response"] = gen_utils.generate_sharegpt4v(
                    model,
                    tokenizer,
                    image_tensor,
                    input_ids,
                    stopping_criteria,
                    stop_str,
                )
                curr_cr_list.append(curr_cr)

                # Free up memory
                del image_tensor
                del input_ids
                torch.cuda.empty_cache()

        # Saving results to file once computations for each 'size' are done
        print("... Saving data ...")
        data_io.append_to_jsonl(curr_cr_list, counter_resps_path)
        print("Data saved.")
