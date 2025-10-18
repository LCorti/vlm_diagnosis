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

from config_handlers.causal_handler import CausalHandler
from config_handlers.dataset_handler import DatasetHandler
from rk_processing.utils.gen_utils import GenUtils
from rk_processing.utils.path_utils import merge_path

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
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    # Get prompt templates for generation
    question_template = gen_utils.get_question_template(ds_name)

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
                # Load image
                img_path = merge_path(imgs_occ_paths, data["path"])
                print("-- Loading image {}".format(img_path))
                image = image_utils.load_img(img_path)

                # Process inputs
                conv = []
                # Prepare actual question
                curr_message_text = make_message(question_template, question)
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

                curr_cr["response"] = gen_utils.generate_qwen2_5_vl(
                    model, processor, inputs
                )
                curr_cr_list.append(curr_cr)

                # Free up memory
                del inputs
                torch.cuda.empty_cache()

        # Saving results to file once computations for each 'size' are done
        print("... Saving data ...")
        data_io.append_to_jsonl(curr_cr_list, counter_resps_path)
        print("All data saved.")
