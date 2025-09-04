import argparse
import sys
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from utils.model_utils import split_model, make_message

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
MODEL_NAME = "internvl2"
HF_MODEL_NAME = "OpenGVLab/InternVL2-8B"


# Kwargs parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Counterfactual output generation for InternVL2."
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
                image = image_utils.load_img(img_path)
                image_tensor = (
                    image_utils.get_image_tensor(image, max_num=12)
                    .to(torch.bfloat16)
                    .cuda()
                )
                # Format text input
                message = make_message(question_template, question)
                # Get response
                curr_cr["response"], _ = gen_utils.gen_response_internvl2(
                    model, tokenizer, image_tensor, message
                )
                curr_cr_list.append(curr_cr)

                # Free up memory
                del image_tensor
                torch.cuda.empty_cache()

        # Saving results to file once computations for each 'size' are done
        print("... Saving data ...")
        data_io.append_to_jsonl(curr_cr_list, counter_resps_path)
        print("Data saved.")
