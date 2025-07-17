import json
import torch
import argparse

from pathlib import Path
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# LLaVa output includes the user query.
# This is used to find the response and extract it.
SPLIT_KW = "ASSISTANT:"


def parse_args():
    parser = argparse.ArgumentParser(description="llavanext_inference")
    parser.add_argument("--ds_name", required=True, help="Name of the experiment.")
    parser.add_argument(
        "--ds_folder", required=True, help="Path to the dataset folder."
    )
    parser.add_argument(
        "--questions_file", required=True, help="Path to the file with questions."
    )
    parser.add_argument(
        "--out_dir",
        help="Path to folder to save the responses from the LMM.",
        default="results",
    )
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


def format_question(q):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": q},
            ],
        },
    ]


def main():
    print("Initializing model...")
    args = parse_args()
    gpu_device = "cuda:{}".format(args.gpu_id)
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(gpu_device)
    print("Initialisation finished.")

    print("Loading data...")
    print("Loading questions from {}".format(args.questions_file))
    questions = []
    with open(args.questions_file, "r") as f:
        questions = json.load(f)

    print("Running inference...")
    all_responses = []

    for q in questions:
        print("Current question ID: {}".format(q["question_id"]))
        print("-- Text: {}".format(q["question"]))

        img_path = Path(args.ds_folder, q["img_path"])
        print("-- Loading image {}".format(img_path))
        image = Image.open(img_path)

        # format question
        conversation = format_question(q["question"])
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(
            gpu_device
        )

        output = model.generate(**inputs, max_new_tokens=100)
        full_response = processor.decode(output[0], skip_special_tokens=True)

        split_idx = full_response.find(SPLIT_KW)
        if split_idx != -1:
            split_idx += len(SPLIT_KW)
        else:
            split_idx = 0
        response = full_response[split_idx:].lstrip()

        all_responses.append(
            {
                "question_id": q["question_id"],
                "question": q["question"],
                "response": response,
            }
        )

        # Free up memory
        del image
        torch.cuda.empty_cache()

    res_folder = args.out_dir
    Path(res_folder).mkdir(parents=True)

    with open(Path(res_folder, f"exp_{args.ds_name}_responses.jsonl"), "w") as f:
        for r in all_responses:
            json.dump(r, f)
            f.write("\n")


if __name__ == "__main__":
    main()
