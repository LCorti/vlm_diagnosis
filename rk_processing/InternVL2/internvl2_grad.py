import argparse
import gc
import sys
from pathlib import Path

import torch
from captum.attr import IntegratedGradients
from torch.amp import autocast
from transformers import AutoModel, AutoTokenizer
from utils.model_utils import make_message, split_model

module_path = str(Path("..", "..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

import utils.data_io as data_io
import utils.image_utils as image_utils
from config_handlers.rk_handler import RKHandler
from rk_processing.utils.gen_utils import GenUtils

PROMPT_VERSION = 4
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
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


class InternVL2GradientWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self, pixel_values, input_ids, attention_mask, target_token_pos, target_token_id
    ):
        """
        Computes the forward pass and isolates the logit of the target token.

        Args:
            pixel_values (Tensor): Image tensor [num_patches, C, H, W]
            input_ids (Tensor): Token prefix up to t-1 [1, seq_len]
            attention_mask (Tensor): Mask for the prefix [1, seq_len]
            target_token_pos (int): The sequence index of the token being predicted
            target_token_id (int): The vocabulary ID of the target token
        """
        # Unpack from Captum's input [B, P, C, H, W]
        B, P, C, H, W = pixel_values.shape
        pixel_values = pixel_values.view(B * P, C, H, W)

        image_flags = torch.ones(
            (B * P, 1), dtype=torch.long, device=pixel_values.device
        )

        input_ids = input_ids.expand(B, -1).contiguous()
        attention_mask = attention_mask.expand(B, -1).contiguous()

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            use_cache=False,
            output_hidden_states=False,
        )

        # Isolate the logits for the specific token ID at the specific position
        # outputs.logits shape: [B, seq_len, vocab_size]
        target_logits = outputs.logits[:, target_token_pos, target_token_id]

        return target_logits

    def __getattr__(self, name):
        """Delegate necessary internal attributes to the underlying model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


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
    raw_out_f = str(rk_hdl.get_rk_raw_path()).format(PROMPT_VERSION)
    parsed_out_f = str(rk_hdl.get_rk_parsed_path()).format(PROMPT_VERSION)
    # (2) make directory if missing
    base_dir = Path(__file__).parent.parent.parent
    full_raw_out_f = base_dir.joinpath(raw_out_f)
    full_raw_out_dir = base_dir.joinpath(Path(raw_out_f).parent)
    data_io.make_dir(full_raw_out_dir)
    full_parsed_out_f = base_dir.joinpath(parsed_out_f)
    full_parsed_out_dir = base_dir.joinpath(Path(parsed_out_f).parent)
    data_io.make_dir(full_parsed_out_dir)

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
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
    # If more GPUs are available split, use authors' function to split the model
    if torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs; splitting model...")
        device_map = split_model(Path(HF_MODEL_NAME).stem)
        model = AutoModel.from_pretrained(
            HF_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
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
                use_flash_attn=True,
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )
    # Trickeroonies
    model.gradient_checkpointing_enable()
    if hasattr(model, "vision_model"):
        if hasattr(model.vision_model, "gradient_checkpointing"):
            model.vision_model.gradient_checkpointing = True
        elif hasattr(model.vision_model, "encoder"):
            model.vision_model.encoder.gradient_checkpointing = True
    for param in model.parameters():
        param.requires_grad = False
    print("Model loaded.")

    # Wrap model for Captum
    wrapped_model = InternVL2GradientWrapper(model)
    attrib_method = IntegratedGradients(wrapped_model)
    print("Model wrapped.")

    # Load data
    print("Loading data...")
    print(f"Loading questions from {questions_file}")
    questions = data_io.load_json(questions_file)

    # == == == == Get responses and self-explanations == == == ==
    print("Running inference and computing explanations...")
    # all_rk = []
    # all_parsed_rk = []

    for curr_q in questions:
        print(f"Current question ID: {curr_q['question_id']}")
        print(f"-- Text: {curr_q['question']}")

        # Prepare object to store info
        # curr_rk = {"question_id": curr_q["question_id"]}

        # Wrapping image processor for baseline response and attribution.
        # def processor_fn(image):
        #     return image_utils.get_image_tensor(image).to(torch.bfloat16).cuda()

        # Preparing image and prompt
        img_path = base_dir.joinpath(curr_q["img_path"])
        print(f"-- Loading image {img_path}")
        image = image_utils.load_img(img_path)
        image_tensor = (
            image_utils.get_image_tensor(image, max_num=6).to(torch.bfloat16).cuda()
        )

        message = make_message(question_template, curr_q)
        conversation = [{"role": "user", "content": message}]
        prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        # Inject image tokens into input
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        if getattr(model, "img_context_token_id", None) is None:
            model.img_context_token_id = tokenizer.convert_tokens_to_ids(
                IMG_CONTEXT_TOKEN
            )

        num_patches = image_tensor.shape[0]
        num_image_tokens = model.num_image_token * num_patches
        image_token_sequence = (
            "<img>" + (IMG_CONTEXT_TOKEN * num_image_tokens) + "</img>"
        )
        prompt = prompt.replace("<image>", image_token_sequence)

        # Get baseline response
        with torch.inference_mode():
            baseline_response, _ = gen_utils.generate_internvl2(
                model, tokenizer, image_tensor, message
            )
        print("=" * 25)
        print("Baseline response:")
        print(baseline_response)
        print("=" * 25)

        prompt_ids = (
            tokenizer(prompt, return_tensors="pt").input_ids[0].to(model.device)
        )
        response_ids = (
            tokenizer(baseline_response, return_tensors="pt")
            .input_ids[0]
            .to(model.device)
        )

        token_attributions = []
        # Add fake dimension [1, P, C, H, W]
        batched_image = image_tensor.unsqueeze(0)
        batched_baseline = torch.zeros_like(batched_image)

        print("> Running attribution")
        for t in range(len(response_ids)):
            target_token_id = response_ids[t].item()
            # Construct prefix: Prompt + generated tokens up to t-1
            current_input_ids = torch.cat([prompt_ids, response_ids[:t]]).unsqueeze(0)
            current_attention_mask = torch.ones_like(current_input_ids)

            # Predicting the last token in the current sequence
            target_pos = current_input_ids.shape[1] - 1

            # Do attribution for token at target_pos
            with autocast("cuda", dtype=torch.bfloat16):
                attributions = attrib_method.attribute(
                    inputs=batched_image,
                    baselines=batched_baseline,
                    target=None,
                    additional_forward_args=(
                        current_input_ids,
                        current_attention_mask,
                        target_pos,
                        target_token_id,
                    ),
                    n_steps=5,
                    internal_batch_size=1,
                )

            # Remove from GPU and append
            compressed_attr = attributions.detach().cpu().abs().sum(dim=1)
            token_attributions.append(compressed_attr)
            decoded_token = tokenizer.decode([target_token_id])
            print(f"Computed IG for token [{t}]: '{decoded_token}'")

            # Free up memory
            # del image_tensor
            del attributions
            gc.collect()
            torch.cuda.empty_cache()

        # TODO: fix this one
        # all_rk.append(curr_rk)
        # all_parsed_rk.append(gen_utils.parse_raw_rk(curr_rk))

        # TODO: remove this later
        break
        # == == == == == == == == == == == == == == == == == == ==

    # Saving results to file
    # print("... Saving raw Really Knows ...")
    # data_io.save_jsonl(all_rk, full_raw_out_f)
    # print("... Saving parsed Really Knows ...")
    # data_io.save_jsonl(all_parsed_rk, full_parsed_out_f)
    # print("All data saved.")
