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
        self,
        vision_embeds,
        input_ids,
        attention_mask,
        target_token_pos,
        target_token_id,
    ):
        """
        Computes the forward pass and isolates the logit of the target token.

        Args:
            vision_embeds (Tensor): Precomputed embeddings
            input_ids (Tensor): Token prefix up to t-1 [1, seq_len]
            attention_mask (Tensor): Mask for the prefix [1, seq_len]
            target_token_pos (int): The sequence index of the token being predicted
            target_token_id (int): The vocabulary ID of the target token
        """
        B = vision_embeds.shape[0]
        squeezed_embeds = vision_embeds.squeeze(0)
        original_extract_feature = self.model.extract_feature
        # Remove extra dimension added by Captum
        self.model.extract_feature = lambda x: vision_embeds.squeeze(0)

        if squeezed_embeds.dim() == 3:
            # Shape is [num_patches, 256, embed_dim]
            num_patches = squeezed_embeds.shape[0]
        else:
            # Shape is [num_patches * 256, embed_dim]
            num_patches = squeezed_embeds.shape[0] // 256

        # Make fake pixel_values
        fake_pixels = torch.zeros(
            (B * num_patches, 3, 448, 448),
            device=vision_embeds.device,
            dtype=vision_embeds.dtype,
        )
        image_flags = torch.ones(
            (B * num_patches, 1), dtype=torch.long, device=vision_embeds.device
        )

        input_ids = input_ids.expand(B, -1).contiguous()
        attention_mask = attention_mask.expand(B, -1).contiguous()

        # Do I need this try block?
        try:
            outputs = self.model(
                pixel_values=fake_pixels,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                use_cache=False,
                output_hidden_states=False,
            )
        finally:
            # Restore original method
            self.model.extract_feature = original_extract_feature

        # Isolate the logit for the specific token ID at the specific position
        # outputs.logits shape: [batch=1, seq_len, vocab_size]
        target_logit = outputs.logits[:, target_token_pos, target_token_id]

        # Captum expects a 1D tensor matching the batch dimension
        return target_logit

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

    # Prepare output directory
    base_dir = Path(__file__).parent.parent.parent
    out_dir = Path(base_dir, "data", "really_know", "attribs", "internvl2", ds_name)
    data_io.make_dir(out_dir)

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
    # Trickeroonies for speed and (hopefully) avoid CUDA OOM errors
    model.gradient_checkpointing_enable()
    if hasattr(model, "vision_model"):
        if hasattr(model.vision_model, "gradient_checkpointing"):
            model.vision_model.gradient_checkpointing = True
        elif hasattr(model.vision_model, "encoder"):
            model.vision_model.encoder.gradient_checkpointing = True
    for param in model.parameters():
        param.requires_grad = False

    # Define tokens to skip. Do not run attribution on these.
    skip_strings = {"<s>", "</s>", "<pad>", "\n", " ", ""}
    skip_token_ids = set()
    for s in skip_strings:
        tok_id = tokenizer.convert_tokens_to_ids(s)
        if tok_id != tokenizer.unk_token_id:
            skip_token_ids.add(tok_id)
    if tokenizer.bos_token_id is not None:
        skip_token_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        skip_token_ids.add(tokenizer.eos_token_id)
    print("Model loaded.")

    # Wrap model for Captum
    wrapped_model = InternVL2GradientWrapper(model)
    attrib_method = IntegratedGradients(wrapped_model)
    print("Model wrapped.")

    # Load data
    print("Loading data...")
    print(f"Loading questions from {questions_file}")
    questions = data_io.load_json(questions_file)

    # == == == == Get baseline responses and attributions == == == ==
    print("Start processing...")

    for curr_q in questions:
        print(f"Current question ID: {curr_q['question_id']}")
        print(f"-- Text: {curr_q['question']}")

        # Prepare object to store info and make directory for current question
        curr_attrib = {"question_id": curr_q["question_id"]}
        curr_out_dir = Path(out_dir, str(curr_q["question_id"]))
        data_io.make_dir(curr_out_dir)

        # Preparing image and prompt
        img_path = base_dir.joinpath(curr_q["img_path"])
        print(f"-- Loading image {img_path}")
        image = image_utils.load_img(img_path)
        image_tensor = image_utils.get_image_tensor(image).to(torch.bfloat16).cuda()

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
        curr_attrib["baseline_response"] = baseline_response

        print("=" * 25)
        print("Baseline response:")
        print(baseline_response)
        print("=" * 25)

        prompt_ids = (
            tokenizer(prompt, return_tensors="pt").input_ids[0].to(model.device)
        )
        response_ids = (
            tokenizer(baseline_response, return_tensors="pt", add_special_tokens=False)
            .input_ids[0]
            .to(model.device)
        )

        curr_attrib["token_attrib"] = {}

        with torch.no_grad():
            precomputed_vision_embeds = model.extract_feature(image_tensor).detach()
        batched_vision_embeds = precomputed_vision_embeds.unsqueeze(0)
        batched_vision_embeds.requires_grad_()
        batched_baseline = torch.zeros_like(batched_vision_embeds)

        print("> Running attribution")
        for t in range(len(response_ids)):
            target_token_id = response_ids[t].item()
            decoded_token = tokenizer.decode([target_token_id])
            if target_token_id in skip_token_ids or not decoded_token.strip():
                print(f"Skipping token [{t}]: '{decoded_token}'")
                # Add empty attribution
                num_patches = batched_vision_embeds.shape[1] // 256
                curr_attrib["token_attrib"][t] = [
                    decoded_token,
                    torch.zeros(num_patches),
                ]
                continue

            print(f"Computing token [{t}]: '{decoded_token}'...")

            # Construct prefix: Prompt + generated tokens up to t-1
            current_input_ids = torch.cat([prompt_ids, response_ids[:t]]).unsqueeze(0)
            current_attention_mask = torch.ones_like(current_input_ids)
            # Predicting the last token in the current sequence
            target_pos = current_input_ids.shape[1] - 1

            # Do attribution for token at target_pos
            with autocast("cuda", dtype=torch.bfloat16):
                attributions = attrib_method.attribute(
                    inputs=batched_vision_embeds,
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
            attr_token = attributions.detach().cpu().abs().sum(dim=-1).squeeze(0)
            if attr_token.dim() == 2:
                patch_attributions = attr_token.sum(dim=1)
            else:
                patch_attributions = attr_token.view(-1, 256).sum(dim=1)

            curr_attrib["token_attrib"][t] = [decoded_token, patch_attributions]
            print(f"> Computed IG for token [{t}]: '{decoded_token}'")

            # Free up memory
            del attributions
            # gc.collect()
            # torch.cuda.empty_cache()

        # Save attributions
        torch.save(curr_attrib, Path(curr_out_dir, "attribs.pt"))
        print(f"Saved attributions for sample ID: {curr_q['question_id']}")

        del (
            curr_attrib,
            prompt_ids,
            response_ids,
            image_tensor,
            precomputed_vision_embeds,
            batched_vision_embeds,
            batched_baseline,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # break

    print("Finished!")
