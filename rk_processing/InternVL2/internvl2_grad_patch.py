import argparse
import gc
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients
from PIL import Image
from torch.cuda.amp import autocast
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


def plot_heatmap(img_path, patch_attributions, grid_shape, token_label=""):
    # TODO: pass the image directly, do not load a second time
    # 1. Load original image
    original_image = Image.open(img_path).convert("RGB")
    img_w, img_h = original_image.size
    img_array = np.array(original_image)

    # 2. Isolate local patches from the global context patch
    # InternVL2 typically appends the global patch at the very end
    num_local_patches = grid_shape[0] * grid_shape[1]

    if hasattr(patch_attributions, "cpu"):
        scores = patch_attributions.detach().cpu().numpy()
    else:
        scores = np.array(patch_attributions)

    if len(scores) > num_local_patches:
        local_scores = scores[:num_local_patches]
        # The remaining score is the global context thumbnail importance
    else:
        local_scores = scores

    # 3. Normalize scores to [0, 1] for the colormap
    # Optional: Zero-out negative attributions if you only care about positive evidence
    local_scores = np.maximum(local_scores, 0)
    if local_scores.max() > 0:
        local_scores = local_scores / local_scores.max()

    # 4. Reshape to spatial grid
    heatmap_2d = local_scores.reshape(grid_shape)

    # 5. Upsample to original image dimensions
    heatmap_resized = cv2.resize(
        heatmap_2d, (img_w, img_h), interpolation=cv2.INTER_CUBIC
    )

    # 6. Apply Jet colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 7. Blend the heatmap and original image
    alpha = 0.55
    overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)

    # 8. Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"IG Heatmap: '{token_label}'")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("heatmap_test.png")
    plt.close()


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
            pixel_values (Tensor): Image tensor [num_patches, C, H, W]
            input_ids (Tensor): Token prefix up to t-1 [1, seq_len]
            attention_mask (Tensor): Mask for the prefix [1, seq_len]
            target_token_pos (int): The sequence index of the token being predicted
            target_token_id (int): The vocabulary ID of the target token
        """
        B = input_ids.shape[0]
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
        target_logit = outputs.logits[0, target_token_pos, target_token_id]

        # Captum expects a 1D tensor matching the batch dimension
        return target_logit.unsqueeze(0)

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
    # Trickeroonies for speed and avoiding CUDA OOM errors
    model.gradient_checkpointing_enable()
    if hasattr(model, "vision_model"):
        if hasattr(model.vision_model, "gradient_checkpointing"):
            model.vision_model.gradient_checkpointing = True
        elif hasattr(model.vision_model, "encoder"):
            model.vision_model.encoder.gradient_checkpointing = True
    for param in model.parameters():
        param.requires_grad = False

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
        image_tensor = image_utils.get_image_tensor(image).to(torch.bfloat16).cuda()
        baseline_pixel_values = torch.zeros_like(image_tensor)

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

        with torch.no_grad():
            precomputed_vision_embeds = model.extract_feature(image_tensor).detach()
        batched_vision_embeds = precomputed_vision_embeds.unsqueeze(0)
        batched_vision_embeds.requires_grad_()
        batched_baseline = torch.zeros_like(batched_vision_embeds)

        # TODO: see where to simplify
        print("> Running attribution")
        for t in range(len(response_ids)):
            target_token_id = response_ids[t].item()
            decoded_token = tokenizer.decode([target_token_id])
            if target_token_id in skip_token_ids or not decoded_token.strip():
                print(f"Skipping token [{t}]: '{decoded_token}'")
                # Add empty attribution
                num_patches = batched_vision_embeds.shape[1] // 256
                token_attributions.append(torch.zeros(num_patches))
                continue

            print(f"Computing token [{t}]: '{decoded_token}'...")

            # Construct prefix: Prompt + generated tokens up to t-1
            current_input_ids = torch.cat([prompt_ids, response_ids[:t]]).unsqueeze(0)
            current_attention_mask = torch.ones_like(current_input_ids)
            # Predicting the last token in the current sequence
            target_pos = current_input_ids.shape[1] - 1

            # Do attribution for token at target_pos
            with autocast(dtype=torch.bfloat16):
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

            token_attributions.append(patch_attributions)
            decoded_token = tokenizer.decode([target_token_id])
            print(f"> Computed IG for token [{t}]: '{decoded_token}'")
            # print(patch_attributions)

            # Visualise attribs
            num_total_patches = patch_attributions.shape[0]
            num_local_patches = num_total_patches - 1 if num_total_patches > 1 else 1
            img_w, img_h = image.size
            aspect_ratio = img_w / img_h
            best_rows, best_cols = 1, num_local_patches
            for r in range(1, num_local_patches + 1):
                if num_local_patches % r == 0:
                    c = num_local_patches // r
                    if abs((c / r) - aspect_ratio) < abs(
                        (best_cols / best_rows) - aspect_ratio
                    ):
                        best_rows, best_cols = r, c
            grid_shape = (best_rows, best_cols)

            if decoded_token.strip() not in ["", ",", ".", "\n"]:
                plot_heatmap(
                    img_path=img_path,
                    patch_attributions=patch_attributions,
                    grid_shape=grid_shape,
                    token_label=decoded_token,
                )

            # Free up memory
            # del image_tensor
            del attributions
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Finished attributions for sample ID: {curr_q['question_id']}")
        print(token_attributions)

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
