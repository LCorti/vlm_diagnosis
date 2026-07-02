import argparse
import gc
import sys
from pathlib import Path

import torch
from captum.attr import IntegratedGradients
from qwen_vl_utils import process_vision_info
from torch.amp import autocast
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from utils.model_utils import add_conv_step, make_message

module_path = str(Path("..", "..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

import utils.data_io as data_io  # noqa: E402
import utils.image_utils as image_utils  # noqa: E402
import utils.interpret_vis as interpret_vis  # noqa: E402
from rk_processing.utils.gen_utils import GenUtils  # noqa: E402

PROMPT_VERSION = 4
MODEL_NAME = "qwen2_5_vl"
HF_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description="RK Generation for Qwen2.5-VL.")
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


def get_module_device(module):
    return next(module.parameters()).device


class Qwen2_5_VLGradientWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        vision_embeds,
        input_ids,
        attention_mask,
        image_grid_thw,
        target_token_id,
    ):
        """
        Run Qwen2.5-VL from precomputed visual embeddings and return the
        next-token logit for Captum.

        Qwen2.5-VL inserts `self.visual(pixel_values, image_grid_thw)` outputs
        into the text embedding sequence at `config.image_token_id` positions.
        This wrapper mirrors that insertion directly so Integrated Gradients can
        vary only the precomputed image-token embeddings.

        Args:
            vision_embeds: Precomputed Qwen visual features, shaped
                [batch, num_image_tokens, hidden_size]. Captum varies this.
            input_ids: Text/image-token prefix up to t-1.
            attention_mask: Mask for input_ids.
            image_grid_thw: Qwen processor grid metadata for the image.
            target_token_id: Vocabulary ID of the response token to attribute.
        """
        batch_size = vision_embeds.shape[0]
        device = vision_embeds.device

        input_ids = input_ids.to(device).expand(batch_size, -1).contiguous()
        attention_mask = attention_mask.to(device).expand(batch_size, -1).contiguous()
        image_grid_thw = image_grid_thw.to(device)
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        if image_grid_thw.shape[0] == 1 and batch_size > 1:
            image_grid_thw = image_grid_thw.repeat(batch_size, 1)

        inputs_embeds = self.model.model.embed_tokens(input_ids)
        image_mask = input_ids == self.model.config.image_token_id
        num_image_tokens = image_mask.sum(dim=1)
        expected_image_tokens = vision_embeds.shape[1]
        if not torch.all(num_image_tokens == expected_image_tokens):
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens per sample {num_image_tokens.tolist()}, "
                f"features {expected_image_tokens}."
            )

        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, vision_embeds)

        position_ids, _ = self.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=attention_mask,
        )

        outputs = self.model.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = self.model.lm_head(outputs[0])

        target_positions = attention_mask.long().sum(dim=1).to(logits.device) - 1
        batch_indices = torch.arange(batch_size, device=logits.device)
        return logits[batch_indices, target_positions, target_token_id]

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
    out_dir = Path(base_dir, "data", "really_know", "attribs", MODEL_NAME, ds_name)
    data_io.make_dir(out_dir)

    # Load generation config
    gen_utils = GenUtils(MODEL_NAME, prompt_version=PROMPT_VERSION)
    gen_utils.get_gen_config(
        do_sample=args.do_sample,
        temperature=args.temperature,
        use_cache=args.use_cache,
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
    model.eval()
    # The default range for the number of visual tokens per image in the model is 4-16384.
    # Keep the same bounds used by qwen2_5_vl_self.py.
    min_pixels = 4 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        HF_MODEL_NAME, min_pixels=min_pixels, max_pixels=max_pixels
    )
    tokenizer = processor.tokenizer

    # Trickeroonies for speed and (hopefully) avoid CUDA OOM errors
    model.gradient_checkpointing_enable()
    for param in model.parameters():
        param.requires_grad = False

    # Define tokens to skip. Do not run attribution on these.
    skip_strings = {"<s>", "</s>", "<pad>", "\n", " ", ""}
    skip_token_ids = set()
    for s in skip_strings:
        tok_id = tokenizer.convert_tokens_to_ids(s)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            skip_token_ids.add(tok_id)
    for tok_id in (
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    ):
        if tok_id is not None:
            skip_token_ids.add(tok_id)
    print("Model loaded.")

    # Wrap model for Captum
    wrapped_model = Qwen2_5_VLGradientWrapper(model)
    attrib_method = IntegratedGradients(wrapped_model)
    print("Model wrapped.")

    # Load data
    print("Loading data...")
    print(f"Loading questions from {questions_file}")
    questions = data_io.load_json(questions_file)

    # == == == == Get baseline responses and attributions == == == ==
    print("Start processing...")

    language_device = get_module_device(model.model.embed_tokens)
    vision_device = get_module_device(model.visual)
    spatial_merge_size = model.config.vision_config.spatial_merge_size

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

        conv = []
        message = make_message(question_template, curr_q)
        conv = add_conv_step(conv, message, image)
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
        image_grid_thw = inputs.image_grid_thw.detach().clone()
        curr_attrib["image_grid_thw"] = image_grid_thw.cpu()
        curr_attrib["spatial_merge_size"] = spatial_merge_size

        # Get baseline response
        with torch.inference_mode():
            baseline_response = gen_utils.generate_qwen2_5_vl(model, processor, inputs)
        curr_attrib["baseline_response"] = baseline_response

        print("=" * 25)
        print("Baseline response:")
        print(baseline_response)
        print("=" * 25)

        prompt_ids = inputs.input_ids[0].to(language_device)
        response_ids = (
            tokenizer(baseline_response, return_tensors="pt", add_special_tokens=False)
            .input_ids[0]
            .to(language_device)
        )

        curr_attrib["token_attrib"] = {}

        with torch.no_grad():
            pixel_values = inputs.pixel_values.to(
                device=vision_device, dtype=model.visual.dtype
            )
            visual_grid_thw = image_grid_thw.to(vision_device)
            precomputed_vision_embeds = model.visual(
                pixel_values, grid_thw=visual_grid_thw
            ).detach()

        batched_vision_embeds = precomputed_vision_embeds.to(language_device).unsqueeze(
            0
        )
        batched_vision_embeds.requires_grad_()
        batched_baseline = torch.zeros_like(batched_vision_embeds)
        attrib_image_grid_thw = image_grid_thw.to(language_device)

        print("> Running attribution")
        for t in range(len(response_ids)):
            target_token_id = response_ids[t].item()
            decoded_token = tokenizer.decode([target_token_id])
            if target_token_id in skip_token_ids or not decoded_token.strip():
                print(f"Skipping token [{t}]: '{decoded_token}'")
                curr_attrib["token_attrib"][t] = [
                    decoded_token,
                    torch.zeros(batched_vision_embeds.shape[1]),
                ]
                continue

            print(f"Computing token [{t}]: '{decoded_token}'...")

            # Construct prefix: prompt + generated tokens up to t-1.
            current_input_ids = torch.cat([prompt_ids, response_ids[:t]]).unsqueeze(0)
            current_attention_mask = torch.ones_like(current_input_ids)

            # Do attribution for the next token after the current prefix.
            with autocast("cuda", dtype=torch.bfloat16):
                attributions = attrib_method.attribute(
                    inputs=batched_vision_embeds,
                    baselines=batched_baseline,
                    target=None,
                    additional_forward_args=(
                        current_input_ids,
                        current_attention_mask,
                        attrib_image_grid_thw,
                        target_token_id,
                    ),
                    n_steps=5,
                    internal_batch_size=1,
                )

            # Remove from GPU and append.
            patch_attributions = (
                attributions.detach().cpu().abs().sum(dim=-1).squeeze(0)
            )
            curr_attrib["token_attrib"][t] = [decoded_token, patch_attributions]
            print(f"> Computed IG for token [{t}]: '{decoded_token}'")

            # Save test image
            # smooth_heatmap = interpret_vis.qwen2_5_vl_patch_scores_to_heatmap(
            #     patch_attributions,
            #     image,
            #     image_grid_thw=image_grid_thw.cpu(),
            #     spatial_merge_size=spatial_merge_size,
            # )
            # interpret_vis.save_attribution_overlay(
            #     image=image,
            #     heatmap=smooth_heatmap,
            #     out_path=f"testmap_{decoded_token}",
            #     title=f"Token {decoded_token} — smooth",
            # )

            # Free up memory.
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
            inputs,
            image_grid_thw,
            precomputed_vision_embeds,
            batched_vision_embeds,
            batched_baseline,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # break

    print("Finished!")
