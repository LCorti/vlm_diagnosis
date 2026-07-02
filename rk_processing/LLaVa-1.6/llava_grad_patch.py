import argparse
import gc
import sys
from pathlib import Path

import torch
from captum.attr import IntegratedGradients
from torch.amp import autocast

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from utils.model_utils import (
    add_conv_step,
    get_conv_template,
    make_message,
    set_conv_mode,
)

module_path = str(Path("..", "..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

import utils.data_io as data_io  # noqa: E402
import utils.image_utils as image_utils  # noqa: E402
import utils.interpret_vis as interpret_vis  # noqa: E402
from rk_processing.utils.gen_utils import GenUtils  # noqa: E402

PROMPT_VERSION = 4
MODEL_NAME = "llava-1.6"
HF_MODEL_NAME = "liuhaotian/llava-v1.6-vicuna-7b"


# Kwargs parser
def parse_args():
    parser = argparse.ArgumentParser(description="RK Generation for LLaVa-1.6.")
    parser.add_argument("--ds_name", required=True, help="Name of the dataset.")
    parser.add_argument(
        "--questions_file", required=True, help="Path to the file with questions."
    )
    parser.add_argument(
        "--do_sample", type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--num_beams", type=int, default=1)
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


class LlavaGradientWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, vision_embeds, input_ids, attention_mask, target_token_id):
        """
        Run LLaVA from precomputed projected image embeddings and return the
        next-token logit for Captum.

        Args:
            vision_embeds: Projected image features from model.encode_images,
                shaped [batch, num_image_tokens, hidden_size]. Captum varies this.
            input_ids: Text/image-token prefix up to t-1, with IMAGE_TOKEN_INDEX
                marking where LLaVA should splice the image features.
            attention_mask: Mask for input_ids.
            target_token_id: Vocabulary ID of the response token to attribute.
        """
        batch_size = vision_embeds.shape[0]
        input_ids = input_ids.expand(batch_size, -1).contiguous()
        attention_mask = attention_mask.expand(batch_size, -1).contiguous()

        original_encode_images = self.model.encode_images
        self.model.encode_images = lambda images: vision_embeds
        fake_images = torch.zeros(
            (batch_size, 3, 336, 336),
            device=vision_embeds.device,
            dtype=vision_embeds.dtype,
        )

        try:
            (_, position_ids, expanded_attention_mask, _, inputs_embeds, _) = (
                self.model.prepare_inputs_labels_for_multimodal(
                    input_ids=input_ids,
                    position_ids=None,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    labels=None,
                    images=fake_images,
                )
            )
        finally:
            self.model.encode_images = original_encode_images

        outputs = self.model(
            input_ids=None,
            attention_mask=expanded_attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )

        if expanded_attention_mask is None:
            target_positions = torch.full(
                (batch_size,),
                outputs.logits.shape[1] - 1,
                dtype=torch.long,
                device=outputs.logits.device,
            )
        else:
            target_positions = expanded_attention_mask.long().sum(dim=1) - 1

        batch_indices = torch.arange(batch_size, device=outputs.logits.device)
        return outputs.logits[batch_indices, target_positions, target_token_id]

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
    out_dir = Path(base_dir, "data", "really_know", "attribs", "llava-1.6", ds_name)
    data_io.make_dir(out_dir)

    # Load generation config
    gen_utils = GenUtils(MODEL_NAME, prompt_version=PROMPT_VERSION)
    gen_utils.get_gen_config(
        do_sample=args.do_sample,
        num_beams=args.num_beams,
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
    disable_torch_init()
    model_base = None
    model_name = get_model_name_from_path(HF_MODEL_NAME)
    conv_mode = set_conv_mode(model_name)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=HF_MODEL_NAME, model_base=model_base, model_name=model_name
    )
    model.half().eval()
    # model.config.use_cache = True Should already be True
    model.gradient_checkpointing_enable()
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
    wrapped_model = LlavaGradientWrapper(model)
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
        image_tensor = (
            image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            .half()
            .to("cuda:0")
        )

        message = make_message(model, question_template, curr_q)
        conv = get_conv_template(conv_mode)
        conv = add_conv_step(conv, message)
        prompt = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        # Get baseline response
        with torch.inference_mode():
            baseline_response = gen_utils.generate_llava_next(
                model, tokenizer, image_tensor, input_ids
            )
        curr_attrib["baseline_response"] = baseline_response

        print("=" * 25)
        print("Baseline response:")
        print(baseline_response)
        print("=" * 25)

        prompt_ids = input_ids[0].to(model.device)
        response_ids = (
            tokenizer(baseline_response, return_tensors="pt", add_special_tokens=False)
            .input_ids[0]
            .to(model.device)
        )

        curr_attrib["token_attrib"] = {}

        with torch.no_grad():
            precomputed_vision_embeds = model.encode_images(image_tensor).detach()
        batched_vision_embeds = precomputed_vision_embeds.requires_grad_()
        batched_baseline = torch.zeros_like(batched_vision_embeds)

        print("> Running attribution")
        for t in range(len(response_ids)):
            target_token_id = response_ids[t].item()
            decoded_token = tokenizer.decode([target_token_id])
            if target_token_id in skip_token_ids or not decoded_token.strip():
                print(f"Skipping token [{t}]: '{decoded_token}'")
                # Add empty attribution
                num_patches = batched_vision_embeds.shape[1]
                curr_attrib["token_attrib"][t] = [
                    decoded_token,
                    torch.zeros(num_patches),
                ]
                continue

            print(f"Computing token [{t}]: '{decoded_token}'...")

            # Construct prefix: Prompt + generated tokens up to t-1
            current_input_ids = torch.cat([prompt_ids, response_ids[:t]]).unsqueeze(0)
            current_attention_mask = torch.ones_like(current_input_ids)
            # Do attribution for the next token after the current prefix.
            with autocast("cuda", dtype=torch.float16):
                attributions = attrib_method.attribute(
                    inputs=batched_vision_embeds,
                    baselines=batched_baseline,
                    target=None,
                    additional_forward_args=(
                        current_input_ids,
                        current_attention_mask,
                        target_token_id,
                    ),
                    n_steps=5,
                    internal_batch_size=1,
                )

            # Remove from GPU and append
            attr_token = attributions.detach().cpu().abs().sum(dim=-1).squeeze(0)
            patch_attributions = attr_token

            curr_attrib["token_attrib"][t] = [decoded_token, patch_attributions]
            print(f"> Computed IG for token [{t}]: '{decoded_token}'")

            # Save test image
            # align_to_preprocess=True -> Plot on squared input seen by CLIP.
            # align_to_preprocess=False -> Stretch to orginal image; not that faithful.
            # smooth_heatmap = interpret_vis.llava_patch_scores_to_heatmap(
            #     patch_attributions,
            #     image,
            #     image_processor=image_processor,
            #     align_to_preprocess=True,
            #     mask_unseen=False,
            # )

            # interpret_vis.save_attribution_overlay(
            #     image=image,
            #     heatmap=smooth_heatmap,
            #     out_path=f"testmap_{decoded_token}",
            #     title=f"Token {decoded_token} — smooth",
            # )

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
            input_ids,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # break

    print("Finished!")
