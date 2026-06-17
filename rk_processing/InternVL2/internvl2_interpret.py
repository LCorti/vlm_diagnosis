import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from captum.attr import FeatureAblation
from captum.attr._core.llm_attr import LLMAttribution as captum_llm_attr
from captum.attr._utils.interpretable_input import ImageMaskInput
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
GRID_ROWS = 8
GRID_COLS = 10


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


# Create grid-based segmentation mask
def create_grid_mask(image, rows, cols):
    """
    Create a grid mask that divides the image into rows x cols segments.

    Args:
        image: PIL Image
        rows: Number of rows in the grid
        cols: Number of columns in the grid

    Returns:
        Tensor of shape (height, width) with integer segment IDs
    """
    width, height = image.size
    mask = torch.zeros((height, width), dtype=torch.int32)

    h_step = height // rows
    w_step = width // cols

    for row in range(rows):
        for col in range(cols):
            # Calculate segment boundaries
            y_start = row * h_step
            y_end = height if row == rows - 1 else (row + 1) * h_step
            x_start = col * w_step
            x_end = width if col == cols - 1 else (col + 1) * w_step

            # Assign unique segment ID
            segment_id = row * cols + col
            mask[y_start:y_end, x_start:x_end] = segment_id

    return mask


def _fallback_pretty_tokens(token_ids, tokenizer):
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return tokenizer.convert_ids_to_tokens(token_ids)


class InternVL2CaptumWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _broadcast_kwargs(self, kwargs):
        """Expands text tensors, flattens multi-patch batches, and synthesizes flags."""
        if "pixel_values" in kwargs:
            # 1. Distinguish between Captum's batched ablations and the unbatched baseline
            if kwargs["pixel_values"].dim() == 5:
                # Batched Phase (Attribution): [B, P, C, H, W]
                B, P, C, H, W = kwargs["pixel_values"].shape
                kwargs["pixel_values"] = kwargs["pixel_values"].view(B * P, C, H, W)
                b_vit = B * P
                b_text = B
            elif kwargs["pixel_values"].dim() == 4:
                # Baseline Phase (Generation): [P, C, H, W]
                P = kwargs["pixel_values"].shape[0]
                b_vit = P
                b_text = 1
            else:
                b_vit = kwargs["pixel_values"].shape[0]
                b_text = 1

            # 2. Synthesize image_flags for the total flattened visual patches
            if "image_flags" not in kwargs or kwargs["image_flags"] is None:
                kwargs["image_flags"] = torch.ones(
                    (b_vit, 1), dtype=torch.long, device=kwargs["pixel_values"].device
                )

            # 3. Expand text sequences matching the ablation batch size (B)
            if "input_ids" in kwargs and kwargs["input_ids"].shape[0] != b_text:
                kwargs["input_ids"] = (
                    kwargs["input_ids"].expand(b_text, -1).contiguous()
                )

                if "attention_mask" in kwargs:
                    kwargs["attention_mask"] = (
                        kwargs["attention_mask"].expand(b_text, -1).contiguous()
                    )

                if "labels" in kwargs and kwargs["labels"] is not None:
                    kwargs["labels"] = kwargs["labels"].expand(b_text, -1).contiguous()

        return kwargs

    def forward(self, *args, **kwargs):
        kwargs = self._broadcast_kwargs(kwargs)
        outputs = self.model(*args, **kwargs)

        target_device = (
            kwargs["input_ids"].device if "input_ids" in kwargs else self.model.device
        )

        if hasattr(outputs, "logits") and outputs.logits is not None:
            outputs.logits = outputs.logits.to(target_device)
        elif isinstance(outputs, tuple):
            outputs = tuple(
                out.to(target_device) if isinstance(out, torch.Tensor) else out
                for out in outputs
            )

        return outputs

    def generate(self, *args, **kwargs):
        kwargs = self._broadcast_kwargs(kwargs)
        # The generate method does not really want this parameter.
        if "image_flags" in kwargs:
            kwargs.pop("image_flags")

        input_ids = kwargs.get("input_ids")
        outputs = self.model.generate(*args, **kwargs)

        if input_ids is not None and isinstance(outputs, torch.Tensor):
            outputs = torch.cat([input_ids, outputs], dim=1)

        return outputs

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
    rationale_template = gen_utils.get_rationale_template()
    out_format_template = gen_utils.get_out_format_template()

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
    print("Model loaded.")

    # Wrap model
    wrapped_model = InternVL2CaptumWrapper(model)
    fa = FeatureAblation(wrapped_model)
    llm_attr = captum_llm_attr(fa, tokenizer)
    print("Model wrapped.")

    # Load data
    print("Loading data...")
    print("Loading questions from {}".format(questions_file))
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

        # Loading image
        img_path = base_dir.joinpath(curr_q["img_path"])
        print("-- Loading image {}".format(img_path))
        image = image_utils.load_img(img_path)

        message = make_message(question_template, curr_q)
        image_tensor = image_utils.get_image_tensor(image).to(torch.bfloat16).cuda()
        # Get baseline response
        baseline_response, _ = gen_utils.generate_internvl2(
            model, tokenizer, image_tensor, message
        )
        print("=" * 25)
        print("Baseline response:")
        print(baseline_response)
        print("=" * 25)

        # Prepare image segmentation
        # Make this denser if needed.
        grid_mask = create_grid_mask(image, rows=GRID_ROWS, cols=GRID_COLS)
        print(f"Grid mask shape: {grid_mask.shape}")
        print(f"Number of segments: {len(torch.unique(grid_mask))}")
        # Uncomment to preview
        # grid_preview = ImageMaskInput(
        #     image=image, mask=grid_mask, processor_fn=lambda x: x
        # )
        # fig, _ = grid_preview.plot_mask_overlay()
        # fig.savefig("grid_preview.png")

        # Do attribution
        def processor_fn(input_image):
            # Use InternVL2's vision processing pipeline
            image_tensor = (
                image_utils.get_image_tensor(input_image).to(torch.bfloat16).cuda()
            )
            num_patches = image_tensor.shape[0]

            # Inject image tokens into input
            IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
            if getattr(model, "img_context_token_id", None) is None:
                model.img_context_token_id = tokenizer.convert_tokens_to_ids(
                    IMG_CONTEXT_TOKEN
                )

            num_image_tokens = model.num_image_token * num_patches
            image_token_sequence = (
                "<img>" + (IMG_CONTEXT_TOKEN * num_image_tokens) + "</img>"
            )

            # Format message in conversation
            conversation = [{"role": "user", "content": message}]
            prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            prompt = prompt.replace("<image>", image_token_sequence)

            # Use InternVL2's tokeniser to get input_ids and attention_mask
            model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            return {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "pixel_values": image_tensor,
            }

        # Use black baseline (default is white? What's the difference?)
        grid_input = ImageMaskInput(
            image,
            mask=grid_mask,
            processor_fn=processor_fn,
            # baseline=Image.fromarray(np.zeros_like(np.array(image)), "RGB"),
        )
        print("> Running attribution")
        patch_target = "captum.attr._core.llm_attr._convert_ids_to_pretty_tokens"
        with patch(patch_target, new=_fallback_pretty_tokens):
            grid_attr_result = llm_attr.attribute(
                grid_input, forward_in_tokens=False, **gen_config
            )
        fig, _ = grid_attr_result.plot_image_heatmap(show_legends=False)
        fig.savefig(f"grid_attr_{curr_q['question_id']}.png")

        # TODO: look into returned empty tensor
        # print(grid_attr_result)

        # TODO: fix this one
        # all_rk.append(curr_rk)
        # all_parsed_rk.append(gen_utils.parse_raw_rk(curr_rk))

        # Free up memory
        # del image_tensor
        torch.cuda.empty_cache()

        # TODO: remove this later
        break
        # == == == == == == == == == == == == == == == == == == ==

    # Saving results to file
    # print("... Saving raw Really Knows ...")
    # data_io.save_jsonl(all_rk, full_raw_out_f)
    # print("... Saving parsed Really Knows ...")
    # data_io.save_jsonl(all_parsed_rk, full_parsed_out_f)
    # print("All data saved.")
