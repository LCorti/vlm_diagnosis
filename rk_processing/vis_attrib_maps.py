import argparse
import sys
from pathlib import Path

import torch

module_path = str(Path("..", "..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

import utils.data_io as data_io
import utils.image_utils as image_utils
import utils.interpret_vis as interpret_vis


# Kwargs parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse attribution visualisation params."
    )
    parser.add_argument("--model_name", required=True, help="Name of the VLM.")
    parser.add_argument("--ds_name", required=True, help="Name of the dataset.")
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
    args = parse_args()
    model_name = args.model_name
    ds_name = args.ds_name
    base_dir = Path(__file__).parent.parent
    # Save here for now
    out_dir = Path(__file__).parent.joinpath(f"heatmaps_{model_name}_{ds_name}")
    data_io.make_dir(out_dir)

    # TODO: load attribution data for the correct model x dataset combo
    attrib_data = []

    for curr_q in attrib_data:
        print(f">> Processing question: {curr_q['question_id']}")
        # Make directory for current question
        curr_dir = out_dir.joinpath(curr_q["question_id"])
        data_io.make_dir(curr_dir)

        # Skip showing baseline response
        # Load image
        img_path = base_dir.joinpath(curr_q["img_path"])
        image = image_utils.load_img(img_path)

        # Plot maps for individual tokens
        for token_pos, token_info in curr_q["token_attrib"].items():
            # token_info is [token, attribution data]
            print(f"Plotting attribution for token {token_pos} '{token_info[0]}'...")

            # Save as <token_pos>_<token>.pdf
            out_path = curr_dir.joinpath(f"{token_pos}_{token_info[0]}")

            # -----------------
            # Block-y heatmaps
            # -----------------
            block_heatmap = interpret_vis.internvl2_stitch_patch_attributions(
                token_info[1], image=image, use_thumbnail=True
            )
            interpret_vis.save_attribution_overlay(
                image=image,
                heatmap=block_heatmap,
                out_path=out_path,
                title=f"Token {token_info[0]} — block",
            )

            # -----------------
            # Smooth heatmaps
            # -----------------
            smooth_heatmap = interpret_vis.internvl2_patch_scores_to_heatmap(
                token_info[1], image=image, use_thumbnail=True
            )
            interpret_vis.save_attribution_overlay(
                image=image,
                heatmap=smooth_heatmap,
                out_path=out_path,
                title=f"Token {token_info[0]} — smooth",
            )

    print("End.")
