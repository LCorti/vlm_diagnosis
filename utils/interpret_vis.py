import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .image_utils import find_closest_aspect_ratio

"""
Minimal (hopefully) utils for plotting attribution maps for InterVL2.
"""


def get_internvl2_tile_grid(image, input_size=448, max_num=12):
    og_width, og_height = image.size
    aspect_ratio = og_width / og_height
    target_ratios = set(
        (i, j)
        for n in range(1, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if 1 <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    grid_w, grid_h = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, og_width, og_height, input_size
    )
    return grid_w, grid_h, grid_w * grid_h


def stitch_patch_attributions(
    patch_attr,
    image,
    input_size=448,
    max_num=12,
    use_thumbnail=True,
    normalise=True,
):
    if isinstance(patch_attr, torch.Tensor):
        patch_attr = patch_attr.detach().float().cpu().numpy()

    grid_w, grid_h, num_tiles = get_internvl2_tile_grid(
        image, input_size=input_size, max_num=max_num
    )

    if use_thumbnail and patch_attr.shape[0] == num_tiles + 1:
        patch_attr = patch_attr[:num_tiles]

    if patch_attr.shape[0] != num_tiles:
        raise ValueError(
            f"Expected {num_tiles} tile attribution maps, got {patch_attr.shape[0]}."
        )

    tiled_heatmap = np.zeros(
        (grid_h * input_size, grid_w * input_size),
        dtype=np.float32,
    )

    for i in range(num_tiles):
        row = i // grid_w
        col = i % grid_w

        tiled_heatmap[
            row * input_size : (row + 1) * input_size,
            col * input_size : (col + 1) * input_size,
        ] = patch_attr[i]

    if normalise:
        tiled_heatmap = normalise_attribution_map(tiled_heatmap)

    # Resize back to original image size
    orig_w, orig_h = image.size
    heatmap_img = Image.fromarray((tiled_heatmap * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize((orig_w, orig_h), resample=Image.BICUBIC)

    heatmap = np.asarray(heatmap_img).astype(np.float32) / 255.0
    return heatmap


def patch_scores_to_heatmap(
    patch_scores,
    image,
    input_size=448,
    max_num=12,
    use_thumbnail=True,
    normalise=True,
    interpolation=Image.BICUBIC,
):
    # To use with vision embeddings
    if isinstance(patch_scores, torch.Tensor):
        patch_scores = patch_scores.detach().float().cpu().numpy()
    else:
        patch_scores = np.asarray(patch_scores, dtype=np.float32)

    patch_scores = patch_scores.reshape(-1)

    grid_w, grid_h, num_tiles = get_internvl2_tile_grid(
        image, input_size=input_size, max_num=max_num
    )

    if use_thumbnail and patch_scores.shape[0] == num_tiles + 1:
        patch_scores = patch_scores[:num_tiles]

    if patch_scores.shape[0] != num_tiles:
        raise ValueError(
            f"Expected {num_tiles} tile attribution scores, got {patch_scores.shape[0]}."
        )

    score_grid = patch_scores.reshape(grid_h, grid_w)

    if normalise:
        score_grid = normalise_attribution_map(score_grid)

    orig_w, orig_h = image.size
    heatmap_img = Image.fromarray((score_grid * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize((orig_w, orig_h), resample=interpolation)

    heatmap = np.asarray(heatmap_img).astype(np.float32) / 255.0
    return heatmap


def normalise_attribution_map(attr_map):
    attr_map = attr_map.astype(np.float32)
    attr_map = attr_map - attr_map.min()

    max_val = attr_map.max()
    if max_val > 0:
        attr_map = attr_map / max_val

    return attr_map


def save_attribution_overlay(
    image, heatmap, out_path, alpha=0.45, cmap="jet", title=None
):
    image_np = np.asarray(image.convert("RGB"))

    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha)
    plt.axis("off")

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    # Assume path exists, needs to be taken care of elsewhere
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
