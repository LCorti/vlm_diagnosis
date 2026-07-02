import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .image_utils import find_closest_aspect_ratio

"""
Minimal (hopefully) utils for plotting attribution maps for InternVL2.
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


def internvl2_stitch_patch_attributions(
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


def internvl2_patch_scores_to_heatmap(
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


"""
Minimal (hopefully) utils for plotting attribution maps for LLaVa-1.6.
"""


def get_llava_patch_grid(num_patches, grid_size=None):
    if grid_size is not None:
        if isinstance(grid_size, int):
            grid_w = grid_h = grid_size
        else:
            grid_w, grid_h = grid_size

        if grid_w * grid_h != num_patches:
            raise ValueError(
                f"grid_size={grid_size} expects {grid_w * grid_h} patches, "
                f"got {num_patches}."
            )
        return grid_w, grid_h

    side = int(np.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(
            f"Cannot infer a square LLaVA patch grid from {num_patches} scores. "
            "Pass grid_size=(grid_w, grid_h) explicitly."
        )

    return side, side


def _get_llava_preprocess_sizes(image_processor=None, input_size=336):
    crop_size = input_size
    shortest_edge = input_size

    if image_processor is None:
        return crop_size, shortest_edge

    processor_crop_size = getattr(image_processor, "crop_size", None)
    if isinstance(processor_crop_size, dict):
        crop_size = processor_crop_size.get("height", crop_size)
    elif processor_crop_size is not None:
        crop_size = processor_crop_size

    processor_size = getattr(image_processor, "size", None)
    if isinstance(processor_size, dict):
        shortest_edge = processor_size.get(
            "shortest_edge",
            processor_size.get("height", processor_size.get("width", shortest_edge)),
        )
    elif processor_size is not None:
        shortest_edge = processor_size

    return int(crop_size), int(shortest_edge)


def _get_center_crop_box_in_original(image, crop_size=336, shortest_edge=336):
    orig_w, orig_h = image.size

    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image size: {image.size}")

    scale = shortest_edge / min(orig_w, orig_h)
    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))

    crop_x0 = max((resized_w - crop_size) / 2.0, 0.0)
    crop_y0 = max((resized_h - crop_size) / 2.0, 0.0)
    crop_x1 = min(crop_x0 + crop_size, resized_w)
    crop_y1 = min(crop_y0 + crop_size, resized_h)

    left = int(round(crop_x0 / scale))
    top = int(round(crop_y0 / scale))
    right = int(round(crop_x1 / scale))
    bottom = int(round(crop_y1 / scale))

    left = max(0, min(left, orig_w - 1))
    top = max(0, min(top, orig_h - 1))
    right = max(left + 1, min(right, orig_w))
    bottom = max(top + 1, min(bottom, orig_h))

    return left, top, right, bottom


def get_llava_preprocess_mask(image, image_processor=None, input_size=336):
    """
    Return a boolean mask for the original-image region seen by LLaVA/CLIP's
    resize + center-crop preprocessing path.
    """
    crop_size, shortest_edge = _get_llava_preprocess_sizes(
        image_processor=image_processor, input_size=input_size
    )
    left, top, right, bottom = _get_center_crop_box_in_original(
        image, crop_size=crop_size, shortest_edge=shortest_edge
    )

    orig_w, orig_h = image.size
    mask = np.zeros((orig_h, orig_w), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def llava_patch_scores_to_heatmap(
    patch_scores,
    image,
    grid_size=None,
    image_processor=None,
    input_size=336,
    normalise=True,
    interpolation=Image.BICUBIC,
    align_to_preprocess=True,
    drop_cls_token=False,
    mask_unseen=True,
):
    """
    Convert LLaVA image-token attribution scores to an image-sized heatmap.

    This matches the attribution format produced by llava_grad_patch.py, where
    each score corresponds to one projected LLaVA/CLIP image token. For
    liuhaotian/llava-v1.6-vicuna-7b with the current preprocessing path this is
    typically a 24x24 grid, i.e. 576 scores.

    Args:
        patch_scores: 1D tensor/array of per-image-token scores.
        image: Original PIL image.
        grid_size: Optional int or (grid_w, grid_h). If omitted, a square grid is
            inferred from len(patch_scores).
        image_processor: Optional LLaVA/CLIP image processor. When provided, its
            crop and resize sizes are used for crop-aware alignment.
        input_size: Fallback CLIP crop/resize size when image_processor is not
            provided.
        normalise: Whether to min-max normalise scores before resizing.
        interpolation: PIL resampling mode for upsampling the heatmap.
        align_to_preprocess: If True, map the heatmap back only onto the square
            region seen by CLIP's resize + center-crop preprocessing. Pixels
            outside the crop are set to zero. If False, resize the grid over the
            full image; this is easier to view but does not match the pixels seen
            by the current LLaVA inference path for non-square images.
        drop_cls_token: Set True if the first score is a CLS token. LLaVA's
            default `mm_vision_select_feature='patch'` excludes CLS, so this is
            False by default.
        mask_unseen: If True and align_to_preprocess is True, return a masked
            array with pixels outside the CLIP crop masked out. This prevents
            zero-valued unseen pixels from being rendered as the low end of the
            colormap.
    """
    if isinstance(patch_scores, torch.Tensor):
        patch_scores = patch_scores.detach().float().cpu().numpy()
    else:
        patch_scores = np.asarray(patch_scores, dtype=np.float32)

    patch_scores = patch_scores.reshape(-1)

    if drop_cls_token:
        patch_scores = patch_scores[1:]

    grid_w, grid_h = get_llava_patch_grid(patch_scores.shape[0], grid_size=grid_size)
    score_grid = patch_scores.reshape(grid_h, grid_w)

    if normalise:
        score_grid = normalise_attribution_map(score_grid)

    orig_w, orig_h = image.size
    heatmap_img = Image.fromarray((score_grid * 255).astype(np.uint8))

    if not align_to_preprocess:
        heatmap_img = heatmap_img.resize((orig_w, orig_h), resample=interpolation)
        return np.asarray(heatmap_img).astype(np.float32) / 255.0

    crop_size, shortest_edge = _get_llava_preprocess_sizes(
        image_processor=image_processor, input_size=input_size
    )
    left, top, right, bottom = _get_center_crop_box_in_original(
        image, crop_size=crop_size, shortest_edge=shortest_edge
    )

    crop_w = right - left
    crop_h = bottom - top
    heatmap_crop = heatmap_img.resize((crop_w, crop_h), resample=interpolation)
    heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)
    heatmap[top:bottom, left:right] = (
        np.asarray(heatmap_crop).astype(np.float32) / 255.0
    )

    if mask_unseen:
        mask = np.ones((orig_h, orig_w), dtype=bool)
        mask[top:bottom, left:right] = False
        heatmap = np.ma.array(heatmap, mask=mask)

    return heatmap


def normalise_attribution_map(attr_map):
    attr_map = attr_map.astype(np.float32)
    attr_map = attr_map - attr_map.min()

    max_val = attr_map.max()
    if max_val > 0:
        attr_map = attr_map / max_val

    return attr_map


def save_attribution_overlay(
    image, heatmap, out_path, alpha=0.45, cmap="jet", title=None, mask=None
):
    image_np = np.asarray(image.convert("RGB"))

    if mask is not None:
        heatmap = np.ma.array(heatmap, mask=~np.asarray(mask, dtype=bool))

    cmap_obj = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    cmap_obj.set_bad((0, 0, 0, 0))

    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    plt.imshow(heatmap, cmap=cmap_obj, alpha=alpha)
    plt.axis("off")

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    # Assume path exists, needs to be taken care of elsewhere
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
