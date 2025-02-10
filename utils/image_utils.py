from PIL import Image, ImageDraw

# Utils functions taken from the official implementation of IETrans.
# https://github.com/waxnkw/IETrans-SGG.pytorch


def load_img(img_path):
    img = Image.open(img_path)
    return img


def save_img(pil_img, file_path):
    rgb_img = pil_img.convert("RGB")
    rgb_img.save(file_path)


def resize_img(pil_img):
    size = get_size(pil_img.size)
    pil_img = pil_img.resize(size)
    return pil_img


def get_size(image_size):
    min_size = 600
    max_size = 1000
    w, h = image_size
    size = min_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return (ow, oh)


def draw_single_box(pic, box, color="red", draw_info=None):
    draw = ImageDraw.Draw(pic)
    draw.rectangle(
        (
            (box["top_left_x"], box["top_left_y"]),
            (box["bottom_right_x"], box["bottom_right_y"]),
        ),
        outline=color,
    )
    if draw_info:
        draw.rectangle(
            (
                (box["top_left_x"], box["top_left_y"]),
                (box["top_left_x"] + 50, box["top_left_y"] + 10),
            ),
            fill=color,
        )
        info = draw_info
        draw.text((box["top_left_x"], box["top_left_y"]), info)


def draw_bboxes(pil_img, bboxes):
    size = get_size(pil_img.size)
    pil_img = pil_img.resize(size)

    for idx, bbox in enumerate(bboxes):
        draw_single_box(
            pil_img, bboxes[idx], draw_info=bbox["bb_label"]["bb_label_full"]
        )
    return pil_img


def list_relations(relations):
    rel_triples = []
    for idx, rel in enumerate(relations):
        curr_rel = "{} \t: {}, {}, {}".format(
            idx,
            rel["from_concept"]["bb_label_full"],
            rel["rel_label"]["rel_label_text"],
            rel["to_concept"]["bb_label_full"],
        )
        rel_triples.append(curr_rel)
    return rel_triples
