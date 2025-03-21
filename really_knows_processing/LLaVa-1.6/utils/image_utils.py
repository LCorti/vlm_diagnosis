from PIL import Image


def load_image(image_file):
    return Image.open(image_file).convert("RGB")
