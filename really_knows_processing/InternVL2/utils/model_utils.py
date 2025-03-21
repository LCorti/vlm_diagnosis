import math
import torch

# == == == == == == == ==
# Model Utils
# == == == == == == == ==


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2-1B": 24,
        "InternVL2-2B": 24,
        "InternVL2-4B": 32,
        "InternVL2-8B": 32,
        "InternVL2-26B": 48,
        "InternVL2-40B": 60,
        "InternVL2-Llama3-76B": 80,
    }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


def make_message(template, question):
    message = ""
    # When there is a template, add new line
    if len(template) > 0:
        message = f"{template}\n"
    message += question["question"]

    # If multiple-choice question, include the options
    if question["options"]:
        for o in question["options"]:
            message += "\n- {}: {}".format(o, question["options"][o])

    # Add <image> token to message
    message = f"<image>\n{message}"

    return message
