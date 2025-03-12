import argparse
import json
import math
import os
import torch
import torchvision.transforms as T
import yaml

from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PROMPT_VERSION = 4
PROMPT_FILE = '../prompts_rk_v{}.yaml'.format(PROMPT_VERSION)
MODEL_KEY = 'internvl2'

# Kwargs parser
def parse_args():
    parser = argparse.ArgumentParser(description="internvl2_self_expl")
    parser.add_argument("--ds_name", required=True, help="Name of the experiment.")
    parser.add_argument("--ds_folder", required=True, help="Path to the dataset folder.")
    parser.add_argument("--questions_file", required=True, help="Path to the file with questions.")
    parser.add_argument("--out_dir", help="Path to folder to save the responses from the LMM.", default='results')
    parser.add_argument("--do_sample", default=False)
    parser.add_argument("--temperature", help="Model temperature.", default=0.1)
    parser.add_argument("--max_new_tokens", help="Maximum number of tokens to generate", default=256)
    parser.add_argument("--use_cache", default=True)
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

# Utility functions for InternlVL2
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def format_query(q):
    return '<image>\n{}'.format(q)

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# Generation utils
def get_prompts():
    with open(PROMPT_FILE, 'r') as f:
        prompts = yaml.load(f, Loader=yaml.FullLoader)
    return prompts

def get_gen_config(do_sample, temperature, max_new_tokens):
    gen_config = dict(
        do_sample = do_sample,
        temperature = temperature,
        max_new_tokens = max_new_tokens
    )
    return gen_config

def gen_response(model, text_input, image_tensor, gen_config, tokenizer, history=None):
    return model.chat(tokenizer, 
                      image_tensor, 
                      text_input, 
                      gen_config,
                      history = history,
                      return_history = True)

# Utils for saving data
def create_out_folder(out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

def save_attrib_data(attribs, out_dir, ds_name):
    file_path = '{}/exp_{}_rk.jsonl'.format(out_dir, ds_name)
    with open(file_path, 'w') as f:
        for a in attribs:
            json.dump(a, f)
            f.write('\n')

def main():
    print('Loading model...')
    args = parse_args()
    out_dir = '{}_v{}'.format(args.out_dir, PROMPT_VERSION)
    create_out_folder(out_dir)

    # Model loading
    model_name = "OpenGVLab/InternVL2-8B"

    # Prepare generation config
    gen_config = get_gen_config(args.do_sample,
                                args.temperature,
                                args.max_new_tokens)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if torch.cuda.device_count() > 1:
        print('Found {} GPUs; splitting model...'.format(torch.cuda.device_count()))
        device_map = split_model('InternVL2-8B')
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map).eval()
    else:
        print('Found 1 GPU.')
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
    print("Model loaded.")

    # Load data
    print('Loading data...')
    print('Loading questions from {}'.format(args.questions_file))
    questions = []
    with open(args.questions_file, 'r') as f:
        questions = json.load(f)

    prompts = get_prompts()

    print('Running inference and computing explanations...')
    all_model_rk = []

    # == == == == Get responses and self-explanations == == == ==
    for q in questions:
        print('Current question ID: {}'.format(q['question_id']))
        print('-- Text: {}'.format(q['question']))

        # Prepare object to store info
        model_rk = {
            'question_id': q['question_id'],
        }

        # Loading image
        img_path = os.path.join(args.ds_folder, q['img_path'])
        print('-- Loading image {}'.format(img_path))
        image_tensor = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
        
        # Format text input
        user_q = prompts['question_template'][MODEL_KEY][args.ds_name]
        if len(user_q) > 0:
            user_q += '\n'
        user_q += q['question']
        print('-- Prompt: {}'.format(user_q))

        if q['options']:
            for k in q['options']:
                user_q += '\n- {}: {}'.format(k, q['options'][k])

        user_q = format_query(user_q)

        # First generation step: get answer from the model
        model_rk['response'], history = gen_response(model,
                                                     user_q,
                                                     image_tensor,
                                                     gen_config,
                                                     tokenizer)
        print(model_rk['response'])
        print('='*25)

        # Second generation step: get unstructured rationales for model output
        expl_q = prompts['rationale_template'][MODEL_KEY]

        model_rk['rationales'], history = gen_response(model,
                                                       expl_q,
                                                       image_tensor,
                                                       gen_config,
                                                       tokenizer,
                                                       history=history)
        print(model_rk['rationales'])
        print('='*25)

        # Third step: triple extraction and structuring from rationales
        struct_expl_q = prompts['out_format_template'][MODEL_KEY]
        model_rk['triples'], history = gen_response(model,
                                                    struct_expl_q,
                                                    image_tensor,
                                                    gen_config,
                                                    tokenizer,
                                                    history=history)
        print(model_rk['triples'])
        print('='*25)

        all_model_rk.append(model_rk)

        # Free up memory
        del image_tensor
        torch.cuda.empty_cache()
    # == == == == == == == == == == == == == == == == == == == 

        # Saving results to file
        print('... Saving data ...')
        save_attrib_data(all_model_rk, out_dir, args.ds_name)
        print('... Data saved.')

if __name__ == "__main__":
    main()