import argparse
import json
import os
import requests
import torch
from io import BytesIO
from PIL import Image

from share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from share4v.conversation import SeparatorStyle, conv_templates
from share4v.mm_utils import (KeywordsStoppingCriteria,
                              get_model_name_from_path, tokenizer_image_token)
from share4v.model.builder import load_pretrained_model
from share4v.mm_utils import get_model_name_from_path
from share4v.utils import disable_torch_init

def parse_args():
    parser = argparse.ArgumentParser(description="llavanext_inference")
    parser.add_argument("--ds_name", required=True, help="Name of the experiment.")
    parser.add_argument("--ds_folder", required=True, help="Path to the dataset folder.")
    parser.add_argument("--questions_file", required=True, help="Path to the file with questions.")
    parser.add_argument("--out_dir", help="Path to folder to save the responses from the LMM.", default='results')
    parser.add_argument("--do_sample", default=True)
    parser.add_argument("--temperature", help="Model temperature.", default=0.2)
    parser.add_argument("--max_new_tokens", help="Maximum number of tokens to generate", default=1024)
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

# Image loader
def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

# Input formatting
def format_query(q, model):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
            DEFAULT_IM_END_TOKEN + '\n' + q
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + q
    
    return qs

def set_conv_mode(model_name):
    if 'llama-2' in model_name.lower():
        conv_mode = "share4v_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "share4v_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "share4v_v0"
    
    return conv_mode

def get_conv_template(conv_mode):
    return conv_templates[conv_mode].copy()

# Output formatting
def get_stop_str(conv):
    return conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

def get_stopping_criteria(input_ids, tokenizer, stop_str):
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    return stopping_criteria

def decode_tokens(input_ids, output_ids, tokenizer, stop_str):
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()

    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    
    return outputs

def main():
    print('Initializing model...')
    args = parse_args()

    model_path = "Lin-Chen/ShareGPT4V-7B"
    model_base = None
    model_name = get_model_name_from_path(model_path)
    conv_mode = set_conv_mode(model_name)

    # Load model
    disable_torch_init() # This is part of the original implementation
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name
    )
    print('Initialisation finished.')

    print('Loading data...')
    print('Loading questions from {}'.format(args.questions_file))
    questions = []
    with open(args.questions_file, 'r') as f:
        questions = json.load(f)

    print('Running inference...')
    all_responses = []

    for q in questions:
        print('Current question ID: {}'.format(q['question_id']))
        print('-- Text: {}'.format(q['question']))
        
        img_path = os.path.join(args.ds_folder, q['img_path'])
        print('-- Loading image {}'.format(img_path))
        image = load_image(img_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        # format input
        user_q = format_query(q['question'], model)
        conv = get_conv_template(conv_mode)
        conv.append_message(conv.roles[0], user_q) # user query
        conv.append_message(conv.roles[1], None) # agent

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = get_stop_str(conv)
        stopping_criteria = get_stopping_criteria(input_ids, tokenizer, stop_str)

        # Generation step
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=args.do_sample,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=args.use_cache,
                stopping_criteria=[stopping_criteria])

        response = decode_tokens(input_ids, output_ids, tokenizer, stop_str)
        all_responses.append({
            'question_id': q['question_id'],
            'question': q['question'],
            'response': response
        })

        # Free up memory
        del image_tensor
        torch.cuda.empty_cache()
    
    res_folder = args.out_dir
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    
    with open('{}/exp_{}_responses.jsonl'.format(res_folder, args.ds_name), 'w') as f:
        for r in all_responses:
            json.dump(r, f)
            f.write('\n')

if __name__ == "__main__":
    main()