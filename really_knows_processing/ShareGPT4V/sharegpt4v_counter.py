import argparse
import json
import os
import torch
import yaml

from PIL import Image

from share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from share4v.conversation import SeparatorStyle, conv_templates
from share4v.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from share4v.model.builder import load_pretrained_model
from share4v.utils import disable_torch_init
from transformers import GenerationConfig

PROMPT_FILE = '../prompts_rk_counter.yaml'
MODEL_KEY = 'sharegpt4v'

# Kwargs parser
def parse_args():
    parser = argparse.ArgumentParser(description="sharegpt4v_counter")
    parser.add_argument("--ds_name", required=True, help="Name of the experiment.")
    parser.add_argument("--ds_folder", required=True, help="Path to the dataset folder.")
    parser.add_argument("--questions_file", required=True, help="Path to the file with questions.")
    parser.add_argument("--constraint_folder", required=True, help="Path to folder with RK constraints.")
    parser.add_argument("--out_dir", help="Path to folder to save the responses from the LMM.", default="responses_counter")
    parser.add_argument("--do_sample", default=False)
    parser.add_argument("--num_beams", type=int, default=1)
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

# Utility functions for ShareGPT4V
def load_image(image_file):
    return Image.open(image_file).convert('RGB')

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

def get_stop_str(conv):
    return conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

def get_stopping_criteria(input_ids, stop_str, tokenizer):
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

# Generation utils
def get_prompts():
    with open(PROMPT_FILE, 'r') as f:
        prompts = yaml.load(f, Loader=yaml.FullLoader)
    return prompts

def load_rk_constraints(file_path):
    with open(file_path, 'r') as fp:
        rk_cons = json.load(fp)
    return rk_cons

def get_gen_config(do_sample, num_beams, use_cache, temperature, max_new_tokens):
    gen_config = GenerationConfig.from_dict({
        'do_sample': do_sample,
        'num_beams': num_beams,
        'temperature': temperature,
        'use_cache': use_cache,
        'max_new_tokens': max_new_tokens
    })
    return gen_config

def gen_response(model, input_ids, image_tensor, gen_config, stopping_criteria):
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            generation_config=gen_config,
            stopping_criteria=[stopping_criteria])
    return output_ids

def add_conv_step(conv, user_q, prev_resp=None):
    # conv.roles[0] is user
    # conv.roles[1] is assistant
    if prev_resp:
        # Remove last ['Assistant', None] used at the previous generation step
        conv.messages.pop(-1)
        # Add assistant's generated response
        conv.append_message(conv.roles[1], prev_resp)
    conv.append_message(conv.roles[0], user_q)
    conv.append_message(conv.roles[1], None)
    return conv

# Utils for saving data
def create_out_folder(out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

def save_counter_data(attribs, out_dir, ds_name):
    file_path = '{}/resp_{}_counter_rk.jsonl'.format(out_dir, ds_name)
    with open(file_path, 'w') as f:
        for a in attribs:
            json.dump(a, f)
            f.write('\n')

def main():
    print('Loading model...')
    args = parse_args()
    create_out_folder(args.out_dir)

    # Model loading parameters
    model_path = "Lin-Chen/ShareGPT4V-7B"
    model_base = None
    model_name = get_model_name_from_path(model_path)
    conv_mode = set_conv_mode(model_name)

    # Prepare generation config
    gen_config = get_gen_config(args.do_sample,
                                args.num_beams,
                                args.use_cache,
                                args.temperature,
                                args.max_new_tokens)
    print(gen_config.validate())

    # Load model
    disable_torch_init() # This is part of the original implementation

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        device_map='auto',
    )
    model.half()
    print("Model loaded.")

    # Load data
    print('Loading data...')
    print('Loading questions from {}'.format(args.questions_file))
    questions = []
    with open(args.questions_file, 'r') as f:
        questions = json.load(f)

    prompts = get_prompts()
    print(prompts.keys())

    print('Running inference...')
    all_model_resp_counter = []

    # == == == == Get attributions over input image == == == ==
    for q in questions:
        print('Current question ID: {}'.format(q['question_id']))

        # Load RK constraints
        file_path = f'{args.constraint_folder}/graph_alt_text_{q["question_id"]}.json'
        try:
            rk_constraints = load_rk_constraints(file_path)
        except:
            print(f'File not found: {file_path}.')
            continue

        img_path = os.path.join(args.ds_folder, q['img_path'])

        # Format text input
        user_q = prompts['question_template'][MODEL_KEY][args.ds_name]
        if len(user_q) > 0:
            user_q += '\n'
        user_q += q['question']
        print('-- Prompt: {}'.format(user_q))
        if q['options']:
            for k in q['options']:
                user_q += '\n- {}: {}'.format(k, q['options'][k])

        # Iterate over different sizes of constraints
        for size in rk_constraints:
            print(f'-- Working on constraints of size {size}...')

            # For each 'size' iterate over all constraints settings
            for rk_const in rk_constraints[size]:
                # Prepare object to store info
                model_rk = {
                    'question_id': q['question_id'],
                    'size_const': size,
                    'response': '',
                    'constraints': ''
                }
                model_rk['constraints'] = rk_const

                # Loading image
                image = load_image(img_path)
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to('cuda:0')
        
                # Include contraint instruction and data
                constrained_user_q = f'{user_q}\n {prompts["constraint_template"][MODEL_KEY]}'
                constrained_user_q = f'{constrained_user_q}\n {'\n'.join(rk_const)}'
                constrained_user_q = format_query(constrained_user_q, model)
                conv = get_conv_template(conv_mode)
                conv = add_conv_step(conv, constrained_user_q)
                prompt = conv.get_prompt()
                stop_str = get_stop_str(conv)

                # Prepare input token ids
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                stopping_criteria = get_stopping_criteria(input_ids, stop_str, tokenizer)

                # Get response
                output_ids = gen_response(model, input_ids, image_tensor, gen_config, stopping_criteria)
                model_rk['response'] = decode_tokens(input_ids, output_ids, tokenizer, stop_str)
                # print('Response: {}'.format(model_rk['response']))
                # print('='*25)

                # Free up memory
                del image_tensor
                del input_ids
                torch.cuda.empty_cache()

                all_model_resp_counter.append(model_rk)
        # == == == == == == == == == == == == == == == == == == == 

            # Saving results to file once computations for each 'size' are done
            print('... Saving data ...')
            save_counter_data(all_model_resp_counter, args.out_dir, args.ds_name)
            print('... Data saved.')

if __name__ == "__main__":
    main()