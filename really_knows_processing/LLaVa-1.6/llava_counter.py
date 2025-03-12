import os
import json
import torch
import argparse
import yaml

from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                             DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from PIL import Image
from transformers import GenerationConfig

# LLaVa output includes the user query.
# This is used to find the response and extract it.
SPLIT_KW = 'ASSISTANT:'
PROMPT_FILE = '../prompts_rk_counter.yaml'
MODEL_KEY = 'llava-next'

# Kwargs parser
def parse_args():
    parser = argparse.ArgumentParser(description="llava_self_expl")
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

def load_image(image_file):
    return Image.open(image_file).convert("RGB")

# Utility functions for LLaVa-Next
def format_query(q, model):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
            DEFAULT_IM_END_TOKEN + '\n' + q
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + q
    return qs

def set_conv_mode(model_name):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    
    return conv_mode

def get_conv_template(conv_mode):
    return conv_templates[conv_mode].copy()

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
        'max_new_tokens': max_new_tokens,
        'cache_position': None
    })
    return gen_config

def gen_response(model, input_ids, image_tensor, gen_config):
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            generation_config=gen_config)

    return output_ids

def get_stop_str(conv):
    return conv.sep

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

def add_conv_step(conv, user_q, prev_resp=None):
    if prev_resp:
        # If a previous response is passed, add it to the conversation
        # Remove the 'old' last message
        conv.messages.pop(-1)
        # Append the agent's response at previous turn
        conv.append_message(conv.roles[1], prev_resp)
    
    # Append new user message
    conv.append_message(conv.roles[0], user_q)
    # Append new empty agent message
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

    print('Initializing model...')
    args = parse_args()
    create_out_folder(args.out_dir)

    model_path = "liuhaotian/llava-v1.6-vicuna-7b"
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

    disable_torch_init() # from authors' implementation

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name
    )
    # model.half() ??
    print(getattr(model.config, "image_aspect_ratio", None))
    print("Model loaded.")

    print('Loading data...')
    print('Loading questions from {}'.format(args.questions_file))
    questions = []
    with open(args.questions_file, 'r') as f:
        questions = json.load(f)

    prompts = get_prompts()

    print('Running inference...')
    all_model_resp_counter = []

    # == == == == Get responses on restricted information == == == ==
    for q in questions:
        print('Current question ID: {}'.format(q['question_id']))
        print('-- Text: {}'.format(q['question']))

        # Load RK constraints
        file_path = f'{args.constraint_folder}/graph_alt_text_{q["question_id"]}.json'
        try:
            rk_constraints = load_rk_constraints(file_path)
        except:
            print(f'File not found: {file_path}.')
            continue

        img_path = os.path.join(args.ds_folder, q['img_path'])

        # Format input question
        user_q = prompts['question_template'][MODEL_KEY][args.ds_name]
        if len(user_q) > 0:
            user_q += '\n'
        user_q += q['question']
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

                # Prepare input token ids
                input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                    .unsqueeze(0)
                    .cuda()
                )

                # Get response
                output_ids = gen_response(model, input_ids, image_tensor, gen_config)
                model_rk['response'] = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                # print('Response: {}'.format(model_rk['response']))
                # print('='*25)

                # Free up memory
                del image_tensor
                del input_ids
                torch.cuda.empty_cache()                
            
                all_model_resp_counter.append(model_rk)
        # == == == == == == == == == == == == == == == == == == == 

            # Saving results to file
            print('... Saving data ...')
            save_counter_data(all_model_resp_counter, args.out_dir, args.ds_name)
            print('... Data saved.')

if __name__ == "__main__":
    main()