import os
import json
import torch
import argparse
import yaml

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

PROMPT_FILE = '../prompts_rk_counter.yaml'
MODEL_KEY = 'minigpt4'

# ================================================================== #
# The original code can be found in demo.py.                         #
# This is a quick edit to run inference directly from command line.  #
# ================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(description="minigpt4_inference")
    parser.add_argument("--cfg-path", required=True, help="Path to configuration file.")
    parser.add_argument("--ds_name", required=True, help="Name of the experiment.")
    parser.add_argument("--ds_folder", required=True, help="Path to the dataset folder.")
    parser.add_argument("--questions_file", required=True, help="Path to the file with questions.")
    parser.add_argument("--constraint_folder", required=True, help="Path to folder with RK constraints.")
    parser.add_argument("--out_dir", help="Name of the output file with responses from the LLM.", default="responses_counter")
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify the gpu to load the model.")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", help="Maximum number of tokens to generate", default=256)
    parser.add_argument("--temperature", type=int, default=0.1)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def get_chat_state():
    return CONV_VISION_Vicuna0.copy()

def load_img(chat, img_path, chat_state):
    if img_path is None:
        print("Image not found")
        return
    
    img_list = []
    _ = chat.upload_img(img_path, chat_state, img_list)
    chat.encode_img(img_list)
    return chat_state, img_list

# Generation utils

def ask(chat, user_message, chat_state):
    if len(user_message) == 0:
        print("No message received!")
        return
    chat.ask(user_message, chat_state)
    return chat_state

def get_response(chat, img_list, chat_state, 
                 num_repeats=1, num_beams=1, temperature=0.1,
                 max_new_tokens=256):
    all_responses = []
    for _ in range(num_repeats):
        response = chat.answer(conv=chat_state,
                               img_list=img_list,
                               num_beams=num_beams,
                               temperature=temperature,
                               max_new_tokens=max_new_tokens,
                               max_length=2000)[0]
        all_responses.append(response)

    if num_repeats == 1:
        return all_responses[0]
    else:
        return all_responses

def get_prompts():
    with open(PROMPT_FILE, 'r') as f:
        prompts = yaml.load(f, Loader=yaml.FullLoader)
    return prompts

def load_rk_constraints(file_path):
    try: 
        with open(file_path, 'r') as fp:
            rk_cons = json.load(fp)
        return rk_cons
    except:
        print(f'File not found: {file_path}')
        return None

# Utils for saving data
def create_out_folder(out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

def load_existing_data(file_path):
    with open(file_path, 'r') as fp:
        data = [json.loads(l) for l in fp]
    return data

def save_counter_data(attribs, out_dir, ds_name):
    file_path = '{}/resp_{}_counter_rk.jsonl'.format(out_dir, ds_name)
    with open(file_path, 'w') as f:
        for a in attribs:
            json.dump(a, f)
            f.write('\n')

def main():
    print('Loading model...')
    args = parse_args()
    cfg = Config(args)
    create_out_folder(args.out_dir)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Model loaded.')

    print('Loading data...')
    print('Loading questions from {}'.format(args.questions_file))
    questions = []
    with open(args.questions_file, 'r') as f:
        questions = json.load(f)

    prompts = get_prompts()

    print('Running inference...')

    # Check whether the expected output file exists already.
    # If so, load that data and resume inference from there.
    file_name = '{}/resp_{}_counter_rk.jsonl'.format(args.out_dir, args.ds_name)
    if os.path.exists(file_name):
        all_model_resp_counter = load_existing_data(file_name)
    else:
        all_model_resp_counter = []

    print(f'Already have {len(all_model_resp_counter)} counterfactual responses.')

    # == == == == Get responses on restricted information == == == ==
    for q in questions:
        print('Current question ID: {}'.format(q['question_id']))

        # Load RK constraints
        file_path = f'{args.constraint_folder}/graph_alt_text_{q["question_id"]}.json'
        rk_constraints = load_rk_constraints(file_path)

        # If no constraints are found (due to the model), skip to the next iteration
        if rk_constraints is None:
            continue

        img_path = os.path.join(args.ds_folder, q['img_path'])

        # Format text input
        user_q = prompts['question_template'][MODEL_KEY][args.ds_name]
        if len(user_q) > 0:
            user_q += '\n'
        user_q += q['question']
        if q['options']:
            for k in q['options']:
                user_q += '\n- {}: {}'.format(k, q['options'][k])

        # Iterate over different sizes of constraints
        for size in rk_constraints:
            
            # Skip iteration if data for a <question, size> combination exists
            latest_pair = next((rc for rc in all_model_resp_counter
                                if rc['question_id']==q['question_id'] and
                                rc['size_const']==size), None)
            if latest_pair:
                continue

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
        
                # Loading image and conversation
                conv = get_chat_state()
                conv, img_list = load_img(chat, img_path, conv)

                # Get response
                conv = ask(chat, user_q, conv)
                model_rk['response'] = get_response(chat, img_list, conv,
                                                    num_repeats=1,
                                                    num_beams=args.num_beams,
                                                    temperature=args.temperature,
                                                    max_new_tokens=args.max_new_tokens)
                # print('Response: {}'.format(model_rk['response']))
                # print('='*25)

                # Free up memory
                del img_list
                torch.cuda.empty_cache()
                
                all_model_resp_counter.append(model_rk)
    
            # Save results to file
            print('... Saving data ...')
            save_counter_data(all_model_resp_counter, args.out_dir, args.ds_name)
            print('... Data saved.')
    
if __name__ == "__main__":
    main()