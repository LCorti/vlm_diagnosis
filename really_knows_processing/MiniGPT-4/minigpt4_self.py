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

PROMPT_VERSION = 4
PROMPT_FILE = '../prompts_rk_v{}.yaml'.format(PROMPT_VERSION)
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
    parser.add_argument("--out_dir", help="Name of the output file with responses from the LLM.", default="results")
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

    print('Initializing model...')
    args = parse_args()
    cfg = Config(args)
    out_dir = '{}_v{}'.format(args.out_dir, PROMPT_VERSION)
    create_out_folder(out_dir)

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
    print(prompts.keys())

    print('Running inference and computing explanations...')
    all_model_rk = []

    for q in questions:
        print('Current question ID: {}'.format(q['question_id']))
        print('-- Text: {}'.format(q['question']))

        # Prepare object to store info
        model_rk = {
            'question_id': q['question_id']
        }

        # Loading image and conversation
        img_path = os.path.join(args.ds_folder, q['img_path'])
        print('-- Loading image {}'.format(img_path))

        # Format text input
        user_q = prompts['question_template'][MODEL_KEY][args.ds_name]
        if len(user_q) > 0:
            user_q += '\n'
        user_q += q['question']
        if q['options']:
            for k in q['options']:
                user_q += '\n- {}: {}'.format(k, q['options'][k])
        
        conv = get_chat_state()
        conv, img_list = load_img(chat, img_path, conv)
        
        # First generation step: get answer from the model
        conv = ask(chat, user_q, conv)
        model_rk['response'] = get_response(chat, img_list, conv,
                                            num_repeats=1,
                                            num_beams=args.num_beams,
                                            temperature=args.temperature,
                                            max_new_tokens=args.max_new_tokens)
        print('Response: {}'.format(model_rk['response']))
        print('='*25)

        # Second generation step: get unstructured rationales for model output
        expl_q = prompts['rationale_template'][MODEL_KEY]
        conv = ask(chat, expl_q, conv)
        model_rk['rationales'] = get_response(chat, img_list, conv,
                                              num_repeats=1,
                                              num_beams=args.num_beams,
                                              temperature=args.temperature,
                                              max_new_tokens=args.max_new_tokens)
        
        print(model_rk['rationales'])
        print('='*25)
    
        # Third step: triple extraction and structuring from rationales
        struct_expl_q = prompts['out_format_template'][MODEL_KEY]
        conv = ask(chat, struct_expl_q, conv)
        model_rk['triples'] = get_response(chat, img_list, conv,
                                           num_repeats=1,
                                           num_beams=args.num_beams,
                                           temperature=args.temperature,
                                           max_new_tokens=args.max_new_tokens)
        print(model_rk['triples'])
        print('='*25)
        
        all_model_rk.append(model_rk)

        # Free up memory
        del img_list
        torch.cuda.empty_cache()
    
    # Save results to file
    print('... Saving data ...')
    save_attrib_data(all_model_rk, out_dir, args.ds_name)
    print('... Data saved.')
    
if __name__ == "__main__":
    main()