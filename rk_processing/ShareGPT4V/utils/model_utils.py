from share4v.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)
from share4v.conversation import SeparatorStyle, conv_templates
from share4v.mm_utils import KeywordsStoppingCriteria


def set_conv_mode(model_name):
    if "llama-2" in model_name.lower():
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


def get_stop_str(conv):
    return conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2


def get_stopping_criteria(input_ids, stop_str, tokenizer):
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    return stopping_criteria


def make_message(model, template, question):
    message = ""
    # When there is a template, add new line
    if len(template) > 0:
        message = f"{template}\n"
    message += question["question"]

    # If multiple-choice question, include the options
    if question["options"]:
        for o in question["options"]:
            message += f"\n- {o}: {question['options'][o]}"

    # Formatting
    if model.config.mm_use_im_start_end:
        formatted_message = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + message
        )
    else:
        formatted_message = DEFAULT_IMAGE_TOKEN + "\n" + message

    return formatted_message
