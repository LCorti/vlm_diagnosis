from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)
from llava.conversation import conv_templates


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
