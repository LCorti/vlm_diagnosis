# == == == == == == == ==
# Model Utils
# == == == == == == == ==


def get_next_role(conv):
    if len(conv) > 0:
        if conv[-1]["role"] == "user":
            role = "assistant"
        else:
            role = "user"
    else:
        role = "user"
    return role


def add_conv_step(conv, user_q, image, prev_resp=None):
    if prev_resp:
        role = get_next_role(conv)
        # If a previous response is passed, add it to the conversation
        # Append the agent's response at previous turn
        conv.append({"role": role, "content": [{"type": "text", "text": prev_resp}]})

    role = get_next_role(conv)
    new_message = {
        "role": role,
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_q},
        ],
    }
    conv.append(new_message)
    return conv


def make_message(template, question):
    message = ""
    # When there is a template, add new line
    if len(template) > 0:
        message = f"{template}\n"
    message += question["question"]

    # If multiple-choice question, include the options
    if question["options"]:
        for o in question["options"]:
            message += f"\n- {o}: {question['options'][o]}"
    return message
