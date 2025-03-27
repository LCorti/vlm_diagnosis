from minigpt4.conversation.conversation import CONV_VISION_Vicuna0


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

    return message


def load_image(chat, img_path, chat_state):
    if img_path is None:
        print("Image not found")
        return

    img_list = []
    _ = chat.upload_img(img_path, chat_state, img_list)
    chat.encode_img(img_list)
    return chat_state, img_list


def get_chat_state():
    return CONV_VISION_Vicuna0.copy()


def ask(chat, user_message, chat_state):
    if len(user_message) == 0:
        print("No message received!")
        return
    chat.ask(user_message, chat_state)
    return chat_state


def get_response(
    chat,
    img_list,
    chat_state,
    num_beams=1,
    temperature=0.1,
    max_new_tokens=256,
):
    response = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_length=2000,
    )[0]
    return response
