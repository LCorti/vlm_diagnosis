import os
import sys

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from config_loaders.prompt_loader import PromptLoader


class GenUtils:
    def __init__(self, model_name, prompt_version=1):
        self.PROMPT_VERSION = prompt_version
        self.PROMPT_FILE_PATH = "../prompts_rk_v{}.yaml".format(self.PROMPT_VERSION)
        self.MODEL_NAME = model_name
        self.PROMPT_LOADER = PromptLoader(prompt_version=self.PROMPT_VERSION)
        self.GEN_CONFIG = None

    def get_question_template(self, ds_name):
        return self.PROMPT_LOADER.get_question_template(self.MODEL_NAME, ds_name)

    def get_rationale_template(self):
        return self.PROMPT_LOADER.get_rationale_template(self.MODEL_NAME)

    def get_out_format_template(self):
        return self.PROMPT_LOADER.get_out_format_template(self.MODEL_NAME)

    def make_message(self, template, question):
        message = ""
        # When there is a template, add new line
        if len(template) > 0:
            message = f"{template}\n"
        message += question["question"]

        # If multiple-choice question, include the options
        if question["options"]:
            for o in question["options"]:
                message += "\n- {}: {}".format(o, question["options"][o])

        # Add <image> token to message
        message = f"<image>\n{message}"

        return message

    def get_gen_config(self, do_sample, temperature, max_new_tokens):
        if not self.GEN_CONFIG:
            self.GEN_CONFIG = dict(
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        return self.GEN_CONFIG

    def gen_response_internvl2(
        self, model, text_input, image_tensor, tokenizer, history=None
    ):
        return model.chat(
            tokenizer,
            image_tensor,
            text_input,
            self.GEN_CONFIG,
            history=history,
            return_history=True,
        )
