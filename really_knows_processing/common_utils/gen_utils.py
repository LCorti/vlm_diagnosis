import os
import sys
import torch

from transformers import GenerationConfig

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

    def get_gen_config(
        self,
        do_sample=False,
        num_beams=1,
        temperature=0.1,
        max_new_tokens=256,
        use_cache=True,
    ):
        # Return if self.GEN_CONFIG already initialised.
        if self.GEN_CONFIG:
            return self.GEN_CONFIG

        # Otherwise, initialise and return it.
        if self.MODEL_NAME == "internvl2":
            self.GEN_CONFIG = dict(
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        elif self.MODEL_NAME == "llava-1.6":
            self.GEN_CONFIG = GenerationConfig.from_dict(
                {
                    "do_sample": do_sample,
                    "num_beams": num_beams,
                    "temperature": temperature,
                    "use_cache": use_cache,
                    "max_new_tokens": max_new_tokens,
                    "cache_position": None,
                }
            )
        return self.GEN_CONFIG

    def gen_response_internvl2(
        self, model, tokenizer, image_tensor, text_input, history=None
    ):
        return model.chat(
            tokenizer,
            image_tensor,
            text_input,
            self.GEN_CONFIG,
            history=history,
            return_history=True,
        )

    def gen_response_llava_next(self, model, tokenizer, image_tensor, input_ids):
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=image_tensor, generation_config=self.GEN_CONFIG
            )
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
