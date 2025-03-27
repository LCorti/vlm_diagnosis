import os
import re
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

        # After manually inspecting the generated really-knows, these characters
        # seem to be used by models (more or less consistently) when asked to
        # generate a list of structured triples.
        self.FIRST_CHARS = ["\\*", "-", "\\+", "[0-9]+\\."]

        # Parsing patterns
        if self.MODEL_NAME == "sharegpt4v":
            self.SEARCH_PATTERN = r"\((.+), (.+), (.+)\)"
            self.GROUP_PATTERN = (
                r"\((?P<from_concept>.+), (?P<relationship>.+), (?P<to_concept>.+)\)"
            )
        else:
            self.SEARCH_PATTERN = r"\(Entity: (.+), Relationship: (.+), Entity: (.+)\)"
            self.GROUP_PATTERN = r"\(Entity: (?P<from_concept>.+), Relationship: (?P<relationship>.+), Entity: (?P<to_concept>.+)\)"

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
        elif self.MODEL_NAME == "sharegpt4v":
            self.GEN_CONFIG = GenerationConfig.from_dict(
                {
                    "do_sample": do_sample,
                    "num_beams": num_beams,
                    "temperature": temperature,
                    "use_cache": use_cache,
                    "max_new_tokens": max_new_tokens,
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

    def gen_response_sharegpt4v(
        self, model, tokenizer, image_tensor, input_ids, stopping_criteria, stop_str
    ):
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                generation_config=self.GEN_CONFIG,
                stopping_criteria=[stopping_criteria],
            )
        # decode_tokens (input_ids, output_ids, tokenizer, stop_str)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )

        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs

    def parse_raw_rk(self, raw_rk):
        # Prep new object
        parsed_rk = {}
        parsed_rk["question_id"] = raw_rk["question_id"]
        parsed_rk["response"] = raw_rk["response"]
        parsed_rk["rationales"] = raw_rk["rationales"]
        parsed_rk["triples"] = []
        parsed_rk["triple_objs"] = []

        # Parse raw-text triples
        # (1) Split lines
        raw_rk_lines = [
            line.rstrip().rstrip()
            for line in raw_rk["triples"].split("\n")
            if len(line) > 0
        ]

        # (2) Parse following this structure
        # <first_char> (Entity: <x>, Relationship: <y>, Entity: <z>)
        pattern = re.compile(self.SEARCH_PATTERN)
        parsed_rk["triples"].extend(rl for rl in raw_rk_lines if re.search(pattern, rl))
        if len(parsed_rk["triples"]) > 0:
            for rk in parsed_rk["triples"]:
                rk_match = re.search(self.GROUP_PATTERN, rk)
                if rk_match is not None:
                    parsed_rk["triple_objs"].append(rk_match.groupdict())
