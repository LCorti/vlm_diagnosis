import re
import sys
import torch

from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    GenerationConfig,
)
from typing import Any

module_path = str(Path("..").resolve())
if module_path not in sys.path:
    sys.path.append(module_path)

from config_handlers.prompt_handler import PromptHandler


class GenUtils:
    def __init__(self, model_name: str, prompt_version: int = 1) -> None:
        self.MODEL_NAME = model_name
        self.PROMPT_HDL = PromptHandler(prompt_version=prompt_version)
        self.PROMPT_HDL.set_curr_model(self.MODEL_NAME)
        self.GEN_CONFIG = None

        # Parsing patterns
        # self.SEARCH_PATTERN = r"\(entity: (.+), relationship: (.+), entity: (.+)\)"
        self.SEARCH_PATTERN = r"\((.+), (.+), (.+)\)"
        self.GROUP_PATTERN = r"\(entity: (?P<from_concept>[^,]+), relationship: (?P<relationship>[^,]+), entity: (?P<to_concept>[^)]+)\)"
        self.GROUP_PATTERN_SIMP = r"\((?P<from_concept>[^,]+), (?P<relationship>[^,]+), (?P<to_concept>[^)]+)\)"

    def get_question_template(self, ds_name: str) -> str:
        self.PROMPT_HDL.set_curr_ds(ds_name)
        return self.PROMPT_HDL.get_question_template()

    def get_rationale_template(self) -> str:
        return self.PROMPT_HDL.get_rationale_template()

    def get_out_format_template(self) -> str:
        return self.PROMPT_HDL.get_out_format_template()

    def get_gen_config(
        self,
        do_sample: bool = False,
        num_beams: int = 1,
        temperature: float = 0.1,
        max_new_tokens: int = 256,
        use_cache: bool = True,
        top_k: int = 10,
        top_p: float = 0.95,
    ) -> dict | GenerationConfig:
        # Return if self.GEN_CONFIG already initialised.
        if self.GEN_CONFIG:
            return self.GEN_CONFIG

        # Otherwise, initialise and return it.
        config_dict = dict(
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
        )

        if self.MODEL_NAME == "internvl2":
            self.GEN_CONFIG = config_dict
        elif self.MODEL_NAME == "llava-1.6":
            config_dict["num_beams"] = num_beams
            config_dict["use_cache"] = use_cache
            config_dict["cache_position"] = None
            self.GEN_CONFIG = GenerationConfig.from_dict(config_dict)
        elif self.MODEL_NAME == "sharegpt4v":
            config_dict["num_beams"] = num_beams
            config_dict["use_cache"] = use_cache
            self.GEN_CONFIG = GenerationConfig.from_dict(config_dict)
        return self.GEN_CONFIG

    def gen_response_internvl2(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        image_tensor: torch.Tensor,
        text_input: str,
        history: Any = None,
    ) -> str:
        return model.chat(
            tokenizer,
            image_tensor,
            text_input,
            self.GEN_CONFIG,
            history=history,
            return_history=True,
        )

    def gen_response_llava_next(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        image_tensor: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> str:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=image_tensor, generation_config=self.GEN_CONFIG
            )
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def gen_response_sharegpt4v(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        image_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        stopping_criteria: Any,
        stop_str: str,
    ) -> str:
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

    def clean_rk(self, rk: dict) -> dict:
        return {k: v.strip() for k, v in rk.items()}

    def parse_raw_rk(self, raw_rk: str) -> dict:
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
            line.rstrip().rstrip().lower()
            for line in raw_rk["triples"].split("\n")
            if len(line) > 0
        ]

        # (2) Parse following this structure
        # (Entity: <x>, Relationship: <y>, Entity: <z>)
        parsed_rk["triples"].extend(
            rl for rl in raw_rk_lines if re.search(self.SEARCH_PATTERN, rl)
        )

        if len(parsed_rk["triples"]) > 0:
            for rk in parsed_rk["triples"]:
                # Try with complete patterns first, then with simpler one
                rk_match = re.search(self.GROUP_PATTERN, rk)
                if rk_match is None:
                    rk_match = re.search(self.GROUP_PATTERN_SIMP, rk)

                if rk_match is not None:
                    print(rk_match.groupdict())
                    parsed_rk["triple_objs"].append(self.clean_rk(rk_match.groupdict()))

        # Remove duplicate triples
        parsed_rk["triple_objs"] = [
            dict(t)
            for t in {tuple(sorted(d.items())) for d in parsed_rk["triple_objs"]}
        ]

        return parsed_rk
