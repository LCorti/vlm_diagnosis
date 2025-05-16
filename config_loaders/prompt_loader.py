import yaml
from pathlib import Path


class PromptLoader:
    def __init__(self, prompt_version=1, load_counterfactual=False):
        self.SE_PROMPTS_PATH = (
            Path(__file__)
            .parent.joinpath(
                Path(
                    "..",
                    "config",
                    "prompts",
                    "self_explanations",
                    f"self_expl_v{prompt_version}.yaml",
                )
            )
            .resolve()
        )
        self.SE_PROMPTS = self.load_se_prompts()

        # [OLD] Load prompts for counterfactual responses.
        if load_counterfactual:
            self.COUNTER_PROMPTS_PATH = Path(
                "..", "config", "prompts", "counterfactual", "counter_resp.yaml"
            ).resolve()
            self.COUNTER_PROMPTS = self.load_counter_prompts()

    # File loading
    def load_prompts(self, path):
        with open(path) as fp:
            return yaml.safe_load(fp)

    def load_se_prompts(self):
        return self.load_prompts(self.SE_PROMPTS_PATH)

    def load_counter_prompts(self):
        return self.load_prompts(self.COUNTER_PROMPTS_PATH)

    # Prompt retrieval
    def get_question_template(self, model, ds_name):
        return self.SE_PROMPTS["question_template"][model][ds_name]

    def get_rationale_template(self, model):
        return self.SE_PROMPTS["rationale_template"][model]

    def get_out_format_template(self, model):
        return self.SE_PROMPTS["out_format_template"][model]
