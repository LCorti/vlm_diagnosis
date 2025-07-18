import yaml
from pathlib import Path


class PromptHandler:
    def __init__(self, prompt_version: int = 1, load_counterfactual: bool = False):
        self.SE_PROMPTS_PATH = (
            Path(__file__)
            .parent.joinpath(
                Path(
                    "..",
                    "config",
                    "prompts",
                    "self_explanations",
                    f"se_v{prompt_version}.yaml",
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

        self.CURR_MODEL = ""
        self.CURR_DS = ""

    # File loading
    def load_prompts(self, path: str | Path) -> dict:
        with open(path) as fp:
            return yaml.safe_load(fp)

    def load_se_prompts(self) -> dict:
        return self.load_prompts(self.SE_PROMPTS_PATH)

    def load_counter_prompts(self) -> dict:
        return self.load_prompts(self.COUNTER_PROMPTS_PATH)

    # Setup
    def get_curr_model(self) -> str:
        return self.CURR_MODEL

    def set_curr_model(self, new_model: str):
        self.CURR_MODEL = new_model

    def get_curr_ds(self) -> str:
        return self.CURR_DS

    def set_curr_ds(self, new_ds: str):
        self.CURR_DS = new_ds

    # Prompt retrieval
    def get_question_template(self) -> str:
        return self.SE_PROMPTS["question_template"][self.CURR_MODEL][self.CURR_DS]

    def get_rationale_template(self) -> str:
        return self.SE_PROMPTS["rationale_template"][self.CURR_MODEL]

    def get_out_format_template(self) -> str:
        return self.SE_PROMPTS["out_format_template"][self.CURR_MODEL]
