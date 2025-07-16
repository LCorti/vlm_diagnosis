import yaml
from pathlib import Path


class EvalConfigLoader:
    def __init__(self):
        self.EVAL_CONFIG_PATH = (
            Path(__file__)
            .parent.joinpath(Path("..", "config", "eval_paths.yaml"))
            .resolve()
        )
        self.EVAL_PATHS = self.load_eval_config(self.EVAL_CONFIG_PATH)

    def load_eval_config(self, path: str | Path) -> dict:
        with open(path) as fp:
            return yaml.safe_load(fp)

    def get_model_list(self) -> list:
        return list(self.EVAL_PATHS.keys())

    def get_ds_list(self, model: str = "internvl2") -> list:
        return list(self.EVAL_PATHS[model].keys())

    def get_eval_paths(self, model: str, ds_name: str) -> str:
        return self.EVAL_PATHS[model][ds_name]
