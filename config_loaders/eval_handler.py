import yaml
from pathlib import Path


class EvalHandler:
    def __init__(self):
        self.EVAL_CONFIG_PATH = (
            Path(__file__)
            .parent.joinpath(Path("..", "config", "eval_paths.yaml"))
            .resolve()
        )
        self.EVAL_PATHS = self.load_eval_config()
        self.BASE_DIR = Path(".", "data", "eval_performance")
        self.CURR_MODEL = ""
        self.CURR_DS = ""

    def load_eval_config(self) -> dict:
        with open(self.EVAL_CONFIG_PATH) as fp:
            return yaml.safe_load(fp)

    def get_model_list(self) -> list:
        return list(self.EVAL_PATHS.keys())

    def get_ds_list(self, model: str = "internvl2") -> list:
        return list(self.EVAL_PATHS[model].keys())

    def get_eval_path(self) -> Path:
        fn = self.EVAL_PATHS[self.CURR_MODEL][self.CURR_DS]
        return self.BASE_DIR.joinpath(self.CURR_MODEL, self.CURR_DS, fn)
