import yaml
from pathlib import Path


class RKHandler:
    def __init__(self):
        self.CONFIG_PATH = (
            Path(__file__)
            .parent.joinpath(Path("..", "config", "rk_paths.yaml"))
            .resolve()
        )
        self.RK_PATHS = self.load_rk_config()
        self.BASE_DIR = Path(".", "data", "really_know")
        self.CURR_MODEL = ""
        self.CURR_DS = ""

    def load_rk_config(self) -> dict:
        with open(self.CONFIG_PATH) as fp:
            return yaml.safe_load(fp)

    def get_curr_model(self) -> str:
        return self.CURR_MODEL

    def set_curr_model(self, new_model: str):
        self.CURR_MODEL = new_model

    def get_curr_ds(self) -> str:
        return self.CURR_DS

    def set_curr_ds(self, new_ds: str):
        self.CURR_DS = new_ds

    def get_model_list(self) -> list:
        return list(self.RK_PATHS.keys())

    def get_ds_list(self) -> list:
        return list(self.RK_PATHS[self.CURR_MODEL].keys())

    def build_path(self, target: str) -> Path:
        fn = self.RK_PATHS[self.CURR_MODEL][self.CURR_DS][target]
        if target == "raw":
            new_fn = Path("outcomes", self.CURR_MODEL, self.CURR_DS, fn)
        elif target == "counter":
            new_fn = Path("counterfactual", self.CURR_MODEL, self.CURR_DS, fn)
        elif target in ["parsed", "parsed_pkl", "final"]:
            new_fn = Path(target, self.CURR_MODEL, self.CURR_DS, fn)
        return self.BASE_DIR.joinpath(new_fn)

    def get_rk_raw_path(self) -> Path:
        return self.build_path("raw")

    def get_rk_parsed_path(self) -> Path:
        return self.build_path("parsed")

    def get_parsed_pkl_path(self) -> Path:
        return self.build_path("parsed_pkl")

    def get_rk_final_path(self) -> Path:
        return self.build_path("final")

    def get_rk_counterfactual_path(self) -> Path:
        return self.build_path("counter")
