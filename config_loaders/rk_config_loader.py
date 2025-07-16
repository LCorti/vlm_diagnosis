import yaml
from pathlib import Path


class RKConfig:
    def __init__(self):
        self.CONFIG_PATH = (
            Path(__file__)
            .parent.joinpath(Path("..", "config", "rk_paths.yaml"))
            .resolve()
        )
        self.RK_PATHS = self.load_rk_config()

    def load_rk_config(self) -> dict:
        with open(self.CONFIG_PATH) as fp:
            return yaml.safe_load(fp)

    def get_model_list(self) -> list:
        return list(self.RK_PATHS.keys())

    def get_ds_list(self, model: str = "internvl2") -> list:
        return list(self.RK_PATHS[model].keys())

    def get_rk_paths(self, model: str, ds_name: str, rk_type: str) -> str:
        return self.RK_PATHS[model][ds_name][rk_type]

    def get_raw_rk_paths(self, model: str, ds_name: str) -> str:
        return self.get_rk_paths(model, ds_name, "raw")

    def get_parsed_rk_paths(self, model: str, ds_name: str) -> str:
        return self.get_rk_paths(model, ds_name, "parsed")

    def get_parsed_pkl_paths(self, model: str, ds_name: str) -> str:
        return self.get_rk_paths(model, ds_name, "parsed_pkl")

    def get_final_rk_paths(self, model: str, ds_name: str) -> str:
        return self.get_rk_paths(model, ds_name, "final")

    def get_counterfactual_rk_paths(self, model: str, ds_name: str) -> str:
        return self.get_rk_paths(model, ds_name, "counter")
