import yaml
from pathlib import Path


class CAConfig:
    def __init__(self):
        self.CE_CONFIG_PATH = (
            Path(__file__)
            .parent.joinpath(Path("..", "config", "ce_paths.yaml"))
            .resolve()
        )
        self.CE_PATHS = self.load_config(self.CE_CONFIG_PATH)

    def load_config(self, path: str | Path) -> dict:
        with open(path) as fp:
            return yaml.safe_load(fp)

    def get_model_list(self) -> list:
        return list(self.CE_PATHS.keys())

    def get_ds_list(self, model: str = "internvl2") -> list:
        return list(self.CE_PATHS[model].keys())

    def get_paths(self, model: str, ds_name: str, key: str = "data") -> str:
        if model not in self.CE_PATHS:
            raise
        if ds_name not in self.CE_PATHS[model]:
            raise
        return self.CE_PATHS[model][ds_name][key]

    def get_ce_paths(self, model: str, ds_name: str) -> str:
        return self.get_paths(model, ds_name, "estimates")

    def get_ca_data_paths(self, model: str, ds_name: str) -> str:
        return self.get_paths(model, ds_name, "ca_data")
