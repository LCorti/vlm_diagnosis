import yaml
from pathlib import Path


class RKConfig:
    def __init__(self):
        self.CONFIG_PATH = Path("../config/rk_paths.yaml").resolve()
        self.RK_PATHS = self.load_rk_config()

    def load_rk_config(self):
        with open(self.CONFIG_PATH) as fp:
            return yaml.safe_load(fp)

    def get_model_list(self):
        return list(self.RK_PATHS.keys())

    def get_ds_list(self, model="internvl2"):
        return list(self.RK_PATHS[model].keys())

    def get_rk_paths(self, model, ds_name):
        if model not in self.RK_PATHS:
            raise
        if ds_name not in self.RK_PATHS[model]:
            raise
        return self.RK_PATHS[model][ds_name]
