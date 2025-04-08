import yaml
from pathlib import Path


class CAConfig:
    def __init__(self):
        self.CONFIG_PATH = (
            Path(__file__).parent.joinpath("../config/ca_paths.yaml").resolve()
        )
        self.CA_PATHS = self.load_ca_config()

    def load_ca_config(self):
        with open(self.CONFIG_PATH) as fp:
            return yaml.safe_load(fp)

    def get_model_list(self):
        return list(self.CA_PATHS.keys())

    def get_ds_list(self, model="internvl2"):
        return list(self.CA_PATHS[model].keys())

    def get_ca_paths(self, model, ds_name, ca_type):
        if model not in self.CA_PATHS:
            raise
        if ds_name not in self.CA_PATHS[model]:
            raise
        return self.CA_PATHS[model][ds_name][ca_type]

    def get_ca_estimates_paths(self, model, ds_name):
        return self.get_ca_paths(model, ds_name, "estimates")
