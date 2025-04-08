import yaml
from pathlib import Path


class CAConfig:
    def __init__(self):
        self.CE_CONFIG_PATH = (
            Path(__file__).parent.joinpath("../config/ce_paths.yaml").resolve()
        )
        # self.CA_DATA_CONFIG_PATH = (
        #     Path(__file__).parent.joinpath("../config/ca_data_paths.yaml").resolve()
        # )
        self.CE_PATHS = self.load_config(self.CE_CONFIG_PATH)
        # self.CA_DATA_PATHS = self.load_config(self.CA_DATA_CONFIG_PATH)

    def load_config(self, path):
        with open(path) as fp:
            return yaml.safe_load(fp)

    def get_model_list(self):
        return list(self.CE_PATHS.keys())

    def get_ds_list(self, model="internvl2"):
        return list(self.CE_PATHS[model].keys())

    def get_paths(self, model, ds_name, key="data"):
        if model not in self.CE_PATHS:
            raise
        if ds_name not in self.CE_PATHS[model]:
            raise
        return self.CE_PATHS[model][ds_name][key]

    def get_ce_paths(self, model, ds_name):
        return self.get_paths(model, ds_name, "estimates")
    
    def get_ca_data_paths(self, model, ds_name):
        return self.get_paths(model, ds_name, "ca_data")