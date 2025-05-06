import yaml
from pathlib import Path


class SKConfig:
    def __init__(self):
        self.CONFIG_PATH = Path("..", "config", "sk_paths.yaml")
        self.CONFIG_PATH = Path(__file__).parent.joinpath(self.CONFIG_PATH).resolve()
        self.SK_PATHS = self.load_sk_config()

    def load_sk_config(self):
        with open(self.CONFIG_PATH) as fp:
            return yaml.safe_load(fp)

    def get_sk_list(self):
        return list(self.SK_PATHS.keys())

    def get_sk_paths(self, ds_name):
        if ds_name not in self.SK_PATHS:
            raise
        return self.SK_PATHS[ds_name]
