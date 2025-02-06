import yaml
from pathlib import Path


class DatasetConfig:
    def __init__(self):
        self.CONFIG_PATH = Path("../config/files/dataset_paths.yaml").resolve()
        self.DATASET_PATHS = self.load_dataset_config()

    def load_dataset_config(self):
        with open(self.CONFIG_PATH) as fp:
            return yaml.load(fp, Loader=yaml.FullLoader)

    def get_ds_list(self):
        return list(self.DATASET_PATHS.keys())

    def get_ds_paths(self, ds_name):
        if ds_name not in self.DATASET_PATHS:
            raise
        return self.DATASET_PATHS[ds_name]


# if __name__ == "__main__":
#     dataset_config = DatasetConfig()
