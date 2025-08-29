import yaml
from pathlib import Path


class CausalHandler:
    def __init__(self):
        self.CA_CONFIG_PATH = (
            Path(__file__)
            .parent.joinpath(Path("..", "config", "ce_paths.yaml"))
            .resolve()
        )
        self.CA_PATHS = self.load_config(self.CA_CONFIG_PATH)
        self.BASE_DIR = Path(".", "data", "causal_effects")
        self.CURR_MODEL = ""
        self.CURR_DS = ""

    def load_config(self, path: str | Path) -> dict:
        with open(path) as fp:
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
        return list(self.CA_PATHS.keys())

    def get_ds_list(self, model: str = "internvl2") -> list:
        return list(self.CA_PATHS[model].keys())

    def build_path(self, target: str) -> Path:
        if target == "imgs_occluded":
            # In this case, we simply point to the dir with the occluded images.
            # Here, we have (1) a folder for each question idx with all the occluded
            # images inside and (2) files that describe what combination of concepts was
            # masked in them image.
            new_fn = Path(target, self.CURR_MODEL, self.CURR_DS)
        else:
            fn = self.CA_PATHS[self.CURR_MODEL][self.CURR_DS][target]
            if target in ["ca_data", "estimates"]:
                new_fn = Path(target, self.CURR_MODEL, self.CURR_DS, fn)
        return self.BASE_DIR.joinpath(new_fn)

    def get_estimates_path(self) -> Path:
        return self.build_path("estimates")

    def get_ca_data_path(self) -> Path:
        return self.build_path("ca_data")

    def get_imgs_occluded_path(self) -> Path:
        return self.build_path("imgs_occluded")
