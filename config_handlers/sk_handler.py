import yaml
from pathlib import Path


class SKHandler:
    def __init__(self):
        self.CONFIG_PATH = (
            Path(__file__)
            .parent.joinpath(Path("..", "config", "sk_paths.yaml"))
            .resolve()
        )
        self.SK_PATHS = self.load_sk_config()
        self.BASE_DIR = Path(".", "data", "should_know")
        self.CURR_DS = ""
        self.CURR_CLASS = ""

    def load_sk_config(self) -> dict:
        with open(self.CONFIG_PATH) as fp:
            return yaml.safe_load(fp)

    def get_curr_ds(self) -> str:
        return self.CURR_DS

    def set_curr_ds(self, new_ds: str):
        self.CURR_DS = new_ds

    def get_curr_class(self) -> str:
        return self.CURR_CLASS

    def set_curr_class(self, new_class: str):
        self.CURR_CLASS = new_class

    def get_ds_list(self, return_holdout: bool = False) -> list:
        if not return_holdout:
            ds_list = list(self.SK_PATHS.keys())
            ds_list.remove("vqav2_holdout")
            return ds_list
        return list(self.SK_PATHS.keys())

    def get_classes(self) -> list:
        return list(self.SK_PATHS[self.CURR_DS].keys())

    def build_path(self, target: str) -> Path:
        fn = self.SK_PATHS[self.CURR_DS][self.CURR_CLASS][target]
        if target in ["sg_raw", "sg_clean", "sg_merged", "sg_crowd"]:
            new_fn = Path("scene_graphs", fn)
        elif target == "pkl":
            new_fn = Path("pkl", fn)
        else:
            new_fn = fn
        return self.BASE_DIR.joinpath(self.CURR_DS, self.CURR_CLASS, new_fn)

    def get_sg_raw_path(self) -> Path:
        return self.build_path("sg_raw")

    def get_sg_clean_path(self) -> Path:
        return self.build_path("sg_clean")

    def get_sg_merged_path(self) -> Path:
        return self.build_path("sg_merged")

    def get_sg_crowd_path(self) -> Path:
        return self.build_path("sg_crowd")

    def get_val_step_path(self) -> Path:
        return self.build_path("val_step")

    def get_ann_step_path(self) -> Path:
        return self.build_path("ann_step")

    def get_ann_exp_path(self) -> Path:
        return self.build_path("ann_exp")

    def get_sk_final_path(self) -> Path:
        return self.build_path("sk_final")

    def get_pkl_path(self) -> Path:
        return self.build_path("pkl")
