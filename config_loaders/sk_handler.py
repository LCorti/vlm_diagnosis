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

    def get_sk_list(self) -> list:
        return list(self.SK_PATHS.keys())

    def get_classes(self) -> list:
        return list(self.SK_PATHS[self.CURR_DS].keys())

    def build_path(self, target: Path) -> Path:
        return self.BASE_DIR.joinpath(self.CURR_DS, self.CURR_CLASS, target)

    def get_sg_raw_path(self) -> Path:
        target = self.SK_PATHS[self.CURR_DS][self.CURR_CLASS]["sg_raw"]
        target = Path("scene_graphs", target)
        return self.build_path(target)

    def get_sg_clean_path(self) -> Path:
        target = self.SK_PATHS[self.CURR_DS][self.CURR_CLASS]["sg_clean"]
        target = Path("scene_graphs", target)
        return self.build_path(target)

    def get_sg_merged_path(self) -> Path:
        target = self.SK_PATHS[self.CURR_DS][self.CURR_CLASS]["sg_merged"]
        target = Path("scene_graphs", target)
        return self.build_path(target)

    def get_sg_crowd_path(self) -> Path:
        target = self.SK_PATHS[self.CURR_DS][self.CURR_CLASS]["sg_crowd"]
        target = Path("scene_graphs", target)
        return self.build_path(target)

    def get_val_step_path(self) -> Path:
        return self.build_path(self.SK_PATHS[self.CURR_DS][self.CURR_CLASS]["val_step"])

    def get_ann_step_path(self) -> Path:
        return self.build_path(self.SK_PATHS[self.CURR_DS][self.CURR_CLASS]["ann_step"])

    def get_ann_exp_path(self) -> Path:
        return self.build_path(self.SK_PATHS[self.CURR_DS][self.CURR_CLASS]["ann_exp"])

    def get_sk_final_path(self) -> Path:
        return self.build_path(self.SK_PATHS[self.CURR_DS][self.CURR_CLASS]["sk_final"])

    def get_pkl_path(self) -> Path:
        target = self.SK_PATHS[self.CURR_DS][self.CURR_CLASS]["pkl"]
        target = Path("pkl", target)
        return self.build_path(target)
