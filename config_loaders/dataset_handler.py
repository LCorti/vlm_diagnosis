import yaml
from pathlib import Path


class DatasetHandler:
    def __init__(self):
        self.CONFIG_PATH = (
            Path(__file__)
            .parent.joinpath(Path("..", "config", "dataset_paths.yaml"))
            .resolve()
        )
        self.DATASET_PATHS = self.load_ds_config()
        self.BASE_DIR = Path(".", "data", "datasets")
        self.CURR_DS = ""
        self.CURR_CLASS = ""

    def load_ds_config(self) -> dict:
        with open(self.CONFIG_PATH) as fp:
            return yaml.safe_load(fp)

    def get_ds_list(self) -> list:
        return list(self.DATASET_PATHS.keys())

    ## Update from here

    def get_curr_ds(self) -> str:
        return self.CURR_DS

    def set_curr_ds(self, new_ds: str):
        self.CURR_DS = new_ds

    def get_curr_class(self) -> str:
        return self.CURR_CLASS

    def set_curr_class(self, new_class: str):
        self.CURR_CLASS = new_class

    def get_classes(self) -> list:
        return list(self.DATASET_PATHS[self.CURR_DS].keys())

    def build_path(self, target: str) -> Path:
        if target == "imgs":
            p_name = Path(self.CURR_DS, "imgs", self.CURR_CLASS)
        elif target == "imgs_resized":
            p_name = Path(self.CURR_DS, "imgs_resized", self.CURR_CLASS)
        elif target == "imgs_occluded":
            p_name = Path(self.CURR_DS, "imgs_occluded", self.CURR_CLASS)
        elif target in ["questions", "sampled_questions"]:
            p_name = Path(self.CURR_DS, self.CURR_CLASS, self.DATASET_PATHS[target])
        return self.BASE_DIR.joinpath(p_name)

    def get_imgs_path(self) -> Path:
        return self.build_path("imgs")

    def get_imgs_resized_path(self) -> Path:
        return self.build_path("imgs_resized")

    def get_imgs_occluded_path(self) -> Path:
        return self.build_path("imgs_occluded")

    def get_questions_path(self) -> Path:
        return self.build_path("questions")

    def get_sampled_questions_path(self) -> Path:
        return self.build_path("sampled_questions")
