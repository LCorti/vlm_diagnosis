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

    def get_ds_list(self, return_holdout: bool = False) -> list:
        if not return_holdout:
            ds_list = list(self.DATASET_PATHS.keys())
            ds_list.remove("vqav2_holdout")
            return ds_list
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
        if target in ["imgs", "imgs_resized"]:
            p_name = Path(self.CURR_DS, target, self.CURR_CLASS)
        elif target in ["questions", "sampled_questions"]:
            p_name = Path(
                self.CURR_DS, self.DATASET_PATHS[self.CURR_DS][self.CURR_CLASS][target]
            )
        elif target == "crowd_questions":
            p_name = Path(self.CURR_DS, "q_crowd.json")
        return self.BASE_DIR.joinpath(p_name)

    def get_imgs_path(self) -> Path:
        return self.build_path("imgs")

    def get_imgs_resized_path(self) -> Path:
        return self.build_path("imgs_resized")

    def get_questions_path(self) -> Path:
        return self.build_path("questions")

    def get_sampled_questions_path(self) -> Path:
        return self.build_path("sampled_questions")

    def get_crowd_questions_path(self) -> Path:
        return self.build_path("crowd_questions")
