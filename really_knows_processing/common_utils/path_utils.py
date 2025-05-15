from pathlib import Path


def merge_path(path1: str, path2: str) -> Path:
    path1_elems = path1.split("/")
    path2_elems = path2.split("/")
    merged_path = (
        path1
        + "/"
        + "/".join([elem for elem in path2_elems if elem not in path1_elems])
    )
    return Path(merged_path)
