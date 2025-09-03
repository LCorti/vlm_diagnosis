from pathlib import Path


def merge_path(path1: str | Path, path2: str | Path) -> Path:
    path1_elems = str(path1).split("/")
    path2_elems = str(path2).split("/")
    merged_path = (
        str(path1)
        + "/"
        + "/".join([elem for elem in path2_elems if elem not in path1_elems])
    )
    return Path(merged_path)
