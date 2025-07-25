import json
from pathlib import Path
import polars as pl
from pipe.utils.path_utils import find_data_root

DATA_ROOT = find_data_root()


def load_json_to_df(file_path: Path, id_field: str) -> pl.DataFrame:
    with file_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [{id_field: k, **v} for k, v in data.items()]
    return pl.from_dicts(data)


def get_file_path(file_name: str, folder: str = "") -> Path:
    return DATA_ROOT / folder / file_name
