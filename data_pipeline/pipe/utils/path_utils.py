from pathlib import Path


def find_data_root(start: Path = Path.cwd()) -> Path:
    for parent in [start, *start.parents]:
        candidate = parent / "data" / "raw"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Diretório data/raw não encontrado.")
