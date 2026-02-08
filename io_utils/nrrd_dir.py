import nrrd
from pathlib import Path

def read_nrrd_from_dir(dir_path: Path, filename: str):
    file_path = dir_path / filename
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    data, header = nrrd.read(str(file_path))
    return data, header
