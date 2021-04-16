import tarfile
from pathlib import Path
import os
import shutil


def unpack_model(type_name:str, path:Path, n_models=1) -> list:
    if not path.is_file():
        return path

    unpack_folder = Path('unpacked_models')
    unpack_folder /= type_name
    try:
        shutil.rmtree(str(unpack_folder))
    except Exception as e:
        pass
    unpack_folder.mkdir(parents=True, exist_ok=True)

    try:
        tar = tarfile.open(str(path))
        tar.extractall(unpack_folder)
        tar.close()
    except Exception as e:
        print("Untar failed.")
        raise e

    # Pick largest files and assume they
    unpack_folder /= 'S3CONTENTS/'
    unpacked_items = list(unpack_folder.glob('*/*'))
    unpacked_items.sort(reverse=True, key=os.path.getsize)
    new_model_path = ''
    if unpacked_items:
        new_model_paths = unpacked_items[:n_models]
    new_model_paths = [Path(path) for path in new_model_paths]

    for p in new_model_paths:
        print(os.path.getsize(p)/1e6)
    return new_model_paths