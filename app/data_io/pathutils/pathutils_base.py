import datetime
from pathlib import Path
import os
from utils import crypt_utils


def check_is_file(path):
    assert isinstance(path, (str, Path))
    if not Path(path).resolve().is_file():
        raise FileNotFoundError(f"{path}")


def check_is_dir(dir):
    assert isinstance(dir, (str, Path))
    if not Path(dir).resolve().is_dir():
        raise FileNotFoundError(f"{dir}")
    elif Path(dir).resolve().is_file():
        raise NotADirectoryError(f"{dir}")


def is_file_not_dir(path):
    assert isinstance(path, (str, Path))
    if Path(path).resolve().is_dir():
        return False
    elif Path(path).resolve().is_file():
        return True
    else:
        raise FileNotFoundError(f"{path}")


def create_dir(dir):
    assert isinstance(dir, (str, Path))
    if Path(dir).resolve().is_dir():
        pass
    elif Path(dir).resolve().is_file():
        raise NotADirectoryError(f"{dir}")
    else:
        os.makedirs(dir, exist_ok=True)


def create_now_name(prefix=None, postfix=None):
    if prefix is not None:
        assert isinstance(prefix, str)
    if postfix is not None:
        assert isinstance(postfix, str)
    curr_time = datetime.datetime.now()
    curr_time_str = curr_time.strftime("%y%m%d_%H%M%S")
    rand_str = crypt_utils.rand_str_generator(6)
    dir_name = ""
    if prefix:
        dir_name += f"{prefix}_"
    dir_name += f"{curr_time_str}"
    if postfix:
        dir_name += f"_{postfix}"
    dir_name += f"_{rand_str}"
    return dir_name