import mypath
from pathlib import Path
from data_io import pathutils


def get_data_raw_dir(data_name):
    assert isinstance(data_name, str)
    dir_raw = Path(mypath.dir_data_raw, data_name)
    pathutils.check_is_dir(dir_raw)
    return dir_raw


def get_data_dl_dir(data_name):
    assert isinstance(data_name, str)
    return Path(mypath.dir_data_dl, data_name)


def get_data_proc_dir(data_name):
    assert isinstance(data_name, str)
    return Path(mypath.dir_data_proc, data_name)


def get_data_meta_path(data_name, fname="metadata.toml"):
    assert isinstance(data_name, str)
    assert isinstance(fname, (str, Path))
    path_meta = Path(get_data_raw_dir(data_name), fname)
    pathutils.check_is_file(path_meta)
    return path_meta


def get_data_essential_path(data_name, fname="essentials.json"):
    assert isinstance(data_name, str)
    assert isinstance(fname, (str, Path))
    path_ess = Path(get_data_raw_dir(data_name), fname)
    return path_ess


def get_data_extract_dir(data_name, name_subdir="db"):
    assert isinstance(data_name, str)
    dir_extract = Path(get_data_dl_dir(data_name), name_subdir)
    return dir_extract


def get_data_fname_path(data_name, fname="byclass.h5"):
    assert isinstance(data_name, str)
    assert isinstance(fname, (str, Path))
    return Path(get_data_proc_dir(data_name), fname)
