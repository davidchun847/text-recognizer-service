import argparse
import glob
import json
import argparse
import re
import shutil
from pathlib import Path

import mypath
from data_io import pathutils


def get_config_dir(project_name):
    assert isinstance(project_name, str)
    dir_project = Path(mypath.dir_config, project_name)
    pathutils.check_is_dir(dir_project)
    return dir_project


def load_config_args_train(project_name):
    assert isinstance(project_name, str)
    file_config = Path(get_config_dir(project_name), "config_train.json")
    pathutils.check_is_file(file_config)
    args_o = read_args_from_json(file_config)
    return args_o


def load_config_args_predict(project_name):
    assert isinstance(project_name, str)
    file_config = Path(get_config_dir(project_name), "config_pred.json")
    pathutils.check_is_file(file_config)
    args_o = read_args_from_json(file_config)
    return args_o


def read_args_from_json(path):
    pathutils.check_is_file(path)
    with open(path, "r") as file:
        config = json.load(file)
    args_o = argparse.Namespace(**config)
    return args_o


def get_ckpt_dir(project_name):
    assert isinstance(project_name, str)
    dir_ckpt = Path(get_config_dir(project_name), "ckpts")
    return dir_ckpt


def copy_best_ckpt_2_config(project_name, ckpt_path):
    assert isinstance(project_name, str)
    pathutils.check_is_file(ckpt_path)
    dir_ckpt = get_ckpt_dir(project_name)
    shutil.copy(ckpt_path, dir_ckpt)


def get_pt_from_config(project_name):
    dir_config = get_config_dir(project_name)
    fnames_pt = [Path(x).name for x in dir_config.glob("*.pt")]
    if fnames_pt:
        path_pt = Path(get_ckpt_dir(), fnames_pt[0])
        pathutils.check_is_file(path_pt)
        return path_pt
    else:
        raise FileNotFoundError(f"best ckpt not found: project_name: {project_name}")


def get_best_ckpt_path_from_config(project_name, key_best, b_max=False):
    fname_ckpt = _get_best_cktp_name(project_name, key_best, b_max)
    if fname_ckpt:
        path_ckpt = Path(get_ckpt_dir(project_name), fname_ckpt)
        pathutils.check_is_file(path_ckpt)
        return path_ckpt
    else:
        raise FileNotFoundError(
            f"best ckpt not found: project_name: {project_name}, key_best: {key_best}, b_max: {b_max}"
        )


def _get_best_cktp_name(project_name, key_best, b_max):
    assert isinstance(project_name, str)
    assert isinstance(key_best, str)
    assert isinstance(b_max, bool)
    str_re = f"{key_best}[=]([0-9]*[.][0-9]+)"
    reg = re.compile(str_re)

    fnames_ckpt = _get_all_ckpt_names(project_name)

    if b_max:
        val_best = float("-inf")
    else:
        val_best = float("inf")
    idx_best = None
    for idx_curr, fname in enumerate(fnames_ckpt):
        m = reg.search(fname)
        if m:
            val_curr = float(m.group(1))
            if not b_max and val_curr < val_best:
                val_best = min(val_best, val_curr)
                idx_best = idx_curr
            elif b_max and val_curr > val_best:
                val_best = max(val_best, val_curr)
                idx_best = idx_curr

    if idx_best is None:
        return None
    else:
        return fnames_ckpt[idx_best]


def _get_all_ckpt_names(project_name):
    assert isinstance(project_name, str)
    dir_ckpt = Path(mypath.dir_config, project_name, "ckpts")
    pathutils.check_is_dir(dir_ckpt)
    fnames_ckpt = [Path(x).name for x in dir_ckpt.glob("*.ckpt")]
    return fnames_ckpt