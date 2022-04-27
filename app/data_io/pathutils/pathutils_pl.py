import mypath
from pathlib import Path
import os
from data_io import pathutils


def get_train_dir(project_name):
    assert isinstance(project_name, str)
    dir_project = Path(mypath.dir_workspace, "trains", project_name)
    assert pathutils.check_is_dir(dir_project)
    return dir_project


def create_train_dir(project_name):
    assert isinstance(project_name, str)
    name_sub = pathutils.create_now_name()
    dir_project_now = Path(mypath.dir_workspace, "trains", project_name, name_sub)
    pathutils.create_dir(dir_project_now)
    return dir_project_now


def create_predict_dir(project_name):
    assert isinstance(project_name, str)
    name_sub = pathutils.create_now_name()
    dir_project = Path(mypath.dir_workspace, "predictss", project_name, name_sub)
    pathutils.create_dir(dir_project)
    return dir_project


def create_weight_save_dir(dir_workspace):
    dir_weight = Path(dir_workspace, "weights")
    pathutils.create_dir(dir_weight)
    return dir_weight
