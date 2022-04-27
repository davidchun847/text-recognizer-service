from enum import Enum, auto
from data_io import logger_io
from data_io.pathutils import pathutils_pl
from utils.torch_utils import TorchRunType


def create_workspace_train(project_name):
    return create_workspace(TorchRunType.train, project_name)


def create_workspace_predict(project_name):
    return create_workspace(TorchRunType.predict, project_name)


def create_workspace(workspace_type, project_name):
    assert isinstance(workspace_type, TorchRunType)
    if workspace_type == TorchRunType.train:
        dir_workspace = pathutils_pl.create_train_dir(project_name)
    elif workspace_type == TorchRunType.predict:
        dir_workspace = pathutils_pl.create_predict_dir(project_name)
    else:
        raise RuntimeError(
            f"invalid workspace type: project_name: {project_name}, workspace_type: {workspace_type}"
        )
    logger_base = logger_io.create_logger("base", dir_workspace)
    return dir_workspace, logger_base
