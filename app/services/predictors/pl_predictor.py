from logging import Logger
from typing import Sequence, Union, Dict

from torch.jit import ScriptModule

from data_io.torch_io import DataModuleBase


class Pl_Predictor:
    """Class to recognize paragraph text in an image."""

    def __init__(self, project_name):
        self.project_name = project_name
        self._data = None
        self._logger_py = None
        self._model_script = None

    @property
    def project_name(self):
        return self._project_name

    @project_name.setter
    def project_name(self, name):
        assert isinstance(name, str)
        self._project_name = name

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        assert isinstance(data, DataModuleBase)
        self._data = data

    @property
    def logger_py(self):
        return self._logger_py

    @logger_py.setter
    def logger_py(self, logger):
        assert isinstance(logger, Logger)
        self._logger_py = logger

    @property
    def model_script(self):
        return self._model_script

    @model_script.setter
    def model_script(self, model_script):
        assert isinstance(model_script, ScriptModule)
        self._model_script = model_script
