from logging import Logger
from typing import Sequence, Union, Dict

class Text_Recognizer_Predictor:
    """Class to recognize paragraph text in an image."""

    def __init__(self, project_name):
        self.project_name = project_name
        self._data = None
        self._logger_py = None
        self._data = None
        self._transform = None
        self._model_scripted = None

    @property
    def project_name(self):
        return self._project_name

    @project_name.setter
    def project_name(self, name):
        assert isinstance(name, str)
        self._project_name = name

    @property
    def logger_py(self):
        return self._logger_py

    @logger_py.setter
    def logger_py(self, logger):
        assert isinstance(logger, Logger)
        self._logger_py = logger

    @property
    def model_scripted(self):
        return self._model_scripted

    @

        Union[ScriptModule, Dict[str, ScriptModule]