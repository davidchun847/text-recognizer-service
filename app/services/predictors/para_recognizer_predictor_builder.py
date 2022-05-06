from utils import str_utils
from services.predictors.para_recognizer_predictor import Para_Recognizer_Predictor
from services.predictors.predictor_builder_base import Predictor_Builder_Base
from data_io.torch_io import DataModuleBase


class Para_Recognizer_Predictor_Builder(Predictor_Builder_Base):
    @classmethod
    def build_predictor(cls, project_name, data, lit_model, logger_py, args_config):
        predictor = super().build_predictor(
            project_name, data, lit_model, logger_py, args_config
        )
        assert isinstance(predictor, Para_Recognizer_Predictor)
        predictor.transform = cls._get_transform(data)
        predictor.tokens_ignore = cls._get_ignore_tokens(data.inverse_mapping)
        return predictor

    @classmethod
    def _init_predictor(cls, project_name):
        predictor = Para_Recognizer_Predictor(project_name)
        return predictor

    @classmethod
    def _get_transform(cls, data):
        assert isinstance(data, DataModuleBase)
        assert hasattr(data, "get_transform")
        assert callable(data.get_transform)
        transform = data.get_transform(image_shape=data.dims[1:], augment=False)
        return transform

    @classmethod
    def _get_ignore_tokens(cls, inv_mapping: dict) -> list:
        assert isinstance(inv_mapping, dict)
        chars_special = str_utils.get_all_special_chars()
        tokens = [inv_mapping[char_sp] for char_sp in chars_special]
        return tokens
