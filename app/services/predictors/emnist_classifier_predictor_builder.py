import torchvision.transforms

from utils import str_utils
from services.predictors.emnist_classifier_predictor import EMNIST_Classifier_Predictor
from services.predictors.predictor_builder_base import Predictor_Builder_Base
from data_io.torch_io import DataModuleBase


class EMNIST_Classifier_Predictor_Builder(Predictor_Builder_Base):
    @classmethod
    def build_predictor(cls, project_name, data, lit_model, logger_py, args_config):
        predictor = super().build_predictor(
            project_name, data, lit_model, logger_py, args_config
        )
        predictor.transform = cls._get_transform()
        assert isinstance(predictor, EMNIST_Classifier_Predictor)
        return predictor

    @classmethod
    def _init_predictor(cls, project_name):
        predictor = EMNIST_Classifier_Predictor(project_name)
        return predictor

    @classmethod
    def _get_transform(cls):
        return torchvision.transforms.ToTensor()