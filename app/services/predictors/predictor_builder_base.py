import abc

from torch.jit import ScriptModule
import pytorch_lightning as pl

from services.predictors.pl_predictor import Pl_Predictor


class Predictor_Builder_Base:
    @classmethod
    def build_predictor(cls, project_name, data, lit_model, logger_py, args_config):
        assert isinstance(project_name, str)
        predictor = cls._init_predictor(project_name)
        assert isinstance(predictor, Pl_Predictor)
        predictor.logger_py = logger_py
        predictor.data = data
        predictor.model_script = cls._cvt_lit_model_2_script(lit_model)
        return predictor

    @abc.abstractclassmethod
    def _init_predictor(cls, project_name):
        pass

    @classmethod
    def _cvt_lit_model_2_script(cls, lit_model):
        if isinstance(lit_model, pl.LightningModule):
            lit_model.eval()
            model_script = lit_model.to_torchscript(method="script", file_path=None)
        else:
            assert isinstance(lit_model, ScriptModule)
            model_script = lit_model
        return model_script
