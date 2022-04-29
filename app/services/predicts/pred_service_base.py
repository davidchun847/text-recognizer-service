from pathlib import Path
import logging

import pytorch_lightning as pl

if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    print(str(Path(__file__).resolve().parents[2]))

from data_io import torch_io
import models

from services.predictors.pl_predictor import Pl_Predictor


class Pred_Service_Base:
    def __init__(
        self,
        project_name,
        lit_model_builder_cls,
        predictor_builder_cls,
        logger_py,
        args_config,
    ):
        assert isinstance(logger_py, logging.Logger)
        assert isinstance(lit_model_builder_cls, type)
        assert hasattr(lit_model_builder_cls, "build_lit_model")
        assert callable(lit_model_builder_cls.build_lit_model)
        assert isinstance(predictor_builder_cls, type)
        assert hasattr(predictor_builder_cls, "build_predictor")
        assert callable(predictor_builder_cls.build_predictor)

        data = torch_io.import_data(data_name=args_config.data_class, args=args_config)
        model = models.import_model(
            model_name=args_config.model_class,
            config_data=data.config(),
            arg=args_config,
        )
        self.lit_model = lit_model_builder_cls.build_lit_model(
            project_name=project_name, model=model, args=args_config
        )
        self.predictor = predictor_builder_cls.build_predictor(
            project_name=project_name,
            data=data,
            lit_model=self.lit_model,
            logger_py=logger_py,
            args_config=args_config,
        )

    @property
    def lit_model(self):
        return self._lit_model

    @lit_model.setter
    def lit_model(self, lit_model):
        assert isinstance(lit_model, pl.LightningModule)
        self._lit_model = lit_model

    @property
    def predictor(self):
        return self._predictor

    @predictor.setter
    def predictor(self, predictor):
        assert isinstance(predictor, Pl_Predictor)
        self._predictor = predictor
