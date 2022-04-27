from logging import Logger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import LightningLoggerBase

from data_io import logger_io
from data_io import wandb_io
from data_io import config_io
from services.pl_callbacks import CallbackType


class Pl_Trainer_Set:
    def __init__(self, project_name):
        self.project_name = project_name
        self._logger_py = None
        self._loggers_pl = []
        self._callbacks = dict()
        self._trainer = None
        self._b_wandb = True

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

    def get_pl_loggers(self):
        return self._loggers_pl

    def add_pl_logger(self, logger):
        assert isinstance(logger, LightningLoggerBase)
        self._loggers_pl.append(logger)

    def add_callback(self, type_cb, callback):
        assert isinstance(type_cb, CallbackType)
        assert isinstance(callback, Callback)
        self._callbacks[type_cb] = callback

    def get_callback(self, type_cb):
        assert isinstance(type_cb, CallbackType)
        return self._callbacks[type_cb]

    def get_callbacks(self):
        return list(self._callbacks.values())

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        assert isinstance(trainer, pl.Trainer)
        self._trainer = trainer

    @property
    def b_wandb(self):
        return self._b_wandb

    @b_wandb.setter
    def b_wandb(self, b_wandb):
        assert isinstance(b_wandb, bool)
        self._b_wandb = b_wandb

    def tune_fit_test_save(self, lit_model, data):
        assert isinstance(lit_model, pl.LightningModule)
        assert isinstance(data, pl.LightningDataModule)
        assert isinstance(self._trainer, pl.Trainer)
        self._trainer.tune(lit_model, datamodule=data)
        self._trainer.fit(lit_model, datamodule=data)
        self._trainer.test(lit_model, datamodule=data)
        self._save_best_model()

    def _save_best_model(self):
        assert isinstance(self._logger_py, Logger)
        best_model_path = self._get_best_model_path()
        config_io.copy_best_ckpt_2_config(self._project_name, best_model_path)
        if best_model_path:
            logger_io.print_log_info(self._logger_py, f"best_model_path: {best_model_path}")
            if self._b_wandb:
                logger_io.print_log_info(self._logger_py, f"upload 2 wandb")
                wandb_io.save_wandb_model()
        else:
            raise FileNotFoundError("best_model_path not found")

    def _get_best_model_path(self):
        assert CallbackType.ckpt in self._callbacks
        path = self._callbacks[CallbackType.ckpt].best_model_path
        return path