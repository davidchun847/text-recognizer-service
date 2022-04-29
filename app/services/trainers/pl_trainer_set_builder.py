import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn

from data_io import pathutils
from data_io import logger_io
from services.pl_callbacks import CallbackType
from services.trainers.pl_trainer_set import Pl_Trainer_Set
from services import pl_callbacks


class Pl_Trainer_Set_Builder:
    @classmethod
    def build_trainer_set(cls, project_name, dir_workspace, model, logger_py, args):
        assert isinstance(project_name, str)
        assert isinstance(dir_workspace, (str, Path))
        assert isinstance(model, nn.Module)
        assert isinstance(args, argparse.Namespace)
        trainer_set = Pl_Trainer_Set(project_name)
        trainer_set.logger_py = logger_py
        cls._create_loggers_pl(dir_workspace, model, trainer_set, args)
        cls._add_callbacks(dir_workspace, trainer_set, args)
        cls._create_pl_trainer(trainer_set, args)
        return trainer_set

    @classmethod
    def _set_trainer_set_options(cls, trainer_set):
        assert isinstance(trainer_set, Pl_Trainer_Set)

    @classmethod
    def _create_loggers_pl(cls, dir_workspace, model, trainer_set, args):
        assert isinstance(dir_workspace, (str, Path))
        assert isinstance(model, nn.Module)
        assert isinstance(args, argparse.Namespace)
        assert isinstance(trainer_set, Pl_Trainer_Set)
        dir_tb = dir_workspace / "tensorboard"
        trainer_set.add_pl_logger(logger_io.create_pl_tb_logger(dir=dir_tb))
        if args.wandb:
            trainer_set.add_pl_logger(
                logger_io.create_pl_wandb_logger(
                    model, offline=args.wandb_offline, params_loghyper=vars(args)
                )
            )

    @classmethod
    def _add_callbacks(cls, dir_workspace, trainer_set, args):
        assert isinstance(dir_workspace, (str, Path))
        assert isinstance(trainer_set, Pl_Trainer_Set)
        assert isinstance(args, argparse.Namespace)
        dir_weight = pathutils.create_weight_save_dir(dir_workspace)
        trainer_set.add_callback(
            CallbackType.ckpt,
            pl_callbacks.create_model_ckpt_callback(
                dir=dir_weight, monitor=args.ckpt_monitor, mode=args.ckpt_monitor_mode
            ),
        )

    @classmethod
    def _create_pl_trainer(cls, trainer_set, args):
        assert isinstance(trainer_set, Pl_Trainer_Set)
        assert isinstance(args, argparse.Namespace)
        trainer_set.trainer = pl.Trainer.from_argparse_args(
            args=args,
            callbacks=trainer_set.get_callbacks(),
            logger=trainer_set.get_pl_loggers(),
        )
