import abc
import argparse

import torch.nn as nn
import pytorch_lightning as pl
from data_io import config_io
from services import lit_models

from services.lit_models.base import BaseLitModel


class Lit_Model_Builder(abc.ABC):
    @classmethod
    def build_lit_model(cls, project_name, model, args):
        assert isinstance(model, nn.Module)
        assert isinstance(args, argparse.Namespace)
        lit_model_class = cls._get_lit_model_class(args)
        assert isinstance(lit_model_class, type)
        if (args.ckpt_load_from).lower() == "none":
            lit_model = lit_model_class(args=args, model=model)
        elif args.ckpt_load_from == "ckpt":
            ckpt_path = config_io.get_best_ckpt_path_from_config(
                project_name, args.ckpt_load_key, args.ckpt_load_is_max
            )
            lit_model = lit_model_class.load_from_checkpoint(
                checkpoint_path=ckpt_path, args=args, model=model
            )
        elif args.ckpt_load_from in ["pt"]:
            pt_path = config_io.get_pt_from_config(project_name)
            lit_model = lit_model_class.load_from_checkpoint(
                checkpoint_path=pt_path, args=args, model=model
            )
        else:
            raise RuntimeError(
                f"lit_model_load: not valid ckpt type: {args.ckpt_load_from}"
            )
        assert isinstance(lit_model, BaseLitModel)
        return lit_model

    @abc.abstractclassmethod
    def _get_lit_model_class(cls, args):
        pass
