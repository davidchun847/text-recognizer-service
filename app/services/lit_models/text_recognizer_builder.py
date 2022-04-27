import argparse

from .base import BaseLitModel
from .ctc import CTCLitModel
from .transformer import TransformerLitModel
from services.lit_models.lit_model_builder import Lit_Model_Builder


class Lit_Text_Recognizer_Builder(Lit_Model_Builder):
    @classmethod
    def _get_lit_model_class(cls, args):
        assert isinstance(args, argparse.Namespace)
        lit_model_class = cls.__get_model_class_by_loss(args.loss)
        return lit_model_class

    @classmethod
    def __get_model_class_by_loss(cls, loss):
        assert isinstance(loss, str)
        if loss not in ("ctc", "transformer"):
            lit_model_class = BaseLitModel
        if loss == "ctc":
            lit_model_class = CTCLitModel
        if loss == "transformer":
            lit_model_class = TransformerLitModel
        return lit_model_class
