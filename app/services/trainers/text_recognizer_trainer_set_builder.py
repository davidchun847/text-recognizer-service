import argparse

from services import pl_callbacks
from services.pl_callbacks import CallbackType
from services.trainers.pl_trainer_set import Pl_Trainer_Set
from services.trainers.pl_trainer_set_builder import Pl_Trainer_Set_Builder


class Text_Recognizer_Trainer_Set_Builder(Pl_Trainer_Set_Builder):
    @classmethod
    def _set_trainer_set_options(cls, trainer_set):
        super()._set_trainer_set_options(trainer_set)
        assert isinstance(trainer_set, Pl_Trainer_Set)
        trainer_set.b_wandb = True

    @classmethod
    def _add_callbacks(cls, dir_workspace, trainer_set, args):
        super()._add_callbacks(dir_workspace, trainer_set, args)
        assert isinstance(trainer_set, Pl_Trainer_Set)
        assert isinstance(args, argparse.Namespace)
        trainer_set.add_callback(
            CallbackType.early_stop,
            pl_callbacks.create_earlystop_callback(
                mode=args.early_stop_mode,
                monitor=args.early_stop_monitor,
                patience=args.early_stop_patience,
            ),
        )
