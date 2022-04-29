from enum import Enum, auto

import pytorch_lightning as pl
from data_io import pathutils

from pytorch_lightning.callbacks import Callback


class CallbackType(Enum):
    none = 0
    ckpt = auto()
    early_stop = auto()


def create_earlystop_callback(mode="min", monitor="val_loss", patience=10):
    assert isinstance(mode, str)
    assert isinstance(patience, int)
    callback = pl.callbacks.EarlyStopping(mode=mode, monitor=monitor, patience=patience)
    return callback


def create_model_ckpt_callback(
    dir=None, monitor="val_loss", mode="min", save_weights_only=True
):
    # setting save_weights_only=True to avoid errors if multiple gpus are used
    pathutils.check_is_dir(dir)
    assert isinstance(mode, str)
    filename = "{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}"
    callback = pl.callbacks.ModelCheckpoint(
        dirpath=dir,
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_weights_only=save_weights_only,
    )
    return callback