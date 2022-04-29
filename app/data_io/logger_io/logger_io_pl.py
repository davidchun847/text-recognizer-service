import pytorch_lightning as pl
import wandb


def create_pl_tb_logger(dir):
    logger = pl.loggers.TensorBoardLogger(dir)
    return logger


def create_pl_wandb_logger(model, offline=True, params_loghyper=None):
    logger = pl.loggers.WandbLogger(offline=offline)
    logger.watch(model)
    if params_loghyper:
        logger.log_hyperparams(params_loghyper)
    return logger
