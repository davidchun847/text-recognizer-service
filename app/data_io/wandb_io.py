from pathlib import Path
import shutil
import wandb


def save_wandb_model(path):
    assert isinstance(path, (str, Path))
    best_model_fname = Path(path).name
    shutil.copy(path, Path(wandb.run.dir, best_model_fname))
    wandb.save(path)
