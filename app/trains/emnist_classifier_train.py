import os

import numpy as np
import torch

if __name__ == "__main__":
    from pathlib import Path
    import sys

    dir_app = Path(__file__).resolve().parents[1]
    sys.path.append(str(dir_app))
    print(dir_app)

from data_io import config_io
from data_io import logger_io
from data_io import workspace_io
from data_io import torch_io
import models
from services.lit_models.text_recognizer_builder import Lit_Text_Recognizer_Builder
from services.trainers.pl_trainer_set import Pl_Trainer_Set
from services.trainers.text_recognizer_trainer_set_builder import (
    Text_Recognizer_Trainer_Set_Builder,
)

# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


def main():
    project_name = "emnist_classifier"
    dir_workspace, logger_py = workspace_io.create_workspace_train(project_name)
    args_config = config_io.load_config_args_train(project_name)
    logger_io.write_run_start(logger_py, project_name, dir_workspace, args_config)

    try:
        data = torch_io.import_data(data_name=args_config.data_class, args=args_config)
        model = models.import_model(
            model_name=args_config.model_class,
            config_data=data.config(),
            arg=args_config,
        )
        lit_model = Lit_Text_Recognizer_Builder.build_lit_model(
            project_name=project_name, model=model, args=args_config
        )
        trainer_set = Text_Recognizer_Trainer_Set_Builder.build_trainer_set(
            project_name=project_name,
            dir_workspace=dir_workspace,
            model=model,
            logger_py=logger_py,
            args=args_config,
        )
        assert isinstance(trainer_set, Pl_Trainer_Set)
        trainer_set.tune_fit_test_save(lit_model=lit_model, data=data)
    except Exception as e:
        logger_io.write_err_log(logger_py)


if __name__ == "__main__":
    main()
