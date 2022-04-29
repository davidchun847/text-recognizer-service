from pathlib import Path
from typing import Union

from PIL import Image

if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    print(str(Path(__file__).resolve().parents[2]))

from data_io import pathutils
from data_io import config_io
from data_io import logger_io
from data_io import workspace_io
from services.lit_models.lit_model_builder import Lit_Model_Builder
from services.lit_models.text_recognizer_builder import Lit_Text_Recognizer_Builder

from services.predictors.predictor_builder_base import Predictor_Builder_Base

from services.predicts.pred_service_base import Pred_Service_Base
from services.predictors.pl_predictor import Pl_Predictor


class Text_Recognizer_Pred_Service(Pred_Service_Base):
    def __init__(self, project_name, logger_py, predictor_builder_cls, args_config):
        super().__init__(
            project_name=project_name,
            logger_py=logger_py,
            lit_model_builder_cls=Lit_Text_Recognizer_Builder,
            predictor_builder_cls=predictor_builder_cls,
            args_config=args_config,
        )
        assert isinstance(self.predictor, Pl_Predictor)

    def predict(self, image: Union[str, Path, Image.Image]):
        assert isinstance(self.predictor, Pl_Predictor)
        text_gen = self.predictor.predict(image)
        return text_gen


if __name__ == "__main__":

    from services.predictors.text_recognizer_predictor_builder import (
        Text_Recognizer_Predictor_Builder,
    )
    from services.predictors.text_recognizer_predictor import (
        Text_Recognizer_Predictor,
    )
    from services.predictors.emnist_classifier_predictor_builder import (
        EMNIST_Classifier_Predictor_Builder,
    )
    from services.predictors.emnist_classifier_predictor import (
        EMNIST_Classifier_Predictor,
    )

    project_name = "para_text_recognizer"
    project_name = "emnist_classifier"
    dir_workspace, logger_py = workspace_io.create_workspace_predict(project_name)
    args_config = config_io.load_config_args_predict(project_name)
    logger_io.write_run_start(logger_py, project_name, dir_workspace, args_config)
    if project_name == "para_text_recognizer":
        dir_workspace, logger_py = workspace_io.create_workspace_predict(project_name)
        args_config = config_io.load_config_args_predict(project_name)
        logger_io.write_run_start(logger_py, project_name, dir_workspace, args_config)
        try:
            serivce = Text_Recognizer_Pred_Service(
                project_name,
                logger_py=logger_py,
                predictor_builder_cls=Text_Recognizer_Predictor_Builder,
                args_config=args_config,
            )
            data_dir = pathutils.get_data_proc_dir("iam_paragraphs")
            path_img = data_dir / "trainval" / "a01-077.png"

            text_gen = serivce.predict(path_img)
            print(text_gen)
        except Exception as e:
            logger_io.write_err_log(logger_py)
    elif project_name == "emnist_classifier":
        dir_workspace, logger_py = workspace_io.create_workspace_predict(project_name)
        args_config = config_io.load_config_args_predict(project_name)
        logger_io.write_run_start(logger_py, project_name, dir_workspace, args_config)
        try:
            serivce = Text_Recognizer_Pred_Service(
                project_name,
                logger_py=logger_py,
                predictor_builder_cls=EMNIST_Classifier_Predictor_Builder,
                args_config=args_config,
            )
            data_dir = pathutils.get_data_dl_dir("emnist")
            path_dir = data_dir / "samples"
            path_imgs = path_dir.glob("*.png")
            for path_img in path_imgs:
                text_gen = serivce.predict(path_img)
                print(text_gen)
        except Exception as e:
            logger_io.write_err_log(logger_py)
