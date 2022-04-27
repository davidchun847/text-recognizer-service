from logging import Logger
from pathlib import Path
from typing import Sequence, Union
import argparse
import json

from PIL import Image
import torch


if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.append(Path(__file__).resolve().parents[1])
    print(Path(__file__).resolve().parents[1])

from utils import str_utils
from utils import img_utils
from data_io.torch_io import IAMParagraphs
from data_io.torch_io.iam_paragraphs import IMAGE_SCALE_FACTOR
from data_io import pathutils
from data_io import img_io
from data_io import config_io
from data_io import config_io
from data_io import logger_io
from data_io import workspace_io
from data_io import torch_io
import models
from services.lit_models.text_recognizer_builder import Lit_Text_Recognizer_Builder


CONFIG_AND_WEIGHTS_DIRNAME = (
    Path(__file__).resolve().parent / "artifacts" / "paragraph_text_recognizer"
)


class Text_Recognizer_Predictor:
    """Class to recognize paragraph text in an image."""

    def __init__(self, project_name):
        self.project_name = project_name
        self._data = None
        self._logger_py = None
        self._data = None
        self._transform = None
        self._model_scripted = None

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

        project_name = "para_text_recognizer"
        dir_workspace, logger_py = workspace_io.create_workspace(
            workspace_io.WorkspaceType.predict, project_name
        )
        args_config = config_io.load_config_args_predict(project_name)
        logger_io.write_run_start(logger_py, project_name, dir_workspace, args_config)

        try:
            data_class = torch_io.import_data_class(args_config.data_class)
            model_class = models.import_model_class(args_config.model_class)
            data = data_class(args_config)
            model = model_class(data_config=data.config(), args=args_config)

            inv_mapping = self.data.inverse_mapping
            self.ignore_tokens = self.get_ignore_tokens(inv_mapping)
            self.transform = IAMParagraphs.get_transform(
                image_shape=self.data.dims[1:], augment=False
            )

            self.lit_model = Lit_Text_Recognizer_Builder.build_lit_model(
                project_name=project_name, model=model, args=args_config
            )
            self._model_scripted = self.lit_model.to_torchscript(
                method="script", file_path=None
            )
            self.lit_model.eval()
        except Exception as e:
            logger_io.write_err_log(logger_py)

    @property
    def project_name(self):
        return self._project_name

    @project_name.setter
    def project_name(self, name):
        assert isinstance(name, str)
        self._project_name = name

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        """Predict/infer text in input image (which can be a file path)."""
        image_pil = image
        if not isinstance(image, Image.Image):
            image_pil = img_io.read_image_pil(image, grayscale=True)

        image_pil = img_utils.resize_image(image_pil, IMAGE_SCALE_FACTOR)
        image_tensor = self.transform(image_pil)

        y_pred = self._model_scripted(image_tensor.unsqueeze(axis=0))[0]
        pred_str = self.convert_y_label_to_string(
            y=y_pred, mapping=self.data.mapping, ignore_tokens=self.ignore_tokens
        )

        return pred_str

    @classmethod
    def convert_y_label_to_string(
        cls, y: torch.Tensor, mapping: Sequence[str], ignore_tokens: Sequence[int]
    ) -> str:
        return "".join([mapping[i] for i in y if i not in ignore_tokens])

    @classmethod
    def get_ignore_tokens(cls, inv_mapping: dict) -> list:
        assert isinstance(inv_mapping, dict)
        chars_special = str_utils.get_all_special_chars()
        tokens = [inv_mapping[char_sp] for char_sp in chars_special]
        return tokens


def main():
    """
    Example runs:
    ```
    python text_recognizer/paragraph_text_recognizer.py text_recognizer/tests/support/paragraphs/a01-077.png
    python text_recognizer/paragraph_text_recognizer.py https://fsdl-public-assets.s3-us-west-2.amazonaws.com/paragraphs/a01-077.png
    """
    parser = argparse.ArgumentParser(
        description="Recognize handwritten text in an image file."
    )
    parser.add_argument(
        "--filename", type=str, default="tests/support/paragraphs/a01-077.png"
    )
    args = parser.parse_args()

    name_data = pathutils.get_data_proc_dir("iam_paragraphs")
    path_img = name_data / "a01-077.png"

    text_recognizer = ParagraphTextRecognizer()
    pred_str = text_recognizer.predict(args.filename)
    print(pred_str)


if __name__ == "__main__":
    main()
