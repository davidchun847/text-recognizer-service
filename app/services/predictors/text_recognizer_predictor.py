from pathlib import Path
from typing import Sequence, Union
from PIL import Image
import torch
from utils import img_utils
from data_io.torch_io.iam_paragraphs import IMAGE_SCALE_FACTOR
from data_io import img_io
from services.lit_models.text_recognizer_builder import Lit_Text_Recognizer_Builder
from services.predictors.pl_predictor import Pl_Predictor


class Text_Recognizer_Predictor(Pl_Predictor):
    """Class to recognize paragraph text in an image."""

    def __init__(self, project_name):
        super().__init__(project_name)
        self._transform = None
        self._tokens_ignore = None

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform

    @property
    def tokens_ignore(self):
        return self._tokens_ignore

    @tokens_ignore.setter
    def tokens_ignore(self, tokens):
        if tokens is not None:
            assert isinstance(tokens, list)
            assert len(tokens) > 0
            assert isinstance(tokens[0], int)
        self._tokens_ignore = tokens

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        """Predict/infer text in input image (which can be a file path)."""
        image_pil = image
        if not isinstance(image, Image.Image):
            image_pil = img_io.read_image_pil(image, grayscale=True)

        image_pil = img_utils.resize_image(image_pil, IMAGE_SCALE_FACTOR)
        image_tensor = self.transform(image_pil)

        y_pred = self._model_script(image_tensor.unsqueeze(axis=0))[0]
        pred_str = self.convert_y_label_to_string(
            y=y_pred, mapping=self.data.mapping, tokens_ignore=self.tokens_ignore
        )

        return pred_str

    @classmethod
    def convert_y_label_to_string(
        cls, y: torch.Tensor, mapping: Sequence[str], tokens_ignore: Sequence[int]
    ) -> str:
        return "".join([mapping[i] for i in y if i not in tokens_ignore])
