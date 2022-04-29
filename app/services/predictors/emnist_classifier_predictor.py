from pathlib import Path
from typing import Sequence, Union
from PIL import Image
import torch
from utils import img_utils
from models.cnn import IMAGE_SIZE
from data_io.torch_io.iam_paragraphs import IMAGE_SCALE_FACTOR
from data_io import img_io
from services.lit_models.text_recognizer_builder import Lit_Text_Recognizer_Builder
from services.predictors.pl_predictor import Pl_Predictor


class EMNIST_Classifier_Predictor(Pl_Predictor):
    """Class to recognize paragraph text in an image."""

    def __init__(self, project_name):
        super().__init__(project_name)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        """Predict/infer text in input image (which can be a file path)."""
        image_pil = image
        if not isinstance(image, Image.Image):
            image_pil = img_io.read_image_pil(image, grayscale=True)

        image_pil = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
        image_tensor = self.transform(image_pil)

        y_pred = self._model_script(image_tensor.unsqueeze(axis=0))[0]
        pred_str = self.convert_y_pred2str(y=y_pred, mapping=self.data.mapping)

        return pred_str

    @classmethod
    def convert_y_pred2str(cls, y: torch.Tensor, mapping: Sequence[str]) -> str:
        pred = y.argmax(-1)
        return mapping[pred]
