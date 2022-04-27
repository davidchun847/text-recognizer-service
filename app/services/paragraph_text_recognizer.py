from pathlib import Path
from typing import Sequence, Union
import argparse
import json

from PIL import Image
import torch


if __name__ == "__main__":
    from pathlib import Path
    import sys
    import os

    sys.path.append(Path(__file__).resolve().parents[1])
    print(Path(__file__).resolve().parents[1])

from utils import str_utils
from utils import img_utils
from data_io.torch_io import IAMParagraphs
from data_io.torch_io.iam_paragraphs import IMAGE_SCALE_FACTOR, get_transform
from data_io import img_io
from data_io import config_io
from models import ResnetTransformer
from services.lit_models import TransformerLitModel
from services.lit_models.text_recognizer_builder import Lit_Text_Recognizer_Builder


CONFIG_AND_WEIGHTS_DIRNAME = config_io.get_config_dir("para_text_recognizer")


class ParagraphTextRecognizer:
    """Class to recognize paragraph text in an image."""

    def __init__(self):
        self._project_name = "para_text_recognizer"
        self.data = IAMParagraphs()

        # tokens
        inv_mapping = self.data.inverse_mapping
        self.ignore_tokens = get_ignore_tokens(inv_mapping)
        # img transform
        self.transform = get_transform(image_shape=self.data.dims[1:], augment=False)

        args_config = config_io.load_config_args_train(self._project_name)
        model = ResnetTransformer(data_config=self.data.config(), args=args_config)
        self.lit_model = Lit_Text_Recognizer_Builder.build_lit_model(
            project_name=self._project_name, model=model, args=args_config
        )
        self.lit_model.eval()
        self.scripted_model = self.lit_model.to_torchscript(
            method="script", file_path=None
        )

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        """Predict/infer text in input image (which can be a file path)."""
        image_pil = image
        if not isinstance(image, Image.Image):
            image_pil = img_io.read_image_pil(image, grayscale=True)

        image_pil = img_utils.resize_image(image_pil, IMAGE_SCALE_FACTOR)
        image_tensor = self.transform(image_pil)

        y_pred = self.scripted_model(image_tensor.unsqueeze(axis=0))[0]
        pred_str = convert_y_label_to_string(
            y=y_pred, mapping=self.data.mapping, ignore_tokens=self.ignore_tokens
        )

        return pred_str


def convert_y_label_to_string(
    y: torch.Tensor, mapping: Sequence[str], ignore_tokens: Sequence[int]
) -> str:
    return "".join([mapping[i] for i in y if i not in ignore_tokens])


def get_ignore_tokens(inv_mapping: dict) -> list:
    assert isinstance(inv_mapping, dict)
    tokens = str_utils.get_all_special_chars()
    ignore_tokens = [inv_mapping[token] for token in tokens]
    return ignore_tokens


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

    text_recognizer = ParagraphTextRecognizer()
    pred_str = text_recognizer.predict(args.filename)
    print(pred_str)


if __name__ == "__main__":
    main()
