"""Utility functions for text_recognizer module."""
from io import BytesIO
from pathlib import Path
from typing import Union
import base64

from PIL import Image
import numpy as np
import smart_open


def read_image_pil(image_uri: Union[Path, str], grayscale=False) -> Image:
    with smart_open.open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file, grayscale)


def read_image_pil_file(image_file, grayscale=False) -> Image:
    with Image.open(image_file) as image:
        if grayscale:
            image = image.convert(mode="L")
        else:
            image = image.convert(mode=image.mode)
        return image


def encode_img_2_b64(img_path):
    b64_string = None
    with open(img_path, "rb") as file:
        b64_string = base64.b64encode(file.read)
    return b64_string


def decode_b64_2_img(b64_string, grayscale=False):  # pylint: disable=unused-argument
    """Load base64-encoded images."""
    try:
        _, b64_data = b64_string.split(",")  # pylint: disable=unused-variable
        image_file = BytesIO(base64.b64decode(b64_data))
        return read_image_pil_file(image_file, grayscale)
    except Exception as exception:
        raise ValueError("Could not load image from b64 {}: {}".format(b64_string, exception)) from exception
