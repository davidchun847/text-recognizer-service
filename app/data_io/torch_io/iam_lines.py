"""
IamLinesDataset class.

We will use a processed version of this dataset, without including code that did the processing.
We will look at how to generate processed data from raw IAM data in the IamParagraphsDataset.
"""
import os
from pathlib import Path
from typing import Sequence
import argparse
import json
import random

from PIL import Image, ImageFile, ImageOps
import numpy as np
from torchvision import transforms

import mypath
from utils import torch_utils
from data_io.torch_io.dataset_base import DatasetBase
from data_io.torch_io.data_module_base import DataModuleBase
from data_io.torch_io.emnist import EMNIST
from data_io.torch_io.iam import IAM
from data_io import img_io
from data_io import torch_io

ImageFile.LOAD_TRUNCATED_IMAGES = True

TRAIN_FRAC = 0.8
IMAGE_HEIGHT = 56
IMAGE_WIDTH = 2048  # Rounding up the actual empirical max to a power of 2


class IAMLines(DataModuleBase):
    """
    IAM Handwriting database lines.
    """

    data_name = "iam_lines"

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.augment = self.args.get("augment_data", "true") == "true"
        self.mapping = EMNIST().mapping
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
        self.dims = (
            1,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )  # We assert that this is correct in setup()
        self.output_dims = (89, 1)  # We assert that this is correct in setup()

        self.data_train = None
        self.data_val = None
        self.data_test = None

    @staticmethod
    def add_to_argparse(parser):
        DataModuleBase.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        if os.path.isdir(self._get_data_proc_dir()):
            return

        print("Cropping IAM line regions...")
        iam = IAM()
        iam.prepare_data()
        crops_trainval, labels_trainval = self.line_crops_and_labels(iam, "trainval")
        crops_test, labels_test = self.line_crops_and_labels(iam, "test")

        shapes = np.array([crop.size for crop in crops_trainval + crops_test])
        aspect_ratios = shapes[:, 0] / shapes[:, 1]

        print("Saving images, labels, and statistics...")
        self.save_images_and_labels(
            crops_trainval, labels_trainval, "trainval", self._get_data_proc_dir()
        )
        self.save_images_and_labels(
            crops_test, labels_test, "test", self._get_data_proc_dir()
        )
        with open(self._get_data_proc_dir() / "_max_aspect_ratio.txt", "w") as file:
            file.write(str(aspect_ratios.max()))

    def setup(self, stage: str = None) -> None:
        with open(self._get_data_proc_dir() / "_max_aspect_ratio.txt") as file:
            max_aspect_ratio = float(file.read())
            image_width = int(IMAGE_HEIGHT * max_aspect_ratio)
            assert image_width <= IMAGE_WIDTH

        if stage == "fit" or stage is None:
            x_trainval, labels_trainval = self.load_line_crops_and_labels(
                "trainval", self._get_data_proc_dir()
            )
            assert (
                self.output_dims[0] >= max([len(_) for _ in labels_trainval]) + 2
            )  # Add 2 for start/end tokens.

            y_trainval = torch_utils.convert_strings_to_labels(
                labels_trainval, self.inverse_mapping, length=self.output_dims[0]
            )
            data_trainval = DatasetBase(
                x_trainval,
                y_trainval,
                transform=self.get_transform(IMAGE_WIDTH, self.augment),
            )

            self.data_train, self.data_val = data_trainval.split_dataset(
                fraction=TRAIN_FRAC, seed=42
            )

        # Note that test data does not go through augmentation transforms
        if stage == "test" or stage is None:
            x_test, labels_test = self.load_line_crops_and_labels(
                "test", self._get_data_proc_dir()
            )
            assert self.output_dims[0] >= max([len(_) for _ in labels_test]) + 2

            y_test = torch_utils.convert_strings_to_labels(
                labels_test, self.inverse_mapping, length=self.output_dims[0]
            )
            self.data_test = DatasetBase(
                x_test, y_test, transform=self.get_transform(IMAGE_WIDTH)
            )

        if stage is None:
            self._verify_output_dims(
                labels_trainval=labels_trainval, labels_test=labels_test
            )

    def _verify_output_dims(self, labels_trainval, labels_test):
        max_label_length = (
            max([len(label) for label in labels_trainval + labels_test]) + 2
        )
        output_dims = (max_label_length, 1)
        if output_dims != self.output_dims:
            raise RuntimeError(output_dims, self.output_dims)

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "IAM Lines Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Dims: {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data

    @classmethod
    def line_crops_and_labels(cls, iam: IAM, split: str):
        """Load IAM line labels and regions, and load line image crops."""
        crops = []
        labels = []
        for filename in iam.form_filenames:
            if not iam.split_by_id[filename.stem] == split:
                continue
            image = img_io.read_image_pil(filename)
            image = ImageOps.grayscale(image)
            image = ImageOps.invert(image)
            labels += iam.line_strings_by_id[filename.stem]
            crops += [
                image.crop([region[_] for _ in ["x1", "y1", "x2", "y2"]])
                for region in iam.line_regions_by_id[filename.stem]
            ]
        assert len(crops) == len(labels)
        return crops, labels

    @classmethod
    def save_images_and_labels(
        cls,
        crops: Sequence[Image.Image],
        labels: Sequence[str],
        split: str,
        data_dirname: Path,
    ):
        (data_dirname / split).mkdir(parents=True, exist_ok=True)

        with open(cls._get_label_path(split, data_dirname), "w") as f:
            json.dump(labels, f)
        for ind, crop in enumerate(crops):
            crop.save(data_dirname / split / f"{ind}.png")

    @classmethod
    def load_line_crops_and_labels(cls, split: str, data_dirname: Path):
        """Load line crops and labels for given split from processed directory."""
        with open(cls._get_label_path(split, data_dirname)) as file:
            labels = json.load(file)

        crop_filenames = sorted(
            (data_dirname / split).glob("*.png"),
            key=lambda filename: int(Path(filename).stem),
        )
        crops = [
            img_io.read_image_pil(filename, grayscale=True)
            for filename in crop_filenames
        ]
        assert len(crops) == len(labels)
        return crops, labels

    @classmethod
    def get_transform(cls, image_width, augment=False):
        """Augment with brightness, slight rotation, slant, translation, scale, and Gaussian noise."""

        def embed_crop(crop, augment=augment, image_width=image_width):
            # crop is PIL.image of dtype="L" (so values range from 0 -> 255)
            image = Image.new("L", (image_width, IMAGE_HEIGHT))

            # Resize crop
            crop_width, crop_height = crop.size
            new_crop_height = IMAGE_HEIGHT
            new_crop_width = int(new_crop_height * (crop_width / crop_height))
            if augment:
                # Add random stretching
                new_crop_width = int(new_crop_width * random.uniform(0.9, 1.1))
                new_crop_width = min(new_crop_width, image_width)
            crop_resized = crop.resize(
                (new_crop_width, new_crop_height), resample=Image.BILINEAR
            )

            # Embed in the image
            x = min(28, image_width - new_crop_width)
            y = IMAGE_HEIGHT - new_crop_height
            # if augment:
            #     x = random.randint(0, (image_width - new_crop_width))
            #     y = random.randint(0, (IMAGE_HEIGHT - new_crop_height))
            image.paste(crop_resized, (x, y))

            return image

        transforms_list = [transforms.Lambda(embed_crop)]
        if augment:
            transforms_list += [
                transforms.ColorJitter(brightness=(0.8, 1.6)),
                transforms.RandomAffine(
                    degrees=1, shear=(-30, 20), resample=Image.BILINEAR, fillcolor=0
                ),
            ]
        transforms_list.append(transforms.ToTensor())
        return transforms.Compose(transforms_list)

    @classmethod
    def _get_label_path(cls, split: str, data_dirname: Path):
        return Path(data_dirname, split, "_labels.json")


if __name__ == "__main__":
    torch_io.load_and_print_info(IAMLines)
