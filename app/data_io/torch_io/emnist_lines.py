import os

from typing import Dict, Sequence
from collections import defaultdict
from pathlib import Path
import argparse

from torchvision import transforms
import h5py
import numpy as np
import torch

from utils import np_utils
from utils import str_utils
from data_io import torch_io
from data_io.torch_io.dataset_base import DatasetBase
from data_io.torch_io.data_module_base import DataModuleBase
from data_io.torch_io.emnist import EMNIST

MAX_LENGTH = 32
MIN_OVERLAP = 0
MAX_OVERLAP = 0.33
NUM_TRAIN = 10000
NUM_VAL = 2000
NUM_TEST = 2000


class EMNISTLines(DataModuleBase):
    """EMNIST Lines dataset: synthetic handwriting lines dataset made from EMNIST characters."""

    data_name = "emnist_lines"

    def __init__(
        self,
        args: argparse.Namespace = None,
    ):
        super().__init__(args)

        self.max_length = self.args.get("max_length", MAX_LENGTH)
        self.min_overlap = self.args.get("min_overlap", MIN_OVERLAP)
        self.max_overlap = self.args.get("max_overlap", MAX_OVERLAP)
        self.num_train = self.args.get("num_train", NUM_TRAIN)
        self.num_val = self.args.get("num_val", NUM_VAL)
        self.num_test = self.args.get("num_test", NUM_TEST)
        self.with_start_end_tokens = self.args.get("with_start_end_tokens", False)

        self.emnist = EMNIST()
        self.mapping = self.emnist.mapping
        self.dims = (
            self.emnist.dims[0],
            self.emnist.dims[1],
            self.emnist.dims[2] * self.max_length,
        )
        self.output_dims = (self.max_length, 1)
        self.transform = transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def add_to_argparse(parser):
        DataModuleBase.add_to_argparse(parser)
        parser.add_argument(
            "--max_length",
            type=int,
            default=MAX_LENGTH,
            help="Max line length in characters.",
        )
        parser.add_argument(
            "--min_overlap",
            type=float,
            default=MIN_OVERLAP,
            help="Min overlap between characters in a line, between 0 and 1.",
        )
        parser.add_argument(
            "--max_overlap",
            type=float,
            default=MAX_OVERLAP,
            help="Max overlap between characters in a line, between 0 and 1.",
        )
        parser.add_argument(
            "--with_start_end_tokens", action="store_true", default=False
        )
        return parser

    @property
    def data_filename(self):
        return Path(
            self._get_data_proc_dir,
            f"ml_{self.max_length}_o{self.min_overlap:f}_{self.max_overlap:f}_ntr{self.num_train}_ntv{self.num_val}_nte{self.num_test}_{self.with_start_end_tokens}.h5",
        )

    def prepare_data(self, *args, **kwargs) -> None:
        if os.path.isfile(self.data_filename):
            return
        np.random.seed(42)
        self._generate_data("train")
        self._generate_data("val")
        self._generate_data("test")

    def setup(self, stage: str = None) -> None:
        print("EMNISTLinesDataset loading data from HDF5...")
        if stage == "fit" or stage is None:
            with h5py.File(self.data_filename, "r") as f:
                x_train = f["x_train"][:]
                y_train = f["y_train"][:].astype(int)
                x_val = f["x_val"][:]
                y_val = f["y_val"][:].astype(int)

            self.data_train = DatasetBase(x_train, y_train, transform=self.transform)
            self.data_val = DatasetBase(x_val, y_val, transform=self.transform)

        if stage == "test" or stage is None:
            with h5py.File(self.data_filename, "r") as f:
                x_test = f["x_test"][:]
                y_test = f["y_test"][:].astype(int)
            self.data_test = DatasetBase(x_test, y_test, transform=self.transform)

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "EMNIST Lines Dataset\n"  # pylint: disable=no-member
            f"Min overlap: {self.min_overlap}\n"
            f"Max overlap: {self.max_overlap}\n"
            f"Num classes: {len(self.mapping)}\n"
            f"Dims: {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data

    def _generate_data(self, split: str) -> None:
        print(f"EMNISTLinesDataset generating data for {split}...")

        # pylint: disable=import-outside-toplevel
        from data_io.torch_io.sentence_generator import SentenceGenerator

        sentence_generator = SentenceGenerator(
            self.max_length - 2
        )  # Subtract two because we will add start/end tokens

        emnist = self.emnist
        emnist.prepare_data()
        emnist.setup()

        if split == "train":
            samples_by_char = str_utils.get_samples_by_char(
                emnist.x_trainval, emnist.y_trainval, emnist.mapping
            )
            num = self.num_train
        elif split == "val":
            samples_by_char = str_utils.get_samples_by_char(
                emnist.x_trainval, emnist.y_trainval, emnist.mapping
            )
            num = self.num_val
        else:
            samples_by_char = str_utils.get_samples_by_char(
                emnist.x_test, emnist.y_test, emnist.mapping
            )
            num = self.num_test

        self._get_data_proc_dir().mkdir(parents=True, exist_ok=True)
        with h5py.File(self.data_filename, "a") as f:
            x, y = self.create_dataset_of_images(
                num,
                samples_by_char,
                sentence_generator,
                self.min_overlap,
                self.max_overlap,
                self.dims,
            )
            y = np_utils.convert_strings_to_labels(
                y,
                emnist.inverse_mapping,
                length=self.output_dims[0],
                with_start_end_tokens=self.with_start_end_tokens,
            )
            f.create_dataset(f"x_{split}", data=x, dtype="u1", compression="lzf")
            f.create_dataset(f"y_{split}", data=y, dtype="u1", compression="lzf")

    @classmethod
    def select_letter_samples_for_string(cls, string, samples_by_char):
        zero_image = torch.zeros((28, 28), dtype=torch.uint8)
        sample_image_by_char = {}
        for char in string:
            if char in sample_image_by_char:
                continue
            samples = samples_by_char[char]
            sample = samples[np.random.choice(len(samples))] if samples else zero_image
            sample_image_by_char[char] = sample.reshape(28, 28)
        return [sample_image_by_char[char] for char in string]

    @classmethod
    def construct_image_from_string(
        cls,
        string: str,
        samples_by_char: dict,
        min_overlap: float,
        max_overlap: float,
        width: int,
    ) -> torch.Tensor:
        overlap = np.random.uniform(min_overlap, max_overlap)
        sampled_images = cls.select_letter_samples_for_string(string, samples_by_char)
        H, W = sampled_images[0].shape
        next_overlap_width = W - int(overlap * W)
        concatenated_image = torch.zeros((H, width), dtype=torch.uint8)
        x = 0
        for image in sampled_images:
            concatenated_image[:, x : (x + W)] += image
            x += next_overlap_width
        return torch.minimum(torch.Tensor([255]), concatenated_image)

    @classmethod
    def create_dataset_of_images(
        cls, N, samples_by_char, sentence_generator, min_overlap, max_overlap, dims
    ):
        images = torch.zeros((N, dims[1], dims[2]))
        labels = []
        for n in range(N):
            label = sentence_generator.generate()
            images[n] = cls.construct_image_from_string(
                label, samples_by_char, min_overlap, max_overlap, dims[-1]
            )
            labels.append(label)
        return images, labels

    @classmethod
    def convert_strings_to_labels(
        cls,
        strings: Sequence[str],
        mapping: Dict[str, int],
        length: int,
        with_start_end_tokens: bool,
    ) -> np.ndarray:
        """
        Convert sequence of N strings to a (N, length) ndarray, with each string wrapped with <S> and <E> tokens,
        and padded with the <P> token.
        """
        str_pad = str_utils.get_special_char(str_utils.ESpecialChar.pad)
        str_start = str_utils.get_special_char(str_utils.ESpecialChar.start)
        str_end = str_utils.get_special_char(str_utils.ESpecialChar.end)
        labels = np.ones((len(strings), length), dtype=np.uint8) * mapping[str_pad]
        for i, string in enumerate(strings):
            tokens = list(string)
            if with_start_end_tokens:
                tokens = [str_start, *tokens, str_end]
            for ii, token in enumerate(tokens):
                labels[i, ii] = mapping[token]
        return labels


if __name__ == "__main__":
    torch_io.load_and_print_info(EMNISTLines)
