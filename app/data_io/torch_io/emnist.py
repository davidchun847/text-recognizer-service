"""
EMNIST dataset. Downloads from NIST website and saves as .npz file if not already present.
"""
from pathlib import Path
from typing import Sequence
import json
import os
import shutil
import zipfile

from torchvision import transforms
import h5py
import numpy as np
import toml

from data_io.torch_io.data_module_base import DataModuleBase
from data_io.torch_io.dataset_base import DatasetBase
from utils import str_utils
from data_io import torch_io

NUM_SPECIAL_TOKENS = str_utils.get_num_special_chars()
SAMPLE_TO_BALANCE = True  # If true, take at most the mean number of instances per class.
TRAIN_FRAC = 0.8


class EMNIST(DataModuleBase):
    """
    "The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset."
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset

    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
    """

    data_name = "emnist"

    def __init__(self, args=None):
        super().__init__(args)

        path_essen = self._get_essential_path()
        if not os.path.exists(path_essen):
            self._download_and_process_emnist()
        with open(path_essen) as f:
            essentials = json.load(f)
        self.mapping = list(essentials["characters"])
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (1, *essentials["input_shape"])  # Extra dimension is added by ToTensor()
        self.output_dims = (1,)

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self, *args, **kwargs) -> None:
        if not os.path.exists(self._get_data_fname_path()):
            self._download_and_process_emnist()
        with open(self._get_essential_path()) as f:
            _essentials = json.load(f)

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            with h5py.File(self._get_data_fname_path(), "r") as f:
                self.x_trainval = f["x_train"][:]
                self.y_trainval = f["y_train"][:].squeeze().astype(int)

            data_trainval = DatasetBase(self.x_trainval, self.y_trainval, transform=self.transform)
            self.data_train, self.data_val = data_trainval.split_dataset(fraction=TRAIN_FRAC, seed=42)

        if stage == "test" or stage is None:
            with h5py.File(self._get_data_fname_path(), "r") as f:
                self.x_test = f["x_test"][:]
                self.y_test = f["y_test"][:].squeeze().astype(int)
            self.data_test = DatasetBase(self.x_test, self.y_test, transform=self.transform)

    def __repr__(self):
        basic = f"EMNIST Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\nDims: {self.dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data

    @classmethod
    def _download_and_process_emnist(cls):
        metadata = toml.load(cls._get_data_meta_path())
        cls._download_raw_dataset(metadata, cls._get_data_dl_dir())
        cls._process_raw_dataset(metadata["filename"], cls._get_data_dl_dir())

    @classmethod
    def _process_raw_dataset(cls, filename: str, dirname: Path):
        print("Unzipping EMNIST...")
        curdir = os.getcwd()
        os.chdir(dirname)
        zip_file = zipfile.ZipFile(filename, "r")
        zip_file.extract("matlab/emnist-byclass.mat")

        from scipy.io import loadmat  # pylint: disable=import-outside-toplevel

        # NOTE: If importing at the top of module, would need to list scipy as prod dependency.

        print("Loading training data from .mat file")
        data = loadmat("matlab/emnist-byclass.mat")
        x_train = data["dataset"]["train"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
        y_train = data["dataset"]["train"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
        x_test = data["dataset"]["test"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
        y_test = data["dataset"]["test"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
        # NOTE that we add NUM_SPECIAL_TOKENS to targets, since these tokens are the first class indices

        if SAMPLE_TO_BALANCE:
            print("Balancing classes to reduce amount of data")
            x_train, y_train = cls._sample_to_balance(x_train, y_train)
            x_test, y_test = cls._sample_to_balance(x_test, y_test)

        print("Saving to HDF5 in a compressed format...")
        cls._get_data_proc_dir().mkdir(parents=True, exist_ok=True)
        with h5py.File(cls._get_data_fname_path(), "w") as f:
            f.create_dataset("x_train", data=x_train, dtype="u1", compression="lzf")
            f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
            f.create_dataset("x_test", data=x_test, dtype="u1", compression="lzf")
            f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")

        print("Saving essential dataset parameters to text_recognizer/datasets...")
        mapping = {int(k): chr(v) for k, v in data["dataset"]["mapping"][0, 0]}
        characters = cls._augment_emnist_characters(list(mapping.values()))
        essentials = {"characters": characters, "input_shape": list(x_train.shape[1:])}
        with open(cls._get_essential_path(), "w") as f:
            json.dump(essentials, f)

        print("Cleaning up...")
        shutil.rmtree("matlab")
        os.chdir(curdir)

    @classmethod
    def _sample_to_balance(cls, x, y):
        """Because the dataset is not balanced, we take at most the mean number of instances per class."""
        np.random.seed(42)
        num_to_sample = int(np.bincount(y.flatten()).mean())
        all_sampled_inds = []
        for label in np.unique(y.flatten()):
            inds = np.where(y == label)[0]
            sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
            all_sampled_inds.append(sampled_inds)
        ind = np.concatenate(all_sampled_inds)
        x_sampled = x[ind]
        y_sampled = y[ind]
        return x_sampled, y_sampled

    @classmethod
    def _augment_emnist_characters(cls, characters: Sequence[str]) -> Sequence[str]:
        """Augment the mapping with extra symbols."""
        # Extra characters from the IAM dataset
        iam_characters = [
            " ",
            "!",
            '"',
            "#",
            "&",
            "'",
            "(",
            ")",
            "*",
            "+",
            ",",
            "-",
            ".",
            "/",
            ":",
            ";",
            "?",
        ]

        # Also add special tokens:
        # - CTC blank token at index 0
        # - Start token at index 1
        # - End token at index 2
        # - Padding token at index 3
        # NOTE: Don't forget to update NUM_SPECIAL_TOKENS if changing this!
        return [*(str_utils.get_all_special_chars()), *characters, *iam_characters]


if __name__ == "__main__":
    torch_io.load_and_print_info(EMNIST)
