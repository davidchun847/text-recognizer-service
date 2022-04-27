"""Base DataModule class."""
import os
from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union
import argparse

from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl

from utils import crypt_utils
from data_io import pathutils, web_request
from data_io.torch_io.dataset_base import DatasetBase

BATCH_SIZE = 128
NUM_WORKERS = 0


class DataModuleBase(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    data_name = None

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # Make sure to set the variables below in subclasses
        self.dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.mapping: Collection
        self.data_train: Union[DatasetBase, ConcatDataset]
        self.data_val: Union[DatasetBase, ConcatDataset]
        self.data_test: Union[DatasetBase, ConcatDataset]

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE,
            help="Number of examples to operate on per forward step.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=NUM_WORKERS,
            help="Number of additional processes to load data.",
        )
        return parser

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {
            "input_dims": self.dims,
            "output_dims": self.output_dims,
            "mapping": self.mapping,
        }

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    @classmethod
    def _get_data_raw_dir(cls):
        assert isinstance(cls.data_name, str)
        return pathutils.get_data_raw_dir(cls.data_name)

    @classmethod
    def _get_data_dl_dir(cls):
        assert isinstance(cls.data_name, str)
        return pathutils.get_data_dl_dir(cls.data_name)

    @classmethod
    def _get_data_proc_dir(cls):
        assert isinstance(cls.data_name, str)
        return pathutils.get_data_proc_dir(cls.data_name)

    @classmethod
    def _get_data_meta_path(cls):
        assert isinstance(cls.data_name, str)
        return pathutils.get_data_meta_path(cls.data_name)

    @classmethod
    def _get_essential_path(cls):
        assert isinstance(cls.data_name, str)
        return pathutils.get_data_essential_path(cls.data_name)

    @classmethod
    def _get_data_extract_dir(cls, dir_sub):
        assert isinstance(cls.data_name, str)
        return pathutils.get_data_extract_dir(cls.data_name, dir_sub)

    @classmethod
    def _get_data_fname_path(cls):
        assert isinstance(cls.data_name, str)
        return pathutils.get_data_fname_path(cls.data_name)

    @classmethod
    def _download_raw_dataset(cls, metadata: Dict, dl_dirname: Path) -> Path:
        dl_dirname.mkdir(parents=True, exist_ok=True)
        filename = dl_dirname / metadata["filename"]
        if os.path.isfile(filename):
            return filename
        print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
        web_request.download_url(metadata["url"], filename)
        print("Computing SHA-256...")
        sha256 = crypt_utils.compute_sha256(filename)
        if sha256 != metadata["sha256"]:
            raise ValueError(
                "Downloaded data file SHA-256 does not match that listed in metadata document."
            )
        return filename