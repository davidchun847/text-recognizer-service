"""MNIST DataModule"""
import argparse

from torch.utils.data import random_split
from torchvision.datasets import MNIST as TorchMNIST
from torchvision import transforms

import mypath
from data_io.torch_io.data_module_base import DataModuleBase
from data_io import torch_io

# NOTE: temp fix until https://github.com/pytorch/vision/issues/1938 is resolved
from six.moves import urllib  # pylint: disable=wrong-import-position, wrong-import-order

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


class MNIST(DataModuleBase):
    """
    MNIST DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_dir = mypath.dir_data_dl
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dims = (1, 28, 28)  # dims are returned when calling `.size()` on this object.
        self.output_dims = (1,)
        self.mapping = list(range(10))

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self, *args, **kwargs) -> None:
        """Download train and test MNIST data from PyTorch canonical source."""
        TorchMNIST(self.data_dir, train=True, download=True)
        TorchMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        mnist_full = TorchMNIST(self.data_dir, train=True, transform=self.transform)
        self.data_train, self.data_val = random_split(mnist_full, [55000, 5000])  # type: ignore
        self.data_test = TorchMNIST(self.data_dir, train=False, transform=self.transform)


if __name__ == "__main__":
    torch_io.load_and_print_info(MNIST)
