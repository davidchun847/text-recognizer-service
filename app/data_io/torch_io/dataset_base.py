"""Base Dataset class."""
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import torch


SequenceOrTensor = Union[Sequence, torch.Tensor]


class DatasetBase(torch.utils.data.Dataset):
    """
    Base Dataset class that simply processes data and targets through optional transforms.

    Read more: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by transforms.

        Parameters
        ----------
        index

        Returns
        -------
        (datum, target)
        """
        datum, target = self.data[index], self.targets[index]

        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target

    def split_dataset(self, fraction: float, seed: int):
        """
        Split input base_dataset into 2 base datasets, the first of size fraction * size of the base_dataset and the
        other of size (1 - fraction) * size of the base_dataset.
        """
        split_a_size = int(fraction * len(self))
        split_b_size = len(self) - split_a_size
        return torch.utils.data.random_split(  # type: ignore
            self, [split_a_size, split_b_size], generator=torch.Generator().manual_seed(seed)
        )
