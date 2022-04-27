from typing import Union
from typing import Dict, Sequence
from enum import Enum, auto

import torch

from utils import str_utils


class TorchRunType(Enum):
    none = 0
    train = auto()
    predict = auto()


def first_element(
    x: torch.Tensor, element: Union[int, float], dim: int = 1
) -> torch.Tensor:
    """
    Return indices of first occurence of element in x. If not found, return length of x along dim.

    Based on https://discuss.pytorch.org/t/first-nonzero-index/24769/9

    Examples
    --------
    >>> first_element(torch.tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
    tensor([2, 1, 3])
    """
    nonz = x == element
    ind = ((nonz.cumsum(dim) == 1) & nonz).max(dim).indices
    ind[ind == 0] = x.shape[dim]
    return ind


def convert_strings_to_labels(
    strings: Sequence[str], mapping: Dict[str, int], length: int
) -> torch.Tensor:
    """
    Convert sequence of N strings to a (N, length) ndarray, with each string wrapped with <S> and <E> tokens,
    and padded with the <P> token.
    """
    str_pad = str_utils.get_special_char(str_utils.ESpecialChar.pad)
    str_start = str_utils.get_special_char(str_utils.ESpecialChar.start)
    str_end = str_utils.get_special_char(str_utils.ESpecialChar.end)
    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping[str_pad]
    for i, string in enumerate(strings):
        tokens = list(string)
        tokens = [str_start, *tokens, str_end]
        for ii, token in enumerate(tokens):
            labels[i, ii] = mapping[token]
    return labels
