from typing import Dict, Sequence

import numpy as np

from utils import str_utils


def to_categorical(y, num_classes):
    """1-hot encode a tensor."""
    return np.eye(num_classes, dtype="uint8")[y]


def convert_strings_to_labels(
    strings: Sequence[str], mapping: Dict[str, int], length: int, with_start_end_tokens: bool
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