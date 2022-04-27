import enum
from collections import defaultdict

from utils import torch_utils


class ESpecialChar(enum.Enum):
    none = 0
    blank = (enum.auto(),)
    start = (enum.auto(),)
    end = (enum.auto(),)
    pad = (enum.auto(),)


class SpecialChars:

    mapping = dict()
    mapping[ESpecialChar.blank] = "<B>"
    mapping[ESpecialChar.start] = "<S>"
    mapping[ESpecialChar.end] = "<E>"
    mapping[ESpecialChar.pad] = "<P>"


def get_all_special_chars():
    tokens = []
    for _, v in SpecialChars.mapping.items():
        tokens.append(v)
    return tokens


def get_num_special_chars():
    return len(SpecialChars.mapping)


def get_special_char(echar):
    assert isinstance(echar, ESpecialChar)
    assert echar in SpecialChars.mapping
    return SpecialChars.mapping[echar]


def get_samples_by_char(samples, labels, mapping):
    samples_by_char = defaultdict(list)
    for sample, label in zip(samples, labels):
        samples_by_char[mapping[label]].append(sample)
    return samples_by_char
