"""SentenceGenerator class and supporting functions."""
import itertools
import re
import string
from typing import Optional

import nltk
import numpy as np

import mypath
from data_io.torch_io.data_module_base import DataModuleBase


class SentenceGenerator:
    """Generate text sentences using the Brown corpus."""

    def __init__(self, max_length: Optional[int] = None):
        self.text = self.brown_text()
        self.word_start_inds = [0] + [
            _.start(0) + 1 for _ in re.finditer(" ", self.text)
        ]
        self.max_length = max_length

    def generate(self, max_length: Optional[int] = None) -> str:
        """
        Sample a string from text of the Brown corpus of length at least one word and at most max_length.
        """
        if max_length is None:
            max_length = self.max_length
        if max_length is None:
            raise ValueError(
                "Must provide max_length to this method or when making this object."
            )

        for _ in range(10):  # Try several times to generate before actually erroring
            try:
                first_ind = np.random.randint(0, len(self.word_start_inds) - 1)
                start_ind = self.word_start_inds[first_ind]
                end_ind_candidates = []
                for ind in range(first_ind + 1, len(self.word_start_inds)):
                    if self.word_start_inds[ind] - start_ind > max_length:
                        break
                    end_ind_candidates.append(self.word_start_inds[ind])
                end_ind = np.random.choice(end_ind_candidates)
                sampled_text = self.text[start_ind:end_ind].strip()
                return sampled_text
            except Exception:  # pylint: disable=broad-except
                pass
        raise RuntimeError("Was not able to generate a valid string")

    @classmethod
    def brown_text(
        cls,
    ):
        """Return a single string with the Brown corpus with all punctuation stripped."""
        sents = cls.load_nltk_brown_corpus()
        text = " ".join(itertools.chain.from_iterable(sents))
        text = text.translate({ord(c): None for c in string.punctuation})
        text = re.sub("  +", " ", text)
        return text

    @classmethod
    def load_nltk_brown_corpus(
        cls,
    ):
        """Load the Brown corpus using the NLTK library."""
        nltk.data.path.append(mypath.dir_data_nltk)
        try:
            nltk.corpus.brown.sents()
        except LookupError:
            mypath.dir_data_nltk.mkdir(parents=True, exist_ok=True)
            nltk.download("brown", download_dir=mypath.dir_data_nltk)
        return nltk.corpus.brown.sents()
