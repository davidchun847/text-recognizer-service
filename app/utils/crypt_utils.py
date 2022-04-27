import string
import random
import hashlib
from typing import Union
from pathlib import Path


def compute_sha256(filename: Union[Path, str]):
    """Return SHA256 checksum of a file."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rand_str_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))
