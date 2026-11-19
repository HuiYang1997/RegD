"""Loader for the benchmark splits used in the RegD paper (the format of Ganea et al. 2018 / ShadowCones):
<prefix>.train_<k>percent, .valid, .valid_neg, .test, .test_neg -- one 'parent<TAB>child' id pair per line."""
from __future__ import annotations

import os

from .graph import read_edges


def load_split(prefix, percent=0):
    """Returns a dict of lists of (child, parent) pairs: train, valid, valid_neg, test, test_neg."""
    files = {"train": f".train_{percent}percent", "valid": ".valid", "valid_neg": ".valid_neg", "test": ".test",
             "test_neg": ".test_neg"}
    return {k: read_edges(prefix + suffix, parent_first=True) for k, suffix in files.items()}


def load_vocab(prefix):
    """id -> name, if <prefix>.vocab exists."""
    path = prefix + ".vocab"
    if not os.path.exists(path):  # MCG / Hearst: <dir>/<name>_vocab.tsv
        path = prefix.replace("_closure.tsv", "_vocab.tsv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf8") as f:
        return dict(line.rstrip("\n").split("\t")[:2] for line in f if line.strip())
