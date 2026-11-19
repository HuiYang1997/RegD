"""RegD: hierarchy embeddings as Euclidean regions (boxes / balls)."""
from .graph import Hierarchy, read_edges
from .model import PRESETS, Config, RegD, options_table

__version__ = "0.1.0"
__all__ = ["RegD", "Config", "PRESETS", "Hierarchy", "read_edges", "options_table"]
