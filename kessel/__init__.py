"""Kessel: model merging by tensor name. Mergekit-compatible configs; quantized checkpoints; rank-1 edits."""

from ._version import __version__
from .config import MergeConfig, parse
from .merge import run
from .methods import METHODS, Method, MergeInput, register
from .ops import apply_directions, diff, extract_directions, load_directions, project, save_directions

__all__ = ["__version__", "MergeConfig", "parse", "run", "METHODS", "Method", "MergeInput", "register",
           "apply_directions", "diff", "extract_directions", "load_directions", "project", "save_directions"]
