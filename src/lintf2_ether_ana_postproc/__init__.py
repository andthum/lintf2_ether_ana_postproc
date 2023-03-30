"""
`lintf2_ether_ana_postproc` is just a small helper package that contains
analysis postprocessing utilities for my molecular dynamics simulations
of LiTFSI-ether mixtures.
"""


# Local imports
from . import io_handler, misc, plot, simulation


__all__ = ["io_handler", "misc", "plot", "simulation"]

# Project metadata.  Keep in sync with `pyproject.toml`!
__title__ = "lintf2_ether_ana_postproc"
__version__ = "0.0.0"  # Keep in sync with `pyproject.toml`!
__author__ = "Andreas Thum"
__email__ = "coding.andthum@e.mail.de"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 " + __author__
__copyright_notice__ = (
    "This program is part of {}.\n".format(__title__)
    + __copyright__
    + ".\n"
    + __license__
    + "."
)
