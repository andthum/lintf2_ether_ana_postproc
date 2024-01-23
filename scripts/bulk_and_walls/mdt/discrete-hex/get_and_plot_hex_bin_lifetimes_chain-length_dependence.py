#!/usr/bin/env python3


"""
Calculate and plot the residence time / lifetime of the given compound
on the hexagonal lattice sites on the electrode surface as function of
the chain length.
"""


# Standard libraries
import argparse
import glob
import os
import re

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# First-party libraries
import lintf2_ether_ana_postproc as leap


def get_slab(dir_name, prefix):
    """Get the position of the analyzed slab from the directory name."""
    if not os.path.isdir(dir_name):
        raise FileNotFoundError("No such directory: '{}'".format(dir_name))
    dir_name = os.path.basename(dir_name)  # Remove path to the directory.
    if not dir_name.startswith(prefix):
        raise ValueError(
            "The directory name '{}' does not start with"
            " '{}'".format(dir_name, prefix)
        )
    slab = dir_name[len(prefix) :]  # Remove `prefix`.
    slab = re.sub("[^0-9|.|-]", "", slab)  # Remove non-numeric characters.
    slab = slab.strip(".")  # Remove leading and trailing periods.
    slab = slab.split("-")  # Split at hyphens.
    if len(slab) != 2:
        raise ValueError("Invalid slab: {}".format(slab))
    slab = [float(slab) for slab in slab]
    slab_start, slab_stop = min(slab), max(slab)
    return slab_start, slab_stop


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Calculate and plot the residence time / lifetime of the given"
        " compound on the hexagonal lattice sites on the electrode surface as"
        " function of the chain length."
    )
)
parser.add_argument(
    "--surfq",
    type=str,
    required=False,
    choices=("q1",),  # "q0", "q0.25", "q0.5", "q0.75"),
    default="q1",
    # Other layers than the 1st and 2nd Li-layer at negative electrodes
    # with a surface charge of q = -1 e/nm^2 (might) require clustering
    # of the slabs.
    help="Surface charge in e/nm^2.",
)
parser.add_argument(
    "--cmp",
    type=str,
    required=False,
    choices=("Li",),  # "NBT", "OBT", "OE"),
    default="Li",
    # Other layers than the 1st and 2nd Li-layer at negative electrodes
    # with a surface charge of q = -1 e/nm^2 (might) require clustering
    # of the slabs.
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--continuous",
    required=False,
    default=False,
    action="store_true",
    help="Use the 'continuous' definition of the remain probability function.",
)
args = parser.parse_args()

if args.continuous:
    con = "_continuous"
else:
    con = ""

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "discrete-hex"  # Analysis name.
analysis_suffix = "_" + args.cmp
tool = "mdt"  # Analysis software.
outfile_base = (
    settings
    + "_lintf2_peoN_20-1_gra_"
    + args.surfq
    + "_sc80_"
    + analysis
    + "_"
    + args.cmp
    + "_lifetimes"
    + con
)
outfile_txt = outfile_base + ".txt.gz"
outfile_pdf = outfile_base + ".pdf"

# Time conversion factor to convert trajectory steps to ns.
time_conv = 2e-3
# Number of moments to calculate.  For calculating the skewness, the 2nd
# and 3rd (central) moments are required, for the kurtosis the 2nd and
# 4th (central) moments are required.
n_moms = 4
# Fit method of `scipy.optimize.curve_fit` to use for all fits.
fit_method = "trf"


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_gra_" + args.surfq + "_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, args.surfq, sort_key="O_per_chain"
)


print("Getting analysis directories...")
ana_parent_dir = os.path.join(tool, analysis, analysis + analysis_suffix)
ana_dirs = [None for Sim in Sims.sims]
for sim_ix, path in enumerate(Sims.paths_ana):
    dir_pattern = analysis + analysis_suffix + "_[0-9]*-[0-9]*"
    dir_pattern = os.path.join(path, ana_parent_dir, dir_pattern)
    dirs = glob.glob(dir_pattern)
    if len(dirs) == 0:
        raise ValueError(
            "Could not find any directory matching the pattern"
            " '{}'".format(dir_pattern)
        )

    # Get the directory that contains the data for the slab/bin that is
    # closest to the negative electrode.
    dir_prefix = analysis + analysis_suffix
    slab_starts = np.full(len(dirs), np.nan, dtype=np.float64)
    for d_ix, directory in enumerate(dirs):
        slab_starts[d_ix], _slab_stops = get_slab(directory, dir_prefix)
    ix_max = np.argmax(slab_starts)
    ana_dirs[sim_ix] = dirs[ix_max]
del dirs, slab_starts, _slab_stops


print("Calculating lifetimes directly from `dtrj`...")
# Read discrete trajectory.
