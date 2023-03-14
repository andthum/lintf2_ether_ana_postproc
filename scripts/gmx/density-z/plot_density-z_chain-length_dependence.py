#!/usr/bin/env python3


"""Plot density profiles"""


# Standard libraries
import glob
import os

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
q = "q0"  # Surface charge in e/nm^2.
settings = "pr_nvt423_nh"
analysis = "density-z"
suffix = "_number"
tool = "gmx"
outfile = (
    settings
    + "_lintf2_peoN_20-1_gra_"
    + q
    + "_sc80_"
    + analysis
    + suffix
    + ".pdf"
)
cols = (  # Columns to read from the input files.
    # 0,  # bin edges [nm]
    2,  # Li number density [nm^-3]
    5,  # NBT number density [nm^-3]
    6,  # OBT number density [nm^-3]
    7,  # OE number density [nm^-3]
)
compounds = ("Li", "NBT", "OBT", "OE")
if len(compounds) != len(cols):
    raise ValueError(
        "`len(compounds)` ({}) != `len(cols)`"
        " ({})".format(len(compounds), len(cols))
    )

# Get simulation directories.
SimPaths = leap.simulation.SimPaths()
pattern_system = "lintf2_*[0-9]*_20-1_gra_q[0-9]*_sc80"
pattern_settings = "[0-9][0-9]_" + settings + "_" + pattern_system
pattern = os.path.join(SimPaths.PATHS[q], pattern_system, pattern_settings)
paths = glob.glob(pattern)
Sims = leap.simulation.Simulations(*paths)

# Get input files.
infiles = []
file_suffix = analysis + suffix + ".xvg"
for i, path in enumerate(Sims.paths_ana):
    fname = Sims.fnames_ana_base[i] + file_suffix
    fpath = os.path.join(path, tool, analysis, fname)
    if not os.path.isfile(fpath):
        raise FileNotFoundError("No such file: '{}'".format(fpath))
    infiles.append(fpath)
n_infiles = len(infiles)

# Create plot
Elctrd = leap.simulation.Electrode()
xmax = None
ymax = [6.5, 4, 3, 3]
cmap = plt.get_cmap()
mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Left electrode (x original, y normalized).
    for cmp_ix, cmp in enumerate(compounds):
        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(
            color=[cmap(i / n_infiles) for i in range(n_infiles)]
        )
        for sim_ix, Sim in enumerate(Sims.sims):
            x, y = np.loadtxt(
                infiles[sim_ix],
                comments=["#", "@"],
                usecols=(0, cols[cmp_ix]),
                unpack=True,
            )
            x -= Elctrd.ELCTRD_THK
            y /= Sim.dens[cmp]
