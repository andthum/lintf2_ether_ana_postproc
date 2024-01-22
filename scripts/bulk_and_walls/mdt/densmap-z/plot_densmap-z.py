#!/usr/bin/env python3

"""
Plot 2-dimensional density maps as generated by the MDTools script
`densmap.py
<https://github.com/andthum/mdtools/blob/f96f90de92866d0da0bc614ae59d3435d6263a10/scripts/structure/densmap.py>`_
"""


# Maximum Li densities.
# (Determined with the aim to set a common `vmax` for all simulations.)
#
# Surface-charge dependence.
# q0, q0.25 and q0.5: No patterns.
# q0.75: Patterns only for g1 system.
#
# Chain-length dependence (q1, r = 1/20).
# rho_max = 14.5  (g0, 155.26-159.09 A, including g0 system)
# rho_max =  7.0  (g1, 139.25-143.32 A, excluding g0 system)
#
# Concentration dependence (q1).
# g1:    rho_max =  8.0  (r = 1/4, 193.32-197.40 A)
# g4:    rho_max = 11.5  (r = 2/5, 224.59-228.66 A)
# peo63: rho_max = 15.0  (r = 1/4, 166.39-170.11 A)


# Standard libraries
import argparse
import os
import re

# Third-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# First-party libraries
import lintf2_ether_ana_postproc as leap


def get_slab(fname, prefix):
    """Get the position of the analyzed slab from the file name."""
    if not os.path.isfile(fname):
        raise FileNotFoundError("No such file: '{}'".format(fname))
    fname = os.path.basename(fname)  # Remove path to the file.
    fname = os.path.splitext(fname)[0]  # Remove (first) file extension.
    if not fname.startswith(prefix):
        raise ValueError(
            "The file name '{}' does not start with '{}'".format(fname, prefix)
        )
    slab = fname[len(prefix) :]  # Remove `prefix`.
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
        "Plot 2-dimensional density maps as generated by the MDTools script"
        " densmap.py"
    )
)
parser.add_argument(
    "--system",
    type=str,
    required=True,
    help="Name of the simulated system, e.g. lintf2_g1_20-1_gra_q1_sc80.",
)
parser.add_argument(
    "--settings",
    type=str,
    required=False,
    default="pr_nvt423_nh",
    help=(
        "String describing the used simulation settings.  Default:"
        " %(default)s."
    ),
)
parser.add_argument(
    "--cmp",
    type=str,
    required=False,
    default="Li",
    choices=("Li", "NTf2", "ether", "NBT", "OBT", "OE"),
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--slab",
    type=str,
    required=True,
    help="The analyzed slab in xy plane, e.g. 0.12-3.45A.",
)
args = parser.parse_args()

analysis = "densmap-z"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
ana_path = os.path.join(analysis, analysis + analysis_suffix)
tool = "mdt"  # Analysis software.
outfile = (
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis
    + "_"
    + args.cmp
    + "_"
    + args.slab
    + ".pdf"
)


print("Creating Simulation instance(s)...")
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    top_path = "q%g" % surfq
else:
    surfq = None
    top_path = "bulk"
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, top_path)


print("Reading data...")
# Get input file that contain the average compound density in each bin.
analysis_bin_pop = "discrete-z"
file_suffix_bin_pop = analysis_bin_pop + "_Li_bin_population.txt.gz"
infile_bin_pop = leap.simulation.get_ana_file(
    Sim, analysis_bin_pop, tool, file_suffix_bin_pop
)

# Get input file that contains the 2D density map.
file_suffix = analysis + analysis_suffix + "_" + args.slab + ".txt.gz"
infile = leap.simulation.get_ana_file(Sim, ana_path, tool, file_suffix)

# Get the average compound density in the slab/bin.
file_prefix = Sim.fname_ana_base + analysis + analysis_suffix
slab_start, slab_stop = get_slab(infile, file_prefix)
tol = 0.02
bin_starts, bin_stops, bin_dens = np.loadtxt(
    infile_bin_pop, usecols=(0, 1, 6), unpack=True
)
bin_starts, bin_stops = np.round(bin_starts, 2), np.round(bin_stops, 2)
bin_ix_start = np.flatnonzero(
    np.isclose(bin_starts, slab_start, rtol=0, atol=tol)
)
bin_ix_stop = np.flatnonzero(
    np.isclose(bin_stops, slab_stop, rtol=0, atol=tol)
)
if len(bin_ix_start) != 1:
    raise ValueError("`len(bin_ix_start)` ({}) != 1".format(len(bin_ix_start)))
if len(bin_ix_stop) != 1:
    raise ValueError("`len(bin_ix_stop)` ({}) != 1".format(len(bin_ix_stop)))
bin_ix_start, bin_ix_stop = bin_ix_start[0], bin_ix_stop[0]
if bin_ix_stop != bin_ix_start:
    raise ValueError(
        "`bin_ix_stop` ({}) != `bin_ix_start`"
        " ({})".format(bin_ix_stop, bin_ix_start)
    )
slab_dens = bin_dens[bin_ix_start]  # 1/Angstrom^3.

data = np.loadtxt(infile)
if np.any(data < 0):
    raise ValueError("At least one data point is less than zero.")
x = data[1:, 0] / 10  # Angstrom -> nm.
y = data[0, 1:] / 10  # Angstrom -> nm.
z = data[1:, 1:] / slab_dens
del data
# Transform the array from matrix convention (origin at upper left) to
# coordinate convention (origin at lower left).
z = np.ascontiguousarray(z.T[::-1])


print("Creating plot(s)...")
xlabel = r"$x$ / nm"
ylabel = r"$y$ / nm"
zlabel = (
    r"Density $\rho_{"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
    + r"}(x, y) / \rho_{"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
    + r"}^{layer}$"
)
vmax = None
xlim = (0, 1)
ylim = xlim

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    fig, ax = plt.subplots(clear=True)
    # figwidth_original = fig.get_figwidth()
    # fig.set_figwidth(figwidth_original * 0.9)
    img, cbar = mdtplt.imshow_new(
        ax=ax, X=z, vmin=0, vmax=vmax, extent=(0, x.max(), 0, y.max())
    )
    ax.set(xlabel=xlabel, ylabel=ylabel)
    cbar.set_label(zlabel)
    yticks = np.asarray(ax.get_yticks())
    mask = (yticks >= ax.get_xlim()[0]) & (yticks <= ax.get_xlim()[1])
    ax.set_xticks(yticks[mask])
    pdf.savefig(fig)

    # fig.set_figwidth(figwidth_original * 0.95)
    ax.set(xlim=xlim, ylim=ylim)
    yticks = np.asarray(ax.get_yticks())
    mask = (yticks >= ax.get_xlim()[0]) & (yticks <= ax.get_xlim()[1])
    ax.set_xticks(yticks[mask])
    pdf.savefig(fig)
    plt.close(fig)

print("Created {}".format(outfile))
print("Done")
