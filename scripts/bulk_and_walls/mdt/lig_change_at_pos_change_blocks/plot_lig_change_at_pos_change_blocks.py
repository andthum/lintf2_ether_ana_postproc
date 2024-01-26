#!/usr/bin/env python3

"""
Plot the number of lithium-ion ligands that dissociate, associate or
remain coordinated during the crossing of a free-energy barrier for a
single simulation.
"""


# Standard libraries
import argparse
import os

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# First-party libraries
import lintf2_ether_ana_postproc as leap


def equalize_yticks(ax):
    """
    Equalize y-ticks so that plots can be better stacked together.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` for which to equalize the y
        ticks.
    """
    ylim = np.asarray(ax.get_ylim())
    ylim_diff = ylim[-1] - ylim[0]
    yticks = np.asarray(ax.get_yticks())
    yticks_valid = (yticks >= ylim[0]) & (yticks <= ylim[-1])
    yticks = yticks[yticks_valid]
    if ylim_diff >= 10 and ylim_diff < 20:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if np.all(yticks >= 0) and np.all(yticks < 10) and ylim_diff > 2:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the number of lithium-ion ligands that dissociate, associate or"
        " remain coordinated during the crossing of a free-energy barrier for"
        " a single simulation."
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
    required=True,
    choices=("Li-OE", "Li-OBT"),
    help=(
        "Compounds for which to plot the coordination change upon barrier"
        " crossing."
    ),
)
parser.add_argument(
    "--barrier",
    type=str,
    required=True,
    help="Position of the crossed free-energy barrier, e.g. 1.23A.",
)
args = parser.parse_args()
cmp1, cmp2 = args.cmp.split("-")

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "lig_change_at_pos_change_blocks"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
ana_path = os.path.join(
    analysis + analysis_suffix, analysis + analysis_suffix + "_" + args.barrier
)
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis
    + "_"
    + args.cmp
    + "_"
    + args.barrier
    + ".pdf"
)

# Molecules that correspond to the second atom type (`cmp2`).
if cmp2 == "OE":
    cmps2 = ("OE", "PEO")
elif cmp2 == "OBT":
    cmps2 = ("OBT", "NTf2")
else:
    raise ValueError("Unknown `cmp2`: '{}'".format(cmp2))

xticklabels = ("Dissociated", "Associated", "Remained")

# Rows and columns to read from the file that contains the ligand
# exchange information.
row_n_trans = 1  # Row that contains the no. of valid barrier crossings.
# Rows that contain ligand exchange information.
# `rows` must not be a tuple, because it will be used as index array.
rows = [
    4,  # No. dissociated ligands.
    5,  # No. associated ligands.
    6,  # No. remained/stayed/persisted ligands.
]
cols = (tuple(range(2, 10)), tuple(range(10, 18)))
if len(rows) != len(xticklabels):
    raise ValueError(
        "`len(rows)` ({}) != `len(xticklabels)`"
        " ({})".format(len(rows), len(xticklabels))
    )
if len(cols) != len(cmps2):
    raise ValueError(
        "`len(cols)` ({}) != `len(cmps2)` ({})".format(len(cols), len(cmps2))
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


print("Reading data and creating plot(s)...")
Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm.

# File containing the position of the crossed free-energy barrier.
file_suffix = analysis + analysis_suffix + "_" + args.barrier + "_bins.txt.gz"
infile = leap.simulation.get_ana_file(Sim, ana_path, tool, file_suffix)
bins = np.loadtxt(infile)
if bins.shape != (3,):
    raise ValueError("The bin file must contain exactly three bin edges.")
bins /= 10  # Angstrom -> nm.
barrier = bins[1]

# Check whether the free-energy barrier lies in the left or right half
# of the simulation box.
box_z = Sim.box[2] / 10  # Angstrom -> nm.
if barrier <= box_z / 2:
    pkp_type = "left"
    # Convert absolute barrier position to distance to the electrodes.
    barrier -= elctrd_thk
else:
    pkp_type = "right"
    # Convert absolute barrier position to distance to the electrodes.
    barrier += elctrd_thk
    barrier -= box_z
    barrier *= -1  # Ensure positive distance values.

# Assignment of columns indices.
# to/aw = toward/away; succ/unsc = successful, unsuccessful.
if pkp_type == "left":
    to_succ_ix, to_unsc_ix = 2, 6
    aw_succ_ix, aw_unsc_ix = 0, 4
elif pkp_type == "right":
    to_succ_ix, to_unsc_ix = 0, 4
    aw_succ_ix, aw_unsc_ix = 2, 6
else:
    raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))
col_ndx = (to_succ_ix, to_unsc_ix, aw_succ_ix, aw_unsc_ix)


# Create plots.
xdata = np.arange(len(xticklabels))
xlim = (-0.2, 2.2)

labels = ("To, Succ.", "To, Unsc.", "From, Succ.", "From, Unsc.")
colors = ("tab:blue", "tab:cyan", "tab:red", "tab:pink")
linestyles = ("solid", "dashed", "solid", "dashed")
if pkp_type == "left":
    markers = ("<", 4, ">", 5)
elif pkp_type == "right":
    markers = (">", 5, "<", 4)
else:
    raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))

legend_title = (
    r"$n_{EO} = %d$, " % Sim.O_per_chain
    + r"$r = %.4f$" % Sim.Li_O_ratio
    + "\n"
    + r"$F_{"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
    + r"}$ Maximum Position "
    + r"$%.2f$ nm" % barrier
)
if surfq is not None:
    if pkp_type == "left":
        surfq = "+"
    elif pkp_type == "right":
        surfq = "-"
    else:
        raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))
    legend_title = (
        r"$\sigma_s = "
        + surfq
        + r"%.2f$ $e$/nm$^2$" % (Sim.surfq * 100)  # e/A^2 -> e/nm^2
        + "\n"
        + legend_title
    )

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for cmp_ix, cmp2 in enumerate(cmps2):
        fig, ax = plt.subplots(clear=True)

        # File containing the ligand exchange information.
        file_suffix = (
            analysis + analysis_suffix + "_" + args.barrier + ".txt.gz"
        )
        infile = leap.simulation.get_ana_file(Sim, ana_path, tool, file_suffix)
        lig_exchange_data = np.loadtxt(infile, usecols=cols[cmp_ix])

        for cix, col_ix in enumerate(col_ndx):
            # Number of valid barrier crossings.
            n_trans = lig_exchange_data[row_n_trans, col_ix]
            ydata = lig_exchange_data[rows, col_ix]
            yerr = lig_exchange_data[rows, col_ix + 1] / np.sqrt(n_trans)
            ax.errorbar(
                xdata,
                ydata,
                yerr,
                label=labels[cix],
                color=colors[cix],
                marker=markers[cix],
                linestyle=linestyles[cix],
                alpha=leap.plot.ALPHA,
            )

        if cmp2 in ("OE", "OBT"):
            ylabel = r"$" + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp2] + r"$ Atoms"
        elif cmp2 == "PEO":
            ylabel = leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp2] + " Chains"
        elif cmp2 == "NTf2":
            ylabel = leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp2] + " Anions"
        else:
            raise ValueError("Unknown `cmp2`: '{}'".format(cmp2))
        if cmp2 == "OE":
            ylim = (0, 6.4)
        else:
            ylim = (0, 2.9)
        ylabel = "No. of " + ylabel
        ax.set(
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            xticks=xdata,
            xticklabels=xticklabels,
        )
        ax.tick_params(axis="x", which="minor", top=False, bottom=False)
        equalize_yticks(ax)
        legend = ax.legend(
            title=legend_title, ncol=2, **mdtplt.LEGEND_KWARGS_XSMALL
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig)
        plt.close(fig)

print("Created {}".format(outfile))
print("Done")
