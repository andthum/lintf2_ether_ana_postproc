#!/usr/bin/env python3


"""
Plot the average coordination number for two given compounds as function
of the distance to the electrodes for different PEO chain lengths.
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
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

# First-party libraries
import lintf2_ether_ana_postproc as leap


def equalize_xticks(ax):
    """
    Equalize x-ticks so that plots can be better stacked together.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` for which to equalize the x
        ticks.
    """
    xlim = np.asarray(ax.get_xlim())
    xlim_diff = xlim[-1] - xlim[0]
    if xlim_diff > 2.5 and xlim_diff < 5:
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))


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
        "Plot the average coordination number for two given compounds as"
        " function of the distance to the electrodes for different PEO chain"
        " lengths."
    )
)
parser.add_argument(
    "--surfq",
    type=str,
    required=True,
    choices=("q0", "q0.25", "q0.5", "q0.75", "q1"),
    help="Surface charge in e/nm^2.",
)
parser.add_argument(
    "--cmp",
    type=str,
    required=True,
    choices=("Li-OE", "Li-OBT", "Li-PEO", "Li-NTf2"),
    help="Compounds for which to plot the coordination numbers.",
)
parser.add_argument(
    "--common-ylim",
    required=False,
    default=False,
    action="store_true",
    help="Use common y limits for all plots.",
)
args = parser.parse_args()
cmp1, cmp2 = args.cmp.split("-")

analysis_suffix = "_" + cmp1  # Analysis name specification.
if cmp2 in ("OE", "PEO"):
    analysis_suffix += "-OE"
elif cmp2 in ("OBT", "NTf2"):
    analysis_suffix += "-OBT"
else:
    raise ValueError("Unknown `cmp2`: '{}'".format(cmp2))

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "contact_hist_slab-z"  # Analysis name.
ana_path = os.path.join(analysis, analysis + analysis_suffix)
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    settings
    + "_lintf2_peoN_20-1_gra_"
    + args.surfq
    + "_sc80_"
    + analysis
    + "_"
    + args.cmp
)
if args.common_ylim:
    outfile += "_common_ylim.pdf"
else:
    outfile += ".pdf"

# Columns to read from the input files.
cols = (0,)  # Number of contacts N.
if cmp2 in ("OE", "OBT"):
    # Ratio of Li ions that have contact with N different O atoms.
    cols += (1,)
elif cmp2 in ("PEO", "NTf2"):
    # Ratio of Li ions that have contact with N different PEO/TFSI
    # molecules.
    cols += (2,)
else:
    raise ValueError("Unknown --cmp: '{}'".format(args.cmp))


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_gra_" + args.surfq + "_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, args.surfq, sort_key="O_per_chain"
)


print("Reading data...")
file_suffix_pattern = analysis + analysis_suffix + "_[0-9]*-[0-9]*.txt.gz"
xdata = [None for Sim in Sims.sims]
ydata = [None for Sim in Sims.sims]
for sim_ix, Sim in enumerate(Sims.sims):
    fname_pattern = Sim.fname_ana_base + file_suffix_pattern
    fpath_pattern = os.path.join(Sim.path_ana, tool, ana_path, fname_pattern)
    infiles = glob.glob(fpath_pattern)
    if len(infiles) == 0:
        raise ValueError(
            "Could not find any file matching the pattern"
            " '{}'".format(fpath_pattern)
        )

    file_prefix = Sim.fname_ana_base + analysis + analysis_suffix
    slab_starts = np.full(len(infiles), np.nan, dtype=np.float64)
    slab_stops = np.full_like(slab_starts, np.nan)
    cn = np.full_like(slab_starts, np.nan)
    for f_ix, infile in enumerate(infiles):
        # Get the position of the analyzed slab/bin.
        slab_starts[f_ix], slab_stops[f_ix] = get_slab(infile, file_prefix)
        # Read file.
        data = np.loadtxt(infile, usecols=cols)
        # Skip last row that contains the sum of each column.
        data = data[:-1]
        n_contacts, probabilities = data.T
        n_contacts = np.round(n_contacts, out=n_contacts).astype(np.uint16)
        # Calculate the average coordination number.
        cn[f_ix] = np.sum(n_contacts * probabilities)

    # Sort data by the slab position.
    sort_ix_starts = np.argsort(slab_starts)
    sort_ix_stops = np.argsort(slab_stops)
    if not np.array_equal(sort_ix_starts, sort_ix_stops):
        raise ValueError("`sort_ix_starts` != `sort_ix_stops`")
    slab_starts = slab_starts[sort_ix_starts]
    slab_stops = slab_stops[sort_ix_stops]
    cn = cn[sort_ix_starts]
    xdata[sim_ix] = slab_starts + (slab_stops - slab_starts) / 2
    xdata[sim_ix] /= 10  # Angstrom -> nm.
    ydata[sim_ix] = cn
del infiles, slab_starts, slab_stops, cn


print("Creating plot(s)...")
Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # Angstrom -> nm.
box_z_max = np.max(Sims.boxes_z)

plot_sections = ("left", "right", "full")
xmin = 0
xmax = Elctrd.BULK_START / 10  # Angstrom -> nm.

ylabel = "Coordination Number"
if args.common_ylim:
    if args.cmp == "Li-OE":
        ylim = (1.6, 6.4)
    elif args.cmp == "Li-PEO":
        ylim = (0.8, 4.8)  # (0.8, 3) No place for legend.
    elif args.cmp in ("Li-OBT", "Li-NTf2"):
        ylim = (0, 2.6)
    else:
        raise ValueError("Unknown --cmp: '{}'".format(args.cmp))
else:
    ylim = (None, None)

legend_title_base = (
    r"$"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
    + "-"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp2]
    + r"$"
    + "\n"
)
n_legend_cols = 1 + Sims.n_sims // (3 + 1)

cmap = plt.get_cmap()
c_vals = np.arange(Sims.n_sims)
c_norm = max(1, Sims.n_sims - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for plt_sec in plot_sections:
        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(color=colors)

        if plt_sec in ("left", "right"):
            # Also, for the plot of the right electrode, the
            # electrode position will be shifted to zero.
            leap.plot.elctrd_left(ax)

        for sim_ix, Sim in enumerate(Sims.sims):
            x = np.copy(xdata[sim_ix])
            y = np.copy(ydata[sim_ix])
            if plt_sec == "left":
                y = y[: len(y) // 2]
                x = x[: len(x) // 2]
                x -= elctrd_thk
            elif plt_sec == "right":
                y = y[len(y) // 2 :]
                x = x[len(x) // 2 :]
                x += elctrd_thk
                x -= Sim.box[2] / 10  # Angstrom -> nm.
                x *= -1  # Ensure positive x-axis.

            if Sim.O_per_chain == 2:
                color = "tab:red"
                ax.plot([], [])  # Increment color cycle.
            elif Sim.O_per_chain == 3:
                color = "tab:brown"
                ax.plot([], [])  # Increment color cycle.
            elif Sim.O_per_chain == 5:
                color = "tab:orange"
                ax.plot([], [])  # Increment color cycle.
            else:
                color = None
            ax.plot(
                x,
                y,
                label=r"$%d$" % Sim.O_per_chain,
                color=color,
                alpha=leap.plot.ALPHA,
            )

        if plt_sec == "left":
            xlabel = r"Distance to Electrode / nm"
            xlim = (xmin, xmax)
        elif plt_sec == "right":
            xlabel = r"Distance to Electrode / nm"
            xlim = (xmax, xmin)  # Reverse x-axis.
        else:
            xlabel = r"$z$ / nm"
            xlim = (0, box_z_max)
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
        equalize_xticks(ax)
        equalize_yticks(ax)

        legend_title = (
            r"%.2f$ $e$/nm$^2$, " % Sims.surfqs[0]
            + r"$r = %.2f$" % Sims.Li_O_ratios[0]
            + "\n"
            + r"$n_{EO}$"
        )
        if plt_sec == "left":
            if args.surfq == "q0":
                legend_title = r"$\sigma_s = " + legend_title
            else:
                legend_title = r"$\sigma_s = +" + legend_title
            legend_loc = "right"
        elif plt_sec == "right":
            if args.surfq == "q0":
                legend_title = r"$\sigma_s = " + legend_title
            else:
                legend_title = r"$\sigma_s = -" + legend_title
            legend_loc = "left"
        else:
            if args.surfq == "q0":
                legend_title = r"$\sigma_s = " + legend_title
            else:
                legend_title = r"$\sigma_s = \pm" + legend_title
            legend_loc = "center"
        if cmp2 == "OE":
            legend_loc = "lower " + legend_loc
        elif cmp2 in ("PEO", "OBT", "NTf2"):
            legend_loc = "upper " + legend_loc
        else:
            raise ValueError("Unknown `cmp2`: '{}'".format(cmp2))
        legend = ax.legend(
            title=legend_title_base + legend_title,
            ncol=n_legend_cols,
            loc=legend_loc,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")

        pdf.savefig(fig)
        plt.close(fig)

print("Created {}".format(outfile))
print("Done")
