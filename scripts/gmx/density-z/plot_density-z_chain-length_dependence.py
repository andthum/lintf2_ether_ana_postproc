#!/usr/bin/env python3


"""Plot density profiles for different chain lengths."""


# Standard libraries
import argparse

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

    Notes
    -----
    This function relies on global variables!
    """
    ylim = np.asarray(ax.get_ylim())
    ylim_diff = ylim[-1] - ylim[0]
    yticks = np.asarray(ax.get_yticks())
    yticks_valid = (yticks >= ylim[0]) & (yticks <= ylim[-1])
    yticks = yticks[yticks_valid]
    if ylim_diff >= 10 and ylim_diff < 20:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if not args.common_ylim:
        if np.all(yticks >= 0) and np.all(yticks < 10) and ylim_diff > 2:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


# Input parameters.
parser = argparse.ArgumentParser(
    description="Plot density profiles for different chain lengths."
)
parser.add_argument(
    "--surfq",
    type=str,
    required=True,
    choices=("q0", "q0.25", "q0.5", "q0.75", "q1"),
    help="Surface charge in e/nm^2.",
)
parser.add_argument(
    "--common-ylim",
    required=False,
    default=False,
    action="store_true",
    help="Use common y limits for all plots.",
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "density-z"  # Analysis name.
analysis_suffix = "_number"  # Analysis name specification.
tool = "gmx"  # Analysis software.
outfile = (  # Output file name.
    settings
    + "_lintf2_peoN_20-1_gra_"
    + args.surfq
    + "_sc80_"
    + analysis
    + analysis_suffix
)
if args.common_ylim:
    outfile += "_common_ylim.pdf"
else:
    outfile += ".pdf"

cols = (  # Columns to read from the input file(s).
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


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_gra_" + args.surfq + "_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, args.surfq, sort_key="O_per_chain"
)


print("Reading data and creating plot(s)...")
file_suffix = analysis + analysis_suffix + ".xvg.gz"
infiles = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
n_infiles = len(infiles)

Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm
box_z_max = np.max(Sims.boxes_z)

plot_sections = ("left", "right", "full")
xmin = 0
xmax = Elctrd.BULK_START / 10  # A -> nm
if args.surfq == "q0":
    ymax = tuple((6.25, 4, 2.5, 3.1) for _ in plot_sections)
elif args.surfq == "q0.25":
    ymax = ((4, 7, 10.5, 2.8), (12.5, 2, 2, 4.8))
    ymax += (tuple(np.max(ymax, axis=0)),)
elif args.surfq == "q0.5":
    ymax = ((4.2, 11.5, 20, 4), (25, 3.1, 2, 5.8))
    ymax += (tuple(np.max(ymax, axis=0)),)
elif args.surfq == "q0.75":
    ymax = ((4, 23, 36, 4.6), (23, 4.6, 3.1, 7))
    ymax += (tuple(np.max(ymax, axis=0)),)
elif args.surfq == "q1":
    ymax = ((4.6, 46, 62.5, 4.6), (82.5, 4.6, 4.6, 6.25))
    ymax += (tuple(np.max(ymax, axis=0)),)
else:
    raise ValueError("Unknown surface charge --surfq: '{}'".format(args.surfq))
if args.common_ylim:
    # ymax = tuple(
    #     tuple(None for _cmp in compounds)
    #     for _plt_sec in plot_sections
    # )
    ymax = tuple(
        tuple(6.5 for _cmp in compounds) for _plt_sec in plot_sections
    )

cmap = plt.get_cmap()
mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for y_normed in (True, False):
        for ps_ix, plt_sec in enumerate(plot_sections):
            for cmp_ix, cmp in enumerate(compounds):
                fig, ax = plt.subplots(clear=True)
                ax.set_prop_cycle(
                    color=[cmap(i / (n_infiles - 1)) for i in range(n_infiles)]
                )

                if plt_sec in ("left", "right"):
                    # Also, for the plot of the right electrode, the
                    # electrode position will be shifted to zero.
                    leap.plot.elctrd_left(ax)

                for sim_ix, Sim in enumerate(Sims.sims):
                    x, y = np.loadtxt(
                        infiles[sim_ix],
                        comments=["#", "@"],
                        usecols=(0, cols[cmp_ix]),
                        unpack=True,
                    )
                    if plt_sec == "left":
                        y = y[: len(y) // 2]
                        x = x[: len(x) // 2]
                        x -= elctrd_thk
                    elif plt_sec == "right":
                        y = y[len(y) // 2 :]
                        x = x[len(x) // 2 :]
                        x += elctrd_thk
                        x -= Sim.box[2] / 10  # A -> nm
                        x *= -1  # Ensure positive x-axis.
                    if y_normed:
                        bulk_dens = Sim.dens["atm_type"][cmp]["num"]
                        bulk_dens *= 1e3  # 1/A^3 -> 1/nm^3
                        y /= bulk_dens

                    if (
                        args.surfq == "q0"
                        and Sim.O_per_chain == 6
                        and cmp == "Li"
                    ):
                        color = "tab:red"
                        ax.plot([], [])  # Increment color cycle.
                    elif (
                        args.surfq == "q1"
                        and Sim.O_per_chain == 2
                        and cmp in ("NBT", "OBT")
                        and plt_sec == "right"
                    ):
                        color = "tab:red"
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

                ylabel = (
                    r"Density $\rho_{"
                    + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp]
                    + r"}"
                )
                if y_normed:
                    ylabel += (
                        r" / \rho_{"
                        + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp]
                        + r"}^{bulk}$"
                    )
                    ylim = (0, ymax[ps_ix][cmp_ix])
                else:
                    ylabel += r"$ / nm$^{-3}$"
                    ylim = (0, None)
                if plt_sec == "left":
                    xlabel = r"Distance to Electrode / nm"
                    xlim = (xmin, xmax)
                elif plt_sec == "right":
                    xlabel = r"Distance to Electrode / nm"
                    # Reverse x-axis.
                    xlim = (xmax, xmin)
                else:
                    xlabel = r"$z$ / nm"
                    xlim = (0, box_z_max)
                ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
                equalize_xticks(ax)
                equalize_yticks(ax)

                legend_title = (
                    r"%.2f$" % Sims.surfqs[0]
                    + r" $e$/nm$^2$"
                    + "\n"
                    + r"$r = %.2f$" % Sims.Li_O_ratios[0]
                    + "\n"
                    + r"$n_{EO}$"
                )
                if plt_sec == "left":
                    if args.surfq == "q0":
                        legend_title = r"$\sigma_s = " + legend_title
                    else:
                        legend_title = r"$\sigma_s = +" + legend_title
                    legend_loc = "upper right"
                elif plt_sec == "right":
                    if args.surfq == "q0":
                        legend_title = r"$\sigma_s = " + legend_title
                    else:
                        legend_title = r"$\sigma_s = -" + legend_title
                    legend_loc = "upper left"
                else:
                    if args.surfq == "q0":
                        legend_title = r"$\sigma_s = " + legend_title
                    else:
                        legend_title = r"$\sigma_s = \pm" + legend_title
                    legend_loc = "upper center"
                legend = ax.legend(
                    title=legend_title,
                    ncol=1 + n_infiles // (4 + 1),
                    loc=legend_loc,
                    **mdtplt.LEGEND_KWARGS_XSMALL,
                )
                legend.get_title().set_multialignment("center")

                pdf.savefig()
                plt.close()

print("Created {}".format(outfile))
print("Done")
