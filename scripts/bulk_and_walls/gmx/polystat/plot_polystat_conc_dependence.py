#!/usr/bin/env python3


"""
Plot the average end-to-end distance and the average radius of gyration
of the PEO chains as function of the salt concentration.
"""


# Standard libraries
import argparse

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator

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
    ax.xaxis.set_major_locator(MultipleLocator(0.1))


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the average end-to-end distance and the average radius of"
        " gyration of the PEO chains as function of the salt concentration."
    ),
)
parser.add_argument(
    "--sol",
    type=str,
    required=True,
    choices=("g1", "g4", "peo63"),
    help="Solvent name.",
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "polystat"  # Analysis name.
tool = "gmx"  # Analysis software.
outfile = (  # Output file name.
    settings + "_lintf2_" + args.sol + "_r_sc80_" + analysis + ".pdf"
)

cols = (  # Columns to read from the input file(s).
    1,  # End-to-end distance [nm].
    2,  # Radius of gyration [nm].
)


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_" + args.sol + "_[0-9]*-[0-9]*_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="Li_O_ratio"
)


print("Reading data...")
file_suffix = analysis + ".xvg.gz"
infiles = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
n_infiles = len(infiles)

polymer_stats = np.full((2, n_infiles), np.nan, dtype=np.float64)
polymer_stats_sd = np.full_like(polymer_stats, np.nan)
for sim_ix, infile in enumerate(infiles):
    stats = np.loadtxt(infile, comments=["#", "@"], usecols=cols)
    stats **= 2
    polymer_stats[:, sim_ix] = np.nanmean(stats, axis=0)
    polymer_stats_sd[:, sim_ix] = np.nanstd(stats, ddof=1, axis=0)
    polymer_stats_sd[:, sim_ix] /= np.sqrt(Sims.res_nums["solvent"][sim_ix])
del stats

ratio = polymer_stats[0] / polymer_stats[1]
# Propagation of uncertainty:
# https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
# Std[A/B] = |A/B| * sqrt{(Std[A]/A)^2 + (Std[B]/B)^2 - 2 Cov[A,B]/(AB))
ratio_sd = np.abs(polymer_stats[0] / polymer_stats[1]) * (
    (polymer_stats_sd[0] / polymer_stats[0]) ** 2
    + (polymer_stats_sd[1] / polymer_stats[1]) ** 2
)


print("Creating plot(s)...")
xlabel = r"Li-to-EO Ratio $r$"
xlim = (0, 0.4 + 0.0125)
legend_title = r"$n_{EO} = %d$" % Sims.O_per_chain[0]
labels = ("End-to-End", "Gyration")
markers = ("^", "v")
if len(labels) != polymer_stats.shape[0]:
    raise ValueError(
        "`len(labels)` ({}) != `polymer_stats.shape[0]`"
        " ({})".format(len(labels), polymer_stats.shape[0])
    )
if len(markers) != polymer_stats.shape[0]:
    raise ValueError(
        "`len(markers)` ({}) != `polymer_stats.shape[0]`"
        " ({})".format(len(markers), polymer_stats.shape[0])
    )

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot end-to-end distance and radius of gyration.
    ylabel = r"$\langle R^2 \rangle$ / nm$^2$"
    ylim = (None, None)
    fig, ax = plt.subplots(clear=True)
    for stats_ix, stats in enumerate(polymer_stats):
        ax.errorbar(
            Sims.Li_O_ratios,
            stats,
            yerr=polymer_stats_sd[stats_ix],
            label=labels[stats_ix],
            marker=markers[stats_ix],
        )
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    equalize_xticks(ax)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    legend = ax.legend(title=legend_title)
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    yd_min, yd_max = leap.plot.get_ydata_min_max(ax)
    if np.any(np.greater(yd_min, 0)):
        # Log scale y.
        ax.relim()
        ax.autoscale()
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlim=xlim, ylim=ylim)
        pdf.savefig()
    plt.close()

    # Plot end-to-end distance divided by radius of gyration.
    ylabel = r"$\langle R_e^2 \rangle / \langle R_g^2 \rangle$"
    ylim = (None, None)
    fig, ax = plt.subplots(clear=True)
    ax.errorbar(Sims.Li_O_ratios, ratio, yerr=ratio_sd, marker="o")
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    equalize_xticks(ax)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    legend = ax.legend(title=legend_title)
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    yd_min, yd_max = leap.plot.get_ydata_min_max(ax)
    if np.any(np.greater(yd_min, 0)):
        # Log scale y.
        ax.relim()
        ax.autoscale()
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlim=xlim, ylim=ylim)
        pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
