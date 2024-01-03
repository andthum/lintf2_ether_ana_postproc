#!/usr/bin/env python3


"""
Plot the average end-to-end distance and the average radius of gyration
of the PEO chains as function of the salt concentration for various
chain lengths.
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
        " gyration of the PEO chains as function of the salt concentration for"
        " various chain lengths."
    ),
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "polystat"  # Analysis name.
tool = "gmx"  # Analysis software.
outfile = (  # Output file name.
    settings + "_lintf2_g1_g4_peo63_r_sc80_" + analysis + ".pdf"
)

cols = (  # Columns to read from the input file(s).
    1,  # End-to-end distance [nm].
    2,  # Radius of gyration [nm].
)


print("Creating Simulation instance(s)...")
sys_pats = [
    "lintf2_" + sol + "_[0-9]*-[0-9]*_sc80" for sol in ("g1", "g4", "peo63")
]
Sims_lst = []
for sys_pat in sys_pats:
    set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
    Sims = leap.simulation.get_sims(
        sys_pat, set_pat, path_key="bulk", sort_key="Li_O_ratio"
    )
    Sims_lst.append(Sims)


print("Reading data...")
file_suffix = analysis + ".xvg.gz"
polymer_stats_lst, polymer_stats_sd_lst = [], []
ratio_lst, ratio_sd_lst = [], []
for Sims in Sims_lst:
    infiles = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
    n_infiles = len(infiles)

    polymer_stats = np.full((2, n_infiles), np.nan, dtype=np.float64)
    polymer_stats_sd = np.full_like(polymer_stats, np.nan)
    for sim_ix, infile in enumerate(infiles):
        stats = np.loadtxt(infile, comments=["#", "@"], usecols=cols)
        stats **= 2
        polymer_stats[:, sim_ix] = np.nanmean(stats, axis=0)
        polymer_stats_sd[:, sim_ix] = np.nanstd(stats, ddof=1, axis=0)
        polymer_stats_sd[:, sim_ix] /= np.sqrt(
            Sims.res_nums["solvent"][sim_ix]
        )
    del stats
    polymer_stats_lst.append(polymer_stats)
    polymer_stats_sd_lst.append(polymer_stats_sd)

    ratio = polymer_stats[0] / polymer_stats[1]
    # Propagation of uncertainty:
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
    # Std[A/B] =
    #   |A/B| * sqrt{(Std[A]/A)^2 + (Std[B]/B)^2 - 2 Cov[A,B]/(AB))
    ratio_sd = np.abs(polymer_stats[0] / polymer_stats[1]) * (
        (polymer_stats_sd[0] / polymer_stats[0]) ** 2
        + (polymer_stats_sd[1] / polymer_stats[1]) ** 2
    )
    ratio_lst.append(ratio)
    ratio_sd_lst.append(ratio_sd)

del polymer_stats, polymer_stats_sd, ratio, ratio_sd


print("Creating plot(s)...")
xlabel = r"Li-to-EO Ratio $r$"
xlim = (0, 0.4 + 0.0125)
legend_title = r"$n_{EO}$"
labels = ("End-to-End", "Gyration")
markers = ("^", "v")
if len(labels) != polymer_stats_lst[0].shape[0]:
    raise ValueError(
        "`len(labels)` ({}) != `polymer_stats_lst[0].shape[0]`"
        " ({})".format(len(labels), polymer_stats_lst[0].shape[0])
    )
if len(markers) != polymer_stats_lst[0].shape[0]:
    raise ValueError(
        "`len(markers)` ({}) != `polymer_stats_lst[0].shape[0]`"
        " ({})".format(len(markers), polymer_stats_lst[0].shape[0])
    )

cmap = plt.get_cmap()
c_vals = np.arange(len(Sims_lst))
c_norm = max(1, len(Sims_lst) - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot end-to-end distance and radius of gyration.
    ylabel = r"$\langle R^2 \rangle$ / nm$^2$"
    ylim = (2e-2, 5e1)
    fig, ax = plt.subplots(clear=True)
    for stats_ix in range(len(polymer_stats_lst[0])):
        ax.errorbar(
            [],
            [],
            label=labels[stats_ix],
            color="black",
            marker=markers[stats_ix],
        )
    for sims_ix, Sims in enumerate(Sims_lst):
        for stats_ix, stats in enumerate(polymer_stats_lst[sims_ix]):
            ax.errorbar(
                Sims.Li_O_ratios,
                stats,
                yerr=polymer_stats_sd_lst[sims_ix][stats_ix],
                label=(
                    r"$n_{EO} = %d$" % Sims.O_per_chain[0]
                    if stats_ix == 0
                    else None
                ),
                color=colors[sims_ix],
                marker=markers[stats_ix],
            )
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    equalize_xticks(ax)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    legend = ax.legend(**mdtplt.LEGEND_KWARGS_XSMALL)
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
    markers = ("o", "s", "^")
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sims_ix, Sims in enumerate(Sims_lst):
        ax.errorbar(
            Sims.Li_O_ratios,
            ratio_lst[sims_ix],
            yerr=ratio_sd_lst[sims_ix],
            label=r"$%d$" % Sims.O_per_chain[0],
            marker=markers[sims_ix],
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

print("Created {}".format(outfile))
print("Done")
