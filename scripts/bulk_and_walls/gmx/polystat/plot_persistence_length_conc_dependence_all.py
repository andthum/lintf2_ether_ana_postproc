#!/usr/bin/env python3


"""
Plot the average persistence length of the PEO chains as function of the
salt concentration for various chain lengths.
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
        "Plot the average persistence length of the PEO chains as function of"
        " the salt concentration for various chain lengths."
    ),
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "polystat"  # Analysis name.
analysis_suffix = "persist"  # Analysis name specification.
tool = "gmx"  # Analysis software.
outfile = (  # Output file name.
    settings + "_lintf2_g1_g4_peo63_r_sc80_" + analysis_suffix + ".pdf"
)

cols = (  # Columns to read from the input file(s).
    1,  # Persistence length [bonds].
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
file_suffix = analysis_suffix + ".xvg.gz"
plen_lst, plen_sd_lst = [], []
for Sims in Sims_lst:
    infiles = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
    n_infiles = len(infiles)
    plen = np.full(n_infiles, np.nan, dtype=np.float64)
    plen_sd = np.full_like(plen, np.nan)
    for sim_ix, infile in enumerate(infiles):
        pl = np.loadtxt(infile, comments=["#", "@"], usecols=cols)
        plen[sim_ix] = np.nanmean(pl)
        plen_sd[sim_ix] = np.nanstd(pl, ddof=1)
        plen_sd[sim_ix] /= np.sqrt(Sims.res_nums["solvent"][sim_ix])
    del pl
    plen_lst.append(plen)
    plen_sd_lst.append(plen_sd)
del plen, plen_sd


print("Creating plot(s)...")
xlabel = r"Li-to-EO Ratio $r$"
xlim = (0, 0.4 + 0.0125)
legend_title = r"$n_{EO}$"
markers = ("o", "s", "^")

cmap = plt.get_cmap()
c_vals = np.arange(len(Sims_lst))
c_norm = max(1, len(Sims_lst) - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sims_ix, Sims in enumerate(Sims_lst):
        ax.errorbar(
            Sims.Li_O_ratios,
            plen_lst[sims_ix],
            yerr=plen_sd_lst[sims_ix],
            label=r"$%d$" % Sims.O_per_chain[0],
            marker=markers[sims_ix],
        )
    ax.set(xlabel=xlabel, ylabel="Persistence Length / Bonds", xlim=xlim)
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
        ax.set_xlim(xlim)
        pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
