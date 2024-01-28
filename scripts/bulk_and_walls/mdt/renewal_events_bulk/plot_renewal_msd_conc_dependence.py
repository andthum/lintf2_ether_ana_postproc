#!/usr/bin/env python3


"""
Plot the mean squared displacement (MSD) between two renewal events for
two given compounds as function of the salt concentration.
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
        "Plot the mean squared displacement (MSD) between two renewal events"
        " for two given compounds as function of the salt concentration."
    )
)
parser.add_argument(
    "--sol",
    type=str,
    required=True,
    choices=("g1", "g4", "peo63"),
    help="Solvent name.",
)
parser.add_argument(
    "--cmp",
    type=str,
    required=True,
    choices=("Li-ether", "Li-NTf2"),
    help="Compounds for which to calculate the renewal MSD.",
)
args = parser.parse_args()
cmp1, cmp2 = args.cmp.split("-")

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "renewal_events"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
ana_path = os.path.join(analysis, analysis + analysis_suffix)
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    settings
    + "_lintf2_"
    + args.sol
    + "_r_sc80_renewal_msd"
    + analysis_suffix
    + ".pdf"
)


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_" + args.sol + "_[0-9]*-[0-9]*_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="Li_O_ratio"
)


print("Reading data...")
msd = np.full((3, Sims.n_sims), np.nan, dtype=np.float32)
msd_sd = np.full_like(msd, np.nan)
for sim_ix, Sim in enumerate(Sims.sims):
    file_suffix = analysis + analysis_suffix + ".txt.gz"
    try:
        infile = leap.simulation.get_ana_file(Sim, ana_path, tool, file_suffix)
    except FileNotFoundError:
        continue
    displ_ref = np.loadtxt(infile, usecols=(10, 11, 12))  # x, y, z.
    displ_sel = np.loadtxt(infile, usecols=(13, 14, 15))  # x, y, z.
    n_data = displ_ref.shape[0]
    displ_ref /= 10  # A -> nm.
    displ_sel /= 10  # A -> nm.
    # MSD of reference compounds while attached to a selection compound.
    msd_ref = np.sum(displ_ref**2, axis=1)  # x^2 + y^2 + z^2.
    msd[0, sim_ix] = np.mean(msd_ref)
    msd_sd[0, sim_ix] = np.std(msd_ref, ddof=1) / np.sqrt(n_data)
    del msd_ref
    # MSD of selection compounds while attached to a reference compound.
    msd_sel = np.sum(displ_sel**2, axis=1)  # x^2 + y^2 + z^2.
    msd[1, sim_ix] = np.mean(msd_sel)
    msd_sd[1, sim_ix] = np.std(msd_sel, ddof=1) / np.sqrt(n_data)
    del msd_sel
    # MSD of selection compounds relative to the coordinating selection
    # compound.
    msd_relative = np.sum((displ_ref - displ_sel) ** 2, axis=1)
    msd[2, sim_ix] = np.mean(msd_relative)
    msd_sd[2, sim_ix] = np.std(msd_relative, ddof=1) / np.sqrt(n_data)
    del msd_relative


print("Creating plot(s)...")
xlabel = r"Li-to-EO Ratio $r$"
xlim = (0, 0.4 + 0.0125)
legend_title = r"$n_{EO} = %d$" % Sims.O_per_chain[0]

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    fig, ax = plt.subplots(clear=True)
    ax.errorbar(
        Sims.Li_O_ratios,
        msd[1],
        yerr=msd_sd[1],
        label=leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp2],
        color="tab:blue",
        marker="o",
        alpha=leap.plot.ALPHA,
    )
    ax.errorbar(
        Sims.Li_O_ratios,
        msd[0],
        yerr=msd_sd[0],
        label=leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1],
        color="tab:green",
        marker="^",
        alpha=leap.plot.ALPHA,
    )
    ax.errorbar(
        Sims.Li_O_ratios,
        msd[2],
        yerr=msd_sd[2],
        label=leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1] + " Relative",
        color="tab:orange",
        marker="D",
        alpha=leap.plot.ALPHA,
    )
    ax.set(xlabel=xlabel, ylabel=r"MSD($\tau_3$) / nm$^2$", xlim=xlim)
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
