#!/usr/bin/env python3


"""
Plot the average persistence length of the PEO chains as function of the
PEO chain length.
"""


# Standard libraries
import argparse

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the average persistence length of the PEO chains as function of"
        " the PEO chain length."
    ),
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "polystat"  # Analysis name.
analysis_suffix = "persist"  # Analysis name specification.
tool = "gmx"  # Analysis software.
outfile = (  # Output file name.
    settings + "_lintf2_peoN_20-1_sc80_" + analysis_suffix + ".pdf"
)

cols = (  # Columns to read from the input file(s).
    1,  # Persistence length [bonds].
)


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="O_per_chain"
)


print("Reading data...")
file_suffix = analysis_suffix + ".xvg.gz"
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


print("Creating plot(s)...")
xlabel = r"Ether Oxygens per Chain $n_{EO}$"
xlim = (1, 200)
legend_title = r"$r = %.2f$" % Sims.Li_O_ratios[0]

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    fig, ax = plt.subplots(clear=True)
    ax.errorbar(Sims.O_per_chain, plen, yerr=plen_sd, marker="o")
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel="Persistence Length / Bonds", xlim=xlim)
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
