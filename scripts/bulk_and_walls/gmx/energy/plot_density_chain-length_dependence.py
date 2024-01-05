#!/usr/bin/env python3


"""
Plot the time-averaged mass density of the system as function of the PEO
chain length.
"""


# Standard libraries
import argparse

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the time-averaged density of the system as function of the PEO"
        " chain length."
    ),
)
args = parser.parse_args()

# Temperatures [K].
temps = (303, 423)
# Simulation settings.
settings_lst = ["eq_npt%d_pr_nh" % temp for temp in temps]
# System name.
system = "lintf2_peoN_20-1_sc80"
# Output filename.
outfile = "eq_npT_pr_nh_" + system + "_density.pdf"

# Columns to read from the input files.
cols = (
    0,  # Number of ether oxygens per PEO chain.
    15,  # Density [kg/nm^3].
    16,  # Standard deviation of the density [kg/nm^3].
)


print("Reading data...")
infiles = [
    settings + "_" + system + "_energy.txt.gz" for settings in settings_lst
]
n_infiles = len(infiles)
xdata = [None for infile in infiles]
ydata = [None for infile in infiles]
ydata_sd = [None for infiles in infiles]
for set_ix, infile in enumerate(infiles):
    xdata[set_ix], ydata[set_ix], ydata_sd[set_ix] = np.loadtxt(
        infile, usecols=cols, unpack=True
    )


print("Creating plot(s)...")
xlabel = r"Ether Oxygens per Chain $n_{EO}$"
xlim_lin = (0, 135)
xlim_log = (1, 200)
legend_title = r"$r = 0.05$" + "\n" + "T / K"
markers = ("o", "s")
if len(markers) != n_infiles:
    raise ValueError(
        "`len(markers)` ({}) != `n_infiles`"
        " ({})".format(len(markers), n_infiles)
    )

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    fig, ax = plt.subplots(clear=True)
    for set_ix, temp in enumerate(temps):
        ax.errorbar(
            xdata[set_ix],
            ydata[set_ix],
            yerr=ydata_sd[set_ix],
            label=r"$%d$" % temp,
            marker=markers[set_ix],
        )
    ax.set(xlabel=xlabel, ylabel=r"Density / kg m$^{-3}$", xlim=xlim_lin)
    legend = ax.legend(title=legend_title)
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    # Log scale x.
    ax.relim()
    ax.autoscale()
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_xlim(xlim_log)
    pdf.savefig()
    # Log scale y.
    ax.relim()
    ax.autoscale()
    ax.set_xscale("linear")
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set_xlim(xlim_lin)
    pdf.savefig()
    # Log scale xy.
    ax.relim()
    ax.autoscale()
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_xlim(xlim_log)
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
