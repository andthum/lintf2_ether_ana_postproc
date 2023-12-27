#!/usr/bin/env python3


"""
Plot the coordination relaxation times and the renewal times as function
of the PEO chain length in one plot.
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
        "Plot the coordination relaxation times and the renewal times as"
        " function of the PEO chain length in one plot."
    )
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
prefix = settings + "_lintf2_peoN_20-1_sc80_"
path_to_acf = "../../lifetime_autocorr/chain-length_dependence/"
infile_acf = (
    path_to_acf
    + prefix
    + "lifetime_autocorr_combined_Li-OE_Li-OBT_Li-ether_Li-NTf2.txt.gz"
)
infile_renew_peo = prefix + "renewal_events_Li-ether_continuous.txt.gz"
infile_renew_tfsi = prefix + "renewal_events_Li-NTf2_continuous.txt.gz"
outfile = prefix + "lifetime_autocorr_and_renewal_times.pdf"


print("Reading data...")
acf_data = np.loadtxt(infile_acf, usecols=(0, 1, 2))
rnw_peo_data = np.loadtxt(infile_renew_peo, usecols=(0, 29))
rnw_tfsi_data = np.loadtxt(infile_renew_tfsi, usecols=(0, 29))


print("Creating plot(s)...")
xlabel = r"Ether Oxygens per Chain $n_{EO}$"
xlim = (1, 200)
legend_title = r"$r = 0.05$"
labels = (
    r"$Li-O_{PEO}$ Relaxation",
    r"$Li-O_{TFSI}$ Relaxation",
    r"$Li-PEO$ Renewal",
    r"$Li-TFSI$ Renewal",
)
colors = ("tab:blue", "tab:cyan", "tab:red", "tab:orange")
markers = ("^", "v", "s", "D")

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    fig, ax = plt.subplots(clear=True)
    for i, label in enumerate(labels):
        if label.startswith(r"$Li-O_{PEO}$"):
            xdata, ydata = acf_data[:, 0], acf_data[:, 1]
        elif label.startswith(r"$Li-O_{TFSI}$"):
            xdata, ydata = acf_data[:, 0], acf_data[:, 2]
        elif label.startswith(r"$Li-PEO$"):
            xdata, ydata = rnw_peo_data[:, 0], rnw_peo_data[:, 1]
        elif label.startswith(r"$Li-TFSI$"):
            xdata, ydata = rnw_tfsi_data[:, 0], rnw_tfsi_data[:, 1]
        else:
            raise ValueError("Unknown 'label': '{}'".format(label))
        valid = np.isfinite(ydata)
        xdata, ydata = xdata[valid], ydata[valid]
        ax.plot(xdata, ydata, label=label, color=colors[i], marker=markers[i])
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel="Lifetime / ns", xlim=xlim)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    legend = ax.legend(title=legend_title, **mdtplt.LEGEND_KWARGS_XSMALL)
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
