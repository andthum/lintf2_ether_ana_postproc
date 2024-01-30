#!/usr/bin/env python3


"""
Plot the coordination correlation times and the renewal times as
function of the salt concentration in one plot.
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
        "Plot the coordination correlation times and the renewal times as"
        " function of the salt concentration in one plot."
    )
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
prefix = settings + "_lintf2_" + args.sol + "_r_sc80_"
path_to_acf = "../../lifetime_autocorr/conc_dependence/"
infile_acf = (
    path_to_acf
    + prefix
    + "lifetime_autocorr_combined_Li-OE_Li-OBT_Li-ether_Li-NTf2.txt.gz"
)
infile_renew_peo = prefix + "renewal_times_Li-ether_continuous.txt.gz"
infile_renew_tfsi = prefix + "renewal_times_Li-NTf2_continuous.txt.gz"
outfile = prefix + "lifetime_autocorr_and_renewal_times.pdf"


print("Reading data...")
acf_data = np.loadtxt(infile_acf)
rnw_peo_data = np.loadtxt(infile_renew_peo, usecols=(0, 29))
# rnw_tfsi_data = np.loadtxt(infile_renew_tfsi, usecols=(0, 29))


print("Creating plot(s)...")
xlabel = r"Li-to-EO Ratio $r$"
xlim = (0, 0.4 + 0.0125)
if args.sol == "g1":
    legend_title = r"$n_{EO} = 2$"
elif args.sol == "g4":
    legend_title = r"$n_{EO} = 5$"
elif args.sol == "peo63":
    legend_title = r"$n_{EO} = 64$"
else:
    raise ValueError("Unknown --sol: {}".format(args.sol))
labels = (
    r"$Li-O_{PEO}$ Correlation",
    r"$Li-O_{TFSI}$ Correlation",
    r"$Li-PEO$ Correlation",
    r"$Li-TFSI$ Correlation",
    r"$Li-PEO$ Renewal",
    # r"$Li-TFSI$ Renewal",
)
colors = ("tab:blue", "tab:cyan", "tab:red", "tab:orange", "tab:purple")
markers = ("^", "v", "s", "D", "o")
linestyles = ("solid",) * 4 + ("dashed",)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    fig, ax = plt.subplots(clear=True)
    for i, label in enumerate(labels):
        if label.startswith(r"$Li-O_{PEO}$ Correlation"):
            xdata, ydata = acf_data[:, 0], acf_data[:, 1]
        elif label.startswith(r"$Li-O_{TFSI}$ Correlation"):
            xdata, ydata = acf_data[:, 0], acf_data[:, 2]
        elif label.startswith(r"$Li-PEO$ Correlation"):
            xdata, ydata = acf_data[:, 0], acf_data[:, 3]
        elif label.startswith(r"$Li-TFSI$ Correlation"):
            xdata, ydata = acf_data[:, 0], acf_data[:, 4]
        elif label.startswith(r"$Li-PEO$ Renewal"):
            xdata, ydata = rnw_peo_data[:, 0], rnw_peo_data[:, 1]
        # elif label.startswith(r"$Li-TFSI$ Renewal"):
        #     xdata, ydata = rnw_tfsi_data[:, 0], rnw_tfsi_data[:, 1]
        else:
            raise ValueError("Unknown 'label': '{}'".format(label))
        valid = np.isfinite(ydata)
        xdata, ydata = xdata[valid], ydata[valid]
        ax.plot(
            xdata,
            ydata,
            label=label,
            color=colors[i],
            marker=markers[i],
            linestyle=linestyles[i],
            alpha=leap.plot.ALPHA,
        )
    ax.set(xlabel=xlabel, ylabel="Time Scale / ns", xlim=xlim)
    equalize_xticks(ax)
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
