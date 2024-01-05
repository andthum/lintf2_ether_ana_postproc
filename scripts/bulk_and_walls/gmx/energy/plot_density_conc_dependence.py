#!/usr/bin/env python3


"""
Plot the time-averaged mass density of the system as function of the
salt concentration.
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
        "Plot the time-averaged density of the system as function of the salt"
        " concentration."
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

# Temperatures [K].
temps = (303, 423)
# Simulation settings.
settings_lst = ["eq_npt%d_pr_nh" % temp for temp in temps]
# System name.
system = "lintf2_" + args.sol + "_r_sc80"
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
xlabel = r"Li-to-EO Ratio $r$"
xlim_lin = (0, 0.4 + 0.0125)
xlim_log = (1e-2, 5e-1)
if args.sol == "g1":
    legend_title = r"$n_{EO} = 2$"
elif args.sol == "g4":
    legend_title = r"$n_{EO} = 5$"
elif args.sol == "peo63":
    legend_title = r"$n_{EO} = 64$"
else:
    raise ValueError("Unknown --sol: {}".format(args.sol))
legend_title += "\n" + "T / K"
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
    equalize_xticks(ax)
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
    equalize_xticks(ax)
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
