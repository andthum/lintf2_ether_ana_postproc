#!/usr/bin/env python3


"""
Plot the free-energy barriers for diffusion on the electrode surface as
function of the salt concentration for different chain lengths.
"""


# Standard libraries
import argparse

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
from scipy import constants

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


def equalize_yticks(ax):
    """
    Equalize y-ticks so that plots can be better stacked together.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` for which to equalize the y
        ticks.
    """
    ylim = np.asarray(ax.get_ylim())
    ylim_diff = ylim[-1] - ylim[0]
    yticks = np.asarray(ax.get_yticks())
    yticks_valid = (yticks >= ylim[0]) & (yticks <= ylim[-1])
    yticks = yticks[yticks_valid]
    if ylim_diff >= 10 and ylim_diff < 20:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if np.all(yticks >= 0) and np.all(yticks < 10) and ylim_diff > 2:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the free-energy barriers for diffusion on the electrode surface"
        " as function of the salt concentration for different chain lengths."
    )
)
parser.add_argument(
    "--surfq",
    type=str,
    required=False,
    choices=("q1",),  # "q0", "q0.25", "q0.5", "q0.75"),
    default="q1",
    # Other layers than the 1st and 2nd Li-layer at negative electrodes
    # with a surface charge of q = -1 e/nm^2 (might) require clustering
    # of the slabs.
    help="Surface charge in e/nm^2.",
)
parser.add_argument(
    "--cmp",
    type=str,
    required=False,
    choices=("Li",),  # "NBT", "OBT", "OE"),
    default="Li",
    # Other layers than the 1st and 2nd Li-layer at negative electrodes
    # with a surface charge of q = -1 e/nm^2 (might) require clustering
    # of the slabs.
    help="Compound.  Default: %(default)s",
)
args = parser.parse_args()

sols = ("g1", "g4", "peo63")
hex_axes = ("1nn", "2nn")

temp = 423
settings = "pr_nvt%d_nh" % temp  # Simulation settings.
analysis = "axial_hex_dist"  # Analysis name.
analysis_suffix = "_" + args.cmp
outfile = (
    settings
    + "_lintf2_peoN_r_gra_"
    + args.surfq
    + "_sc80_"
    + analysis
    + "_"
    + "_".join(hex_axes)
    + "_"
    + args.cmp
    + "_free_energy.pdf"
)

# beta = k * T
beta = constants.k * temp
beta = 1  # Because the free energy is already given in units of kT.


print("Reading data...")
xdata = [[None for sol in sols] for ax in hex_axes]
ydata = [[None for sol in sols] for ax in hex_axes]
ydata_sd = [[None for sol in sols] for ax in hex_axes]
for hex_ax_ix, hex_ax in enumerate(hex_axes):
    for sol_ix, sol in enumerate(sols):
        infile = (
            settings
            + "_lintf2_"
            + sol
            + "_r_gra_"
            + args.surfq
            + "_sc80_"
            + analysis
            + "_"
            + hex_ax
            + "_"
            + args.cmp
            + "_free_energy.txt.gz"
        )
        x, y, y_sd = np.loadtxt(infile, unpack=True)
        xdata[hex_ax_ix][sol_ix] = x
        ydata[hex_ax_ix][sol_ix] = y
        ydata_sd[hex_ax_ix][sol_ix] = y_sd
del x, y, y_sd


print("Creating plot(s)...")
xlabel = r"Li-to-EO Ratio $r$"
xlim = (0, 0.4 + 0.0125)

legend_title = r"$\sigma_s = -%.2f$ $e$/nm$^2$" % float(args.surfq[1:])
n_legend_cols = 2

labels_ax = (r"Axis $1$", r"Axis $2$")
labels_sol = (r"$n_{EO} = 2$", r"$n_{EO} = 5$", r"$n_{EO} = 64$")
linestyles = ("solid", "dashed")
markers = [("o", "s", "^"), ("p", "D", "v")]

cmap = plt.get_cmap()
c_vals = np.arange(len(sols))
c_norm = max(1, len(sols) - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot free-energy barriers.
    ylabel = (
        r"Barrier Height $\Delta F_{"
        + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
        + r"}(r) / k_B T$"
    )
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for hex_ax_ix, _hex_ax in enumerate(hex_axes):
        for sol_ix, _sol in enumerate(sols):
            ax.errorbar(
                xdata[hex_ax_ix][sol_ix],
                ydata[hex_ax_ix][sol_ix],
                yerr=ydata_sd[hex_ax_ix][sol_ix],
                label=labels_sol[sol_ix] if hex_ax_ix == 0 else None,
                linestyle=linestyles[hex_ax_ix],
                marker=markers[hex_ax_ix][sol_ix],
                # alpha=leap.plot.ALPHA,
            )
    for hex_ax_ix, _hex_ax in enumerate(hex_axes):
        ax.errorbar(
            [],
            [],
            label=labels_ax[hex_ax_ix],
            linestyle=linestyles[hex_ax_ix],
            color="black",
        )
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    equalize_xticks(ax)
    equalize_yticks(ax)
    legend = ax.legend(
        title=legend_title, ncol=n_legend_cols, **mdtplt.LEGEND_KWARGS_XSMALL
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig(fig)
    plt.close(fig)

    # Plot inverse transition rates.
    # Propagation of uncertainty
    # (https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae).
    # Std[1/A] = 1/|A| * Std[A]/|A| = Std[A]/A^2
    # Std[exp(bA)] = exp(bA) * |b| * Std[A]
    # Std[1/exp(bA)] = Std[exp(bA)]/exp(bA)^2
    #               = exp(bA) * |b| * Std[A] / exp(bA)^2
    #               = |b| * Std[A] / exp(bA)
    ylabel = "Inv. Trans. Rate / a.u."
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for hex_ax_ix, _hex_ax in enumerate(hex_axes):
        for sol_ix, _sol in enumerate(sols):
            ax.errorbar(
                xdata[hex_ax_ix][sol_ix],
                1 / np.exp(-beta * ydata[hex_ax_ix][sol_ix]),
                yerr=(
                    np.abs(-beta)
                    * ydata_sd[hex_ax_ix][sol_ix]
                    / np.exp(-beta * ydata[hex_ax_ix][sol_ix])
                ),
                label=labels_sol[sol_ix] if hex_ax_ix == 0 else None,
                linestyle=linestyles[hex_ax_ix],
                marker=markers[hex_ax_ix][sol_ix],
                # alpha=leap.plot.ALPHA,
            )
    for hex_ax_ix, _hex_ax in enumerate(hex_axes):
        ax.errorbar(
            [],
            [],
            label=labels_ax[hex_ax_ix],
            linestyle=linestyles[hex_ax_ix],
            color="black",
        )
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    equalize_xticks(ax)
    equalize_yticks(ax)
    legend = ax.legend(
        title=legend_title, ncol=n_legend_cols, **mdtplt.LEGEND_KWARGS_XSMALL
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig(fig)
    plt.close(fig)

print("Created {}".format(outfile))
print("Done")
