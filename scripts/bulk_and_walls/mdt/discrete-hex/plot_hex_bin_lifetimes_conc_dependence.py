#!/usr/bin/env python3


"""
Plot the residence times of the given compound on the hexagonal lattice
sites on the electrode surface as function of the salt concentration for
different chain lengths.
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
        "Plot the residence times of the given compound on the hexagonal"
        " lattice sites on the electrode surface as function of the salt"
        " concentration for different chain lengths."
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

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "discrete-hex"  # Analysis name.
analysis_suffix = "_" + args.cmp
outfile = (
    settings
    + "_lintf2_g1_g4_peo63_r_gra_"
    + args.surfq
    + "_sc80_"
    + analysis
    + "_"
    + args.cmp
    + "_lifetimes_count_censored.pdf"
)

# Columns to read from the input files.
col_rt_sd = 2
cols = (
    0,  # Li-to-EO ratio.
    1,  # Residence time (count censored) [ns].
    col_rt_sd,  # Std of residence time [ns].
    3,  # Corrected sample standard deviation [ns].
    4,  # Corrected coefficient of variation.
    5,  # Unbiased sample skewness (Fisher).
    6,  # Unbiased sample excess kurtosis (Fisher).
    7,  # Sample median [ns].
    8,  # Non-parametric skewness.
    12,  # Sample minimum [ns].
    13,  # Sample maximum [ns].
    14,  # Number of samples.
)
ylabels = (
    "Residence Time / ps",
    "Std. Dev. / ps",
    "Coeff. of Variation",
    "Skewness",
    "Excess Kurtosis",
    "Median / ps",
    "Non-Parametric Skewness",
    "Min. Residence Time / ps",
    "Max. Residence Time / ps",
    "No. of Samples",
)
if len(ylabels) != len(cols) - 2:
    raise ValueError(
        "`len(ylabels)` ({}) != `len(cols)` - 2"
        " ({})".format(len(ylabels), len(cols) - 2)
    )
ylabels_has_unit = np.array([False for ylb in ylabels])
ylabels_has_unit[[0, 1, 5, 7, 8]] = True


print("Reading data...")
xdata = [None for sol in sols]
ydata = [None for sol in sols]
ydata_sd = [None for sol in sols]
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
        + args.cmp
        + "_lifetimes_continuous.txt.gz"
    )
    data = np.loadtxt(infile, usecols=cols, unpack=True)
    xdata[sol_ix] = data[0]
    ydata[sol_ix] = data[1:]
    ydata[sol_ix] = np.delete(ydata[sol_ix], cols.index(col_rt_sd) - 1, axis=0)
    ydata_sd[sol_ix] = data[cols.index(col_rt_sd)]
    ydata[sol_ix][ylabels_has_unit] *= 1e3  # ns -> ps.
    ydata_sd[sol_ix] *= 1e3  # Std[c*A] = |c| * Std[A]
del data


print("Creating plot(s)...")
xlabel = r"Li-to-EO Ratio $r$"
xlim = (0, 0.4 + 0.0125)
ylim = (None, None)

legend_title = (
    r"$\sigma_s = -%.2f$ $e$/nm$^2$" % float(args.surfq[1:])
    + "\n"
    + r"$n_{EO}$"
)

labels = (r"$2$", r"$5$", r"$64$")
markers = ("o", "s", "^")

cmap = plt.get_cmap()
c_vals = np.arange(len(sols))
c_norm = max(1, len(sols) - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for ylb_ix, ylabel in enumerate(ylabels):
        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(color=colors)

        if ylabel == "Coeff. of Variation":
            y_exp = 1  # Coeff. of variation of exp. distribution.
        elif ylabel == "Skewness":
            y_exp = 2  # Skewness of exponential distribution.
        elif ylabel == "Excess Kurtosis":
            y_exp = 6  # Excess kurtosis of exponential distribution.
        elif ylabel == "Non-Parametric Skewness":
            y_exp = 1 - np.log(2)  # Non-param. skew. of exp. dist.
        else:
            y_exp = None
            n_legend_cols = 3
        if y_exp is not None:
            ax.axhline(
                y=y_exp,
                color="tab:green",
                linestyle="dashed",
                label="Exp. Dist.",
            )
            n_legend_cols = 2

        for sol_ix, _sol in enumerate(sols):
            if ylb_ix == cols.index(col_rt_sd) - 2:
                ax.errorbar(
                    xdata[sol_ix],
                    ydata[sol_ix][ylb_ix],
                    yerr=ydata_sd[sol_ix],
                    label=labels[sol_ix],
                    marker=markers[sol_ix],
                )
            else:
                ax.plot(
                    xdata[sol_ix],
                    ydata[sol_ix][ylb_ix],
                    label=labels[sol_ix],
                    marker=markers[sol_ix],
                )
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
        equalize_xticks(ax)
        equalize_yticks(ax)
        legend = ax.legend(
            title=legend_title,
            ncol=n_legend_cols,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig)

        # Log scale y.
        ax.relim()
        ax.autoscale()
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlim=xlim, ylim=ylim)
        pdf.savefig(fig)
        plt.close(fig)

print("Created {}".format(outfile))
print("Done")
