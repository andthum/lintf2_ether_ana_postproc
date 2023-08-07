#!/usr/bin/env python3


"""
Plot the contact histograms and average coordination numbers for two
given compounds as function of the PEO chain length.
"""


# Standard libraries
import argparse

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# First-party libraries
import lintf2_ether_ana_postproc as leap


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
        "Plot the contact histograms and average coordination numbers for two"
        " given compounds as function of the PEO chain length."
    )
)
parser.add_argument(
    "--cmp",
    type=str,
    required=True,
    choices=("Li-OE", "Li-OBT"),
    help=(
        "Compounds for which to plot the coordination numbers.  Default:"
        " %(default)s"
    ),
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "contact_hist"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    settings + "_lintf2_peoN_20-1_sc80_" + analysis + analysis_suffix + ".pdf"
)

cols = (  # Columns to read from the input file(s).
    # 0,  # Number of contacts N.
    1,  # Ratio of Li ions that have contact with N different O atoms.
    2,  # Ratio of Li ions that have contact with N different PEO/TFSI molecules.  # noqa: E501
)
n_cols = len(cols)
if args.cmp == "Li-OE":
    col_labels = (r"$Li - O_{PEO}$", r"$Li - PEO$")
    xlim_hist = [(0, 8), (0, 6)]
    ylim_cn = (0, 6.5)
elif args.cmp == "Li-OBT":
    col_labels = (r"$Li - O_{TFSI}$", r"$Li - TFSI$")
    xlim_hist = [(0, 8), (0, 6)]
    ylim_cn = (0, 0.42)
else:
    raise ValueError("Invalid --cmp: {}".format(args.cmp))
if len(col_labels) != len(cols):
    raise ValueError(
        "`len(col_labels)` ({}) != `len(cols)`"
        " ({})".format(len(col_labels), len(cols))
    )
if len(xlim_hist) != len(cols):
    raise ValueError(
        "`len(xlim_hist)` ({}) != `len(cols)`"
        " ({})".format(len(xlim_hist), len(cols))
    )


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="O_per_chain"
)


print("Reading data and creating plot(s)...")
file_suffix = analysis + "_" + args.cmp + ".txt.gz"
infiles = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
n_infiles = len(infiles)

# 1st and 2nd moments of the coordination number histograms.
cn_mom1 = np.full((n_cols, n_infiles), np.nan, dtype=np.float64)
cn_mom2 = np.full((n_cols, n_infiles), np.nan, dtype=np.float64)

legend_title = r"$r = %.2f$" % Sims.Li_O_ratios[0]

cmap = plt.get_cmap()
c_vals = np.arange(n_infiles)
c_norm = n_infiles - 1
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for col_ix, col in enumerate(cols):
        # Plot coordination number histograms.
        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(color=colors)
        for sim_ix, Sim in enumerate(Sims.sims):
            # Read input files.
            data = np.loadtxt(infiles[sim_ix], usecols=(0, col))
            # Skip last row that contains the sum of each column.
            data = data[:-1]
            n_contacts, probabilities = data.T
            # Calculate average coordination numbers.
            cn_mom1[col_ix][sim_ix] = np.sum(n_contacts * probabilities)
            cn_mom2[col_ix][sim_ix] = np.sum(n_contacts**2 * probabilities)

            if Sim.O_per_chain == 2:
                color = "tab:red"
                ax.plot([], [])  # Increment color cycle.
            elif Sim.O_per_chain == 5:
                color = "tab:orange"
                ax.plot([], [])  # Increment color cycle.
            else:
                color = None
            ax.plot(
                n_contacts,
                probabilities,
                label=r"$%d$" % Sim.O_per_chain,
                color=color,
                alpha=leap.plot.ALPHA,
            )
        ax.set(
            xlabel="Coordination Number",
            ylabel="Probability",
            xlim=xlim_hist[col_ix],
            ylim=(0, 1),
        )
        ax.xaxis.set_tick_params(which="minor", bottom=False, top=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend = ax.legend(
            title=(
                col_labels[col_ix] + "\n" + legend_title + "\n" + r"$n_{EO}$"
            ),
            ncol=1 + n_infiles // (6 + 1),
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

    # Plot average coordination numbers.
    markers = ("^", "v")
    fig, ax = plt.subplots(clear=True)
    for col_ix in range(n_cols):
        # Uncertainty of the mean value.
        if args.cmp.startswith("Li-"):
            yerr = np.sqrt(
                (cn_mom2[col_ix] - cn_mom1[col_ix] ** 2)
                / Sims.res_nums["cation"]
            )
        else:
            raise NotImplementedError(
                "Uncertainty of the average coordination number not"
                " implemented for compounds that don't start with 'Li-'"
            )
        ax.errorbar(
            Sims.O_per_chain,
            cn_mom1[col_ix],
            yerr=yerr,
            label=col_labels[col_ix],
            marker=markers[col_ix],
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel=r"Ether Oxygens per Chain $n_{EO}$",
        ylabel="Coordination Number",
        xlim=(1, 200),
        ylim=ylim_cn,
    )
    equalize_yticks(ax)
    ax.legend(title=legend_title)
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
