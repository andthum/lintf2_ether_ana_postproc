#!/usr/bin/env python3


"""
Plot the contact histograms and average coordination numbers for two
given compounds as function of the PEO chain length.  Estimate the
standard error of the mean values from block averages.
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
        " given compounds as function of the PEO chain length.  Estimate the"
        " standard error of the mean values from block averages."
    )
)
parser.add_argument(
    "--cmp",
    type=str,
    required=True,
    choices=("Li-OE", "Li-OBT", "Li-O"),
    help="Compounds for which to plot the coordination numbers.",
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "contact_hist"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
analysis_dir = analysis + "_block_average"
ana_path = os.path.join(analysis_dir, analysis + analysis_suffix)
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    settings
    + "_lintf2_peoN_20-1_sc80_"
    + analysis_dir
    + analysis_suffix
    + ".pdf"
)

# Blocks used for block averaging
blocks = list(range(0, 1100, 100))
n_blocks = len(blocks) - 1

cols = (  # Columns to read from the input file(s).
    # 0,  # Number of contacts N.
    1,  # Ratio of Li ions that have contact with N different O atoms.
    2,  # Ratio of Li ions that have contact with N different PEO/TFSI molecules.  # noqa: E501
    5,  # Ratio of Li-PEO/TFSI complexes with N-dentate coordination.
)
n_cols = len(cols)
if args.cmp == "Li-OE":
    col_labels = (r"$Li - O_{PEO}$", r"$Li - PEO$", r"$Li - PEO$")
    xlim_hist = [(0, 8), (0, 6), (0, 8)]
    ylim_cn = (0, 6.5)
    ylim_denticity = ylim_cn
elif args.cmp == "Li-OBT":
    col_labels = (r"$Li - O_{TFSI}$", r"$Li - TFSI$", r"$Li - TFSI$")
    xlim_hist = [(0, 8), (0, 6), (0, 8)]
    ylim_cn = (0, 0.42)
    ylim_denticity = (0.9, 1.3)
elif args.cmp == "Li-O":
    col_labels = (r"$Li - O_{tot}$", r"$Li - PEO/TFSI$", r"$Li - PEO/TFSI$")
    xlim_hist = [(0, 8), (0, 6), (0, 8)]
    ylim_cn = (0, 6.5)
    ylim_denticity = ylim_cn
else:
    raise ValueError("Invalid --cmp: {}".format(args.cmp))
if len(col_labels) != n_cols:
    raise ValueError(
        "`len(col_labels)` ({}) != `n_cols`"
        " ({})".format(len(col_labels), n_cols)
    )
if len(xlim_hist) != n_cols:
    raise ValueError(
        "`len(xlim_hist)` ({}) != `n_cols`"
        " ({})".format(len(xlim_hist), n_cols)
    )

# Maximum number of contacts to consider when plotting the fraction of
# lithium ions that have contact to N different O atoms or N different
# PEO/TFSI molecules as function of the chain length.
if args.cmp == "Li-OE":
    n_cnt_max = (
        7,  # Maximum number of different O atoms.
        3,  # Maximum number of different PEO molecules.
        7,  # Maximum denticity of a Li-PEO complex.
    )
elif args.cmp == "Li-OBT":
    n_cnt_max = (
        2,  # Maximum number of different O atoms.
        2,  # Maximum number of different TFSI molecules.
        2,  # Maximum denticity of a Li-TFSI complex.
    )
elif args.cmp == "Li-O":
    n_cnt_max = (
        7,  # Maximum number of different O atoms.
        4,  # Maximum number of different molecules.
        7,  # Maximum denticity of a Li-PEO or Li-TFSI complex.
    )
else:
    raise ValueError("Unknown --cmp {}".format(args.cmp))
if len(n_cnt_max) != n_cols:
    raise ValueError(
        "`len(n_cnt_max)` ({}) != `n_cols` ({})".format(len(n_cnt_max), n_cols)
    )


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="O_per_chain"
)


print("Calculating coordination numbers...")
# Average coordination number of a lithium ion.
cn = np.zeros((n_blocks, n_cols, Sims.n_sims), dtype=np.float64)
# Probability that a lithium ion has N contacts.
prob_n_cnt = np.zeros(
    (n_blocks, n_cols, max(n_cnt_max) + 2, Sims.n_sims), dtype=np.float64
)

for sim_ix, Sim in enumerate(Sims.sims):
    for block_ix, block_start in enumerate(blocks[:-1]):
        block_end = blocks[block_ix + 1]
        file_suffix = (
            analysis
            + analysis_suffix
            + "_"
            + str(block_start)
            + "-"
            + str(block_end)
            + "ns.txt.gz"
        )
        infile = leap.simulation.get_ana_file(Sim, ana_path, tool, file_suffix)
        # Read input file.
        data = np.loadtxt(infile, usecols=(0,) + cols)
        # Skip last row that contains the sum of each column.
        data = data[:-1]
        # Number of contacts.
        n_contacts = data.T[0]
        n_contacts = np.round(n_contacts, out=n_contacts).astype(np.uint16)
        # Probability that a lithium ion has N contacts.
        probabilities = data.T[1:]
        for col_ix in range(n_cols):
            # Calculate average coordination numbers.
            cn[block_ix][col_ix][sim_ix] = np.sum(
                n_contacts * probabilities[col_ix]
            )
            # Probability that a lithium ion has N contacts.
            for n_cnt, prob in zip(n_contacts, probabilities[col_ix]):
                if n_cnt >= len(prob_n_cnt[block_ix][col_ix]):
                    break
                prob_n_cnt[block_ix][col_ix][n_cnt][sim_ix] = prob


print("Calculating block averages...")
cn_av, cn_se = mdt.statistics.block_average(cn, axis=0, ddof=1)
prob_n_cnt_av, prob_n_cnt_se = mdt.statistics.block_average(
    prob_n_cnt, axis=0, ddof=1
)
del cn, prob_n_cnt
if not np.allclose(np.sum(prob_n_cnt_av, axis=1), 1, atol=5e-3):
    print(np.sum(prob_n_cnt_av, axis=1))
    raise ValueError("The sum of all coordination probabilities is not unity")


print("Creating plot(s)...")
xlim = (1, 200)
ylim_prob = (0, 1)
xlabel = r"Ether Oxygens per Chain $n_{EO}$"
legend_title = r"$r = %.2f$" % Sims.Li_O_ratios[0]

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot coordination number histograms.
    xlabels = ("Coordination Number", "Coordination Number", "Denticity")
    cmap = plt.get_cmap()
    c_vals = np.arange(Sims.n_sims)
    c_norm = max(Sims.n_sims - 1, 1)
    c_vals_normed = c_vals / c_norm
    colors = cmap(c_vals_normed)
    for col_ix, probabilities in enumerate(prob_n_cnt_av):
        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(color=colors)
        for sim_ix, Sim in enumerate(Sims.sims):
            if Sim.O_per_chain == 2:
                color = "tab:red"
                ax.plot([], [])  # Increment color cycle.
            elif Sim.O_per_chain == 3:
                color = "tab:brown"
                ax.plot([], [])  # Increment color cycle.
            elif Sim.O_per_chain == 5:
                color = "tab:orange"
                ax.plot([], [])  # Increment color cycle.
            else:
                color = None
            # ax.plot(
            #     np.arange(
            #         probabilities[:, sim_ix].size, dtype=np.uint8
            #     ),
            #     probabilities[:, sim_ix],
            #     label=r"$%d$" % Sim.O_per_chain,
            #     color=color,
            #     alpha=leap.plot.ALPHA,
            # )
            ax.errorbar(
                np.arange(probabilities[:, sim_ix].size, dtype=np.uint8),
                probabilities[:, sim_ix],
                yerr=prob_n_cnt_se[col_ix, :, sim_ix],
                label=r"$%d$" % Sim.O_per_chain,
                color=color,
                alpha=leap.plot.ALPHA,
            )
        ax.set(
            xlabel=xlabels[col_ix],
            ylabel="Probability",
            xlim=xlim_hist[col_ix],
            ylim=ylim_prob,
        )
        ax.xaxis.set_tick_params(which="minor", bottom=False, top=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend = ax.legend(
            title=(
                col_labels[col_ix] + "\n" + legend_title + "\n" + r"$n_{EO}$"
            ),
            ncol=1 + Sims.n_sims // (6 + 1),
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

    # Plot the probability that a lithium ion has N contacts as function
    # of the chain length.
    legend_title_suffixes = (" Coord. No.", " Coord. No.", " Denticity")
    markers = ("o", "|", "x", "^", "s", "p", "h", "*", "8", "v")
    if len(markers) < max(n_cnt_max):
        raise ValueError(
            "`len(markers)` ({}) < `max(n_cnt_max)`"
            " ({})".format(len(markers), max(n_cnt_max))
        )
    for col_ix, probabilities in enumerate(prob_n_cnt_av):
        cmap = plt.get_cmap()
        c_vals = np.arange(n_cnt_max[col_ix] + 1)
        c_norm = max(n_cnt_max[col_ix], 1)
        c_vals_normed = c_vals / c_norm
        colors = cmap(c_vals_normed)
        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(color=colors)
        for n_cnt, prob in enumerate(probabilities):
            if n_cnt > n_cnt_max[col_ix]:
                break
            # ax.plot(
            #     Sims.O_per_chain,
            #     prob,
            #     label=r"$%d$" % n_cnt,
            #     marker=markers[n_cnt],
            #     alpha=leap.plot.ALPHA,
            # )
            ax.errorbar(
                Sims.O_per_chain,
                prob,
                yerr=prob_n_cnt_se[col_ix, n_cnt],
                label=r"$%d$" % n_cnt,
                marker=markers[n_cnt],
                alpha=leap.plot.ALPHA,
            )
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlabel=xlabel, ylabel="Probability", xlim=xlim, ylim=ylim_prob)
        if args.cmp == "Li-OE" and col_ix in (0, 2):
            legend_loc = "best"
            n_legend_cols = 4
        elif args.cmp == "Li-O" and col_ix in (0, 2):
            legend_loc = "center right"
            n_legend_cols = 4
        else:
            legend_loc = "best"
            n_legend_cols = 3
        legend = ax.legend(
            loc=legend_loc,
            title=(
                legend_title
                + "\n"
                + col_labels[col_ix]
                + legend_title_suffixes[col_ix]
            ),
            ncol=n_legend_cols,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

    # Plot average coordination numbers.
    markers = ("^", "v")
    fig, ax = plt.subplots(clear=True)
    for col_ix in [cols.index(1), cols.index(2)]:
        ax.errorbar(
            Sims.O_per_chain,
            cn_av[col_ix],
            yerr=cn_se[col_ix],
            label=col_labels[col_ix],
            marker=markers[col_ix],
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel=xlabel, ylabel="Coordination Number", xlim=xlim, ylim=ylim_cn
    )
    equalize_yticks(ax)
    ax.legend(title=legend_title)
    pdf.savefig()
    plt.close()

    # Plot average denticities.
    fig, ax = plt.subplots(clear=True)
    for col_ix in [cols.index(5)]:
        ax.errorbar(
            Sims.O_per_chain,
            cn_av[col_ix],
            yerr=cn_se[col_ix],
            label=col_labels[col_ix],
            marker="o",
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel=xlabel,
        ylabel="Average Denticity",
        xlim=xlim,
        ylim=ylim_denticity,
    )
    equalize_yticks(ax)
    ax.legend(title=legend_title)
    pdf.savefig()
    plt.close()

    # Plot the standard error of the probability that a lithium ion has
    # N contacts as function of the chain length.
    legend_title_suffixes = (" Coord. No.", " Coord. No.", " Denticity")
    markers = ("o", "|", "x", "^", "s", "p", "h", "*", "8", "v")
    if len(markers) < max(n_cnt_max):
        raise ValueError(
            "`len(markers)` ({}) < `max(n_cnt_max)`"
            " ({})".format(len(markers), max(n_cnt_max))
        )
    for col_ix, prob_errors in enumerate(prob_n_cnt_se):
        cmap = plt.get_cmap()
        c_vals = np.arange(n_cnt_max[col_ix] + 1)
        c_norm = max(n_cnt_max[col_ix], 1)
        c_vals_normed = c_vals / c_norm
        colors = cmap(c_vals_normed)
        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(color=colors)
        for n_cnt, prob_error in enumerate(prob_errors):
            if n_cnt > n_cnt_max[col_ix]:
                break
            ax.plot(
                Sims.O_per_chain,
                prob_error,
                label=r"$%d$" % n_cnt,
                marker=markers[n_cnt],
                alpha=leap.plot.ALPHA,
            )
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlabel=xlabel, ylabel="Std. Err. of Prob.", xlim=xlim)
        legend = ax.legend(
            loc="best",
            title=(
                legend_title
                + "\n"
                + col_labels[col_ix]
                + legend_title_suffixes[col_ix]
            ),
            ncol=n_legend_cols,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

    # Plot the standard error of the average coordination numbers.
    markers = ("^", "v")
    fig, ax = plt.subplots(clear=True)
    for col_ix in [cols.index(1), cols.index(2)]:
        ax.plot(
            Sims.O_per_chain,
            cn_se[col_ix],
            label=col_labels[col_ix],
            marker=markers[col_ix],
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel="Std. Err. of CN", xlim=xlim)
    equalize_yticks(ax)
    ax.legend(title=legend_title)
    pdf.savefig()
    plt.close()

    # Plot the standard error of the average denticities.
    fig, ax = plt.subplots(clear=True)
    for col_ix in [cols.index(5)]:
        ax.plot(
            Sims.O_per_chain,
            cn_se[col_ix],
            label=col_labels[col_ix],
            marker="o",
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel="Std. Err. of Denticity", xlim=xlim)
    equalize_yticks(ax)
    ax.legend(title=legend_title)
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
