#!/usr/bin/env python3


"""
Plot the lifetime autocorrelation function for a given set of compound
pairs as function of the PEO chain length.
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
        "Plot the lifetime autocorrelation function for a given set of"
        " compound pairs as function of the PEO chain length."
    ),
)
parser.add_argument(
    "--cmps",
    type=str,
    required=False,
    default=["Li-OE", "Li-OBT"],
    nargs="+",
    help=(
        "The compound pairs for which to plot the lifetime autocorrelation"
        " function."
    ),
)
args = parser.parse_args()
if len(args.cmps) == 0:
    raise ValueError(
        "No compound pair specified with --cmps ({})".format(args.cmps)
    )

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "lifetime_autocorr"  # Analysis name.
tool = "mdt"  # Analysis software.
outfile_base = (  # Output file name.
    settings
    + "_lintf2_peoN_20-1_sc80_"
    + analysis
    + "_combined_"
    + "_".join(args.cmps)
)
outfile_txt = outfile_base + ".txt.gz"
outfile_pdf = outfile_base + ".pdf"

cols = (  # Columns to read from the input file(s).
    0,  # Lag times in [ps].
    1,  # Autocorrelation function.
)

# Only calculate the lifetime by directly integrating the ACF if the ACF
# decayed below the given threshold.
int_thresh = 0.01

print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="O_per_chain"
)


print("Reading data and creating plot(s)...")
infiles = [None for cmp in args.cmps]
for cmp_ix, cmp in enumerate(args.cmps):
    file_suffix = analysis + "_" + cmp + ".txt.gz"
    infiles[cmp_ix] = leap.simulation.get_ana_files(
        Sims, analysis, tool, file_suffix
    )
    n_infiles = len(infiles[cmp_ix])
    if n_infiles != Sims.n_sims:
        raise ValueError(
            "The number of input files ({}) for the compound pair {} is not"
            " equal to the number of simulations"
            " ({})".format(n_infiles, cmp, Sims.n_sims)
        )
lifetimes = np.full((len(args.cmps), Sims.n_sims), np.nan, dtype=np.float32)

xlabel = "Lag Time / ns"
ylabel = "Autocorrelation Function"
xlim = (2e-3, 1e3)
ylim = (0, 1)

legend_title_base = r"$r = %.2f$" % Sims.Li_O_ratios[0]
legend_title = legend_title_base + "\n" + r"$n_{EO}$"
n_legend_cols = 1 + Sims.n_sims // (6 + 1)

cmap = plt.get_cmap()
c_vals = np.arange(Sims.n_sims)
c_norm = max(Sims.n_sims - 1, 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile_pdf)
with PdfPages(outfile_pdf) as pdf:
    # Plot autocorrelation functions.
    for cmp_ix, cmp in enumerate(args.cmps):
        cmp1, cmp2 = cmp.split("-")
        cmp_label = (
            r"$"
            + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
            + "-"
            + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp2]
            + r"$"
        )
        legend_title_cmp = cmp_label + "\n" + legend_title
        if cmp in ("Li-OE", "Li-ether"):
            legend_loc = "lower left"
        else:
            legend_loc = "upper right"

        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(color=colors)
        for sim_ix, Sim in enumerate(Sims.sims):
            times, acf = np.loadtxt(
                infiles[cmp_ix][sim_ix], usecols=cols, unpack=True
            )
            times *= 1e-3  # ps -> ns
            if np.any(acf <= int_thresh):
                # Only calculate the lifetime by numerical integration
                # if the ACF decays below the given threshold.
                # Only calculate the ACF until its global minimum, a
                # potential increase of the ACF after the minimum is
                # likely a finite size artifact and should therefore be
                # discarded.
                stop = np.nanargmin(acf) + 1
                lifetimes[cmp_ix][
                    sim_ix
                ] = leap.lifetimes.raw_moment_integrate(
                    sf=acf[:stop], x=times[:stop]
                )
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
            if cmp == "Li-OE" and Sim.O_per_chain == 4:
                linestyle = "dashed"
            elif cmp == "Li-OE" and Sim.O_per_chain == 8:
                linestyle = "dashed"
            else:
                linestyle = None
            ax.plot(
                times,
                acf,
                label=r"%d" % Sim.O_per_chain,
                linestyle=linestyle,
                color=color,
                alpha=leap.plot.ALPHA,
            )
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
        legend = ax.legend(
            title=legend_title_cmp,
            loc=legend_loc,
            ncol=n_legend_cols,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

    # Plot lifetimes.
    xlabel = r"Ether Oxygens per Chain $n_{EO}$"
    xlim = (1, 200)
    markers = ("^", "v", "s", "D")
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, cmp in enumerate(args.cmps):
        cmp1, cmp2 = cmp.split("-")
        cmp_label = (
            r"$"
            + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
            + "-"
            + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp2]
            + r"$"
        )
        ax.plot(
            Sims.O_per_chain,
            lifetimes[cmp_ix],
            label=cmp_label,
            marker=markers[cmp_ix],
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel="Relaxation Time / ns", xlim=xlim)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    legend = ax.legend(title=legend_title_base)
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
    print("Created {}".format(outfile_pdf))


print("Creating output file(s)...")
header = (
    "Coordination relaxation times.\n"
    + "\n"
    + "Average coordination relaxation times are calculated by numerical\n"
    + "integration of the lifetime autocorrelation function."
    + "\n\n"
    + "The columns contain:\n"
    + " 1 Number of ether oxygens per PEO chain\n"
)
for col, cmp in enumerate(args.cmps, start=2):
    header += " {:d} {:s} relaxation time / ns\n".format(col, cmp)
data = np.column_stack([Sims.O_per_chain, lifetimes.T])
leap.io_handler.savetxt(outfile_txt, data, header=header)
print("Created {}".format(outfile_txt))
print("Done")
