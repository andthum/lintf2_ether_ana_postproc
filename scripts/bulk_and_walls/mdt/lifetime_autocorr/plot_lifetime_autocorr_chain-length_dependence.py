#!/usr/bin/env python3


"""
Plot the lifetime autocorrelation function for two given compounds as
function of the PEO chain length.
"""


# Standard libraries
import argparse

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the lifetime autocorrelation function for two given compounds as"
        " function of the PEO chain length."
    ),
)
parser.add_argument(
    "--cmp",
    type=str,
    required=True,
    choices=("Li-OE", "Li-OBT", "Li-ether", "Li-NTf2"),
    help="Compounds for which to plot the lifetime autocorrelation function.",
)
args = parser.parse_args()
cmp1, cmp2 = args.cmp.split("-")

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "lifetime_autocorr"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    settings + "_lintf2_peoN_20-1_sc80_" + analysis + analysis_suffix + ".pdf"
)

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
file_suffix = analysis + analysis_suffix + ".txt.gz"
infiles = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
n_infiles = len(infiles)
lifetimes = np.full(n_infiles, np.nan, dtype=np.float32)

xlabel = "Lag Time / ns"
ylabel = "Autocorrelation Function"
xlim = (2e-3, 1e3)
ylim = (0, 1)

legend_title_base = (
    r"$"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
    + "-"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp2]
    + r"$"
    + "\n"
    + r"$r = %.2f$" % Sims.Li_O_ratios[0]
)
legend_title = legend_title_base + "\n" + r"$n_{EO}$"
n_legend_cols = 1 + n_infiles // (6 + 1)
if args.cmp in ("Li-OE", "Li-ether"):
    legend_loc = "lower left"
else:
    legend_loc = "upper right"

cmap = plt.get_cmap()
c_vals = np.arange(n_infiles)
c_norm = max(n_infiles - 1, 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot autocorrelation functions.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        times, acf = np.loadtxt(infiles[sim_ix], usecols=cols, unpack=True)
        times *= 1e-3  # ps -> ns
        if np.any(acf <= int_thresh):
            # Only calculate the lifetime by numerical integration if
            # the ACF decays below the given threshold.
            # Only calculate the ACF until its global minimum, a
            # potential increase of the ACF after the minimum is likely
            # a finite size artifact and should therefore be discarded.
            stop = np.nanargmin(acf) + 1
            lifetimes[sim_ix] = leap.lifetimes.raw_moment_integrate(
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
        if args.cmp == "Li-OE" and Sim.O_per_chain == 4:
            linestyle = "dashed"
        elif args.cmp == "Li-OE" and Sim.O_per_chain == 8:
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
        if args.cmp == "Li-OE" and sim_ix == Sims.n_sims - 1:
            start, stop = leap.misc.find_nearest(times, [1e0, 2e1])
            times_fit = times[start:stop]
            popt, pcov = curve_fit(
                f=leap.misc.straight_line,
                xdata=np.log(times_fit),
                ydata=acf[start:stop],
                p0=(0.1, 2),
            )
            acf_fit = leap.misc.straight_line(np.log(times_fit), *popt)
            ax.plot(
                times_fit,
                acf_fit,
                color="black",
                linestyle="dashed",
                alpha=leap.plot.ALPHA,
            )
            ax.text(
                times_fit[0],
                acf_fit[0] + 0.01,
                r"$%.2f \ln(t) + %.2f$" % tuple(popt),
                rotation=-50,
                rotation_mode="anchor",
                transform_rotates_text=False,
                horizontalalignment="left",
                verticalalignment="bottom",
                fontsize="small",
            )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    legend = ax.legend(
        title=legend_title,
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
    fig, ax = plt.subplots(clear=True)
    ax.plot(Sims.O_per_chain, lifetimes, marker="o")
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

print("Created {}".format(outfile))
print("Done")
