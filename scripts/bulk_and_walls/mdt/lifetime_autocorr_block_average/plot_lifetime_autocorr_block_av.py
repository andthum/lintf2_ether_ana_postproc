#!/usr/bin/env python3


"""
Plot the lifetime autocorrelation function for two given compounds as
function of the trajectory blocks used for block averaging.
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
from matplotlib.ticker import MaxNLocator

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the lifetime autocorrelation function for two given compounds as"
        " function of the trajectory blocks used for block averaging."
    ),
)
parser.add_argument(
    "--system",
    type=str,
    required=True,
    help="Name of the simulated system, e.g. lintf2_g1_20-1_sc80.",
)
parser.add_argument(
    "--settings",
    type=str,
    required=False,
    default="pr_nvt423_nh",
    help=(
        "String describing the used simulation settings.  Default:"
        " %(default)s."
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

analysis = "lifetime_autocorr"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
analysis_dir = analysis + "_block_average"
ana_path = os.path.join(analysis_dir, analysis + analysis_suffix)
tool = "mdt"  # Analysis software.
outfile_base = (  # Output file name.
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis_dir
    + "_integration_stop"
    + analysis_suffix
)
outfile_txt = outfile_base + ".txt.gz"
outfile_pdf = outfile_base + ".pdf"

# Blocks used for block averaging
blocks = list(range(0, 1200, 200))
n_blocks = len(blocks) - 1

cols = (  # Columns to read from the input file(s).
    0,  # Lag times in [ps].
    1,  # Autocorrelation function.
)

# Only calculate the lifetime by directly integrating the ACF if the ACF
# decayed below the given threshold.
int_thresh = 0.01


print("Creating Simulation instance(s)...")
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, path_key="bulk")


print("Reading data...")
# Lifetime autocorrelation functions for each trajectory block.
times = None
acfs = [None for i in range(n_blocks)]
lifetimes = np.full(n_blocks, np.nan, dtype=np.float64)
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
    times_sim, acf = np.loadtxt(infile, usecols=cols, unpack=True)
    times_sim *= 1e-3  # ps -> ns
    acfs[block_ix] = acf
    if times is None:
        times = np.copy(times_sim)
    elif times_sim.shape != times.shape:
        raise ValueError(
            "`times_sim.shape` ({}) != `times.shape` ({})".format(
                times_sim.shape, times.shape
            )
        )
    elif not np.allclose(times_sim, times, atol=1e-09, rtol=0):
        raise ValueError(
            "The times in `times_sim` differ from those in `times`"
        )
    if np.any(acf <= int_thresh):
        # Only calculate the lifetime by numerical integration if the
        # ACF decays below the given threshold.
        # Only calculate the ACF until its global minimum, a potential
        # increase of the ACF after the minimum is likely a finite size
        # artifact and should therefore be discarded.
        stop = np.flatnonzero(acf <= int_thresh)[0]
        lifetimes[block_ix] = leap.lifetimes.raw_moment_integrate(
            sf=acf[:stop], x=times_sim[:stop]
        )
del times_sim, acf

# Lifetime autocorrelation function over the full trajectory.
file_suffix = analysis + analysis_suffix + ".txt.gz"
infile = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
times_total, acf_total = np.loadtxt(infile, usecols=cols, unpack=True)
times_total *= 1e-3  # ps -> ns
if np.any(acf_total <= int_thresh):
    # Only calculate the lifetime by numerical integration if the ACF
    # decays below the given threshold.
    # Only calculate the ACF until its global minimum, a potential
    # increase of the ACF after the minimum is likely a finite size
    # artifact and should therefore be discarded.
    stop = np.flatnonzero(acf_total <= int_thresh)[0]
    lifetime_total = leap.lifetimes.raw_moment_integrate(
        sf=acf_total, x=times_total
    )
else:
    lifetime_total = np.nan


print("Calculating block averages...")
acfs = np.asarray(acfs)
acf_av, acf_se = mdt.statistics.block_average(acfs, axis=0, ddof=1)
n_lifetime_measurements = np.count_nonzero(np.isfinite(lifetimes))
if n_lifetime_measurements == 1:
    lifetime_av = lifetimes[np.isfinite(lifetimes)][0]
    lifetime_se = np.nan
elif n_lifetime_measurements > 1:
    lifetime_av = np.nanmean(lifetimes)
    lifetime_se = np.nanstd(lifetimes, ddof=1) / np.sqrt(
        n_lifetime_measurements
    )
else:
    lifetime_av, lifetime_se = np.nan, np.nan

# Calculate lifetime from average autocorrelation function.
if np.any(acf_av <= int_thresh):
    # Only calculate the lifetime by numerical integration if the ACF
    # decays below the given threshold.
    # Only calculate the ACF until its global minimum, a potential
    # increase of the ACF after the minimum is likely a finite size
    # artifact and should therefore be discarded.
    stop = np.flatnonzero(acf_av <= int_thresh)[0]
    lifetime_acf_av = leap.lifetimes.raw_moment_integrate(sf=acf_av, x=times)
else:
    lifetime_acf_av = np.nan


print("Creating plot(s)...")
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
    + r"$n_{EO} = %d$" % Sim.O_per_chain
    + ", "
    + r"$r = %.4f$" % Sim.Li_O_ratio
)

cmap = plt.get_cmap()
c_vals = np.arange(n_blocks)
c_norm = max(n_blocks - 1, 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile_pdf)
with PdfPages(outfile_pdf) as pdf:
    # Plot autocorrelation functions.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    # Autocorrelation function for each trajectory block.
    for block_ix, block_start in enumerate(blocks[:-1]):
        block_end = blocks[block_ix + 1]
        ax.plot(
            times,
            acfs[block_ix],
            label="$" + str(block_start) + "-" + str(block_end) + "$ ns",
            alpha=leap.plot.ALPHA,
        )
    # Average autocorrelation function.
    ax.plot(
        times,
        acf_av,
        label="Average",
        color="red",
        linestyle="dashed",
        alpha=leap.plot.ALPHA,
    )
    # ax.fill_between(
    #     times,
    #     y1=acf_av + acf_se,
    #     y2=acf_av - acf_se,
    #     color="red",
    #     edgecolor=None,
    #     alpha=leap.plot.ALPHA / 2,
    #     rasterized=True,
    # )
    # Autocorrelation function for full trajectory.
    ax.plot(
        times_total,
        acf_total,
        label="Full trajectory",
        color="orange",
        linestyle="dotted",
        alpha=leap.plot.ALPHA,
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    legend = ax.legend(
        title=legend_title_base,
        loc="best",
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot lifetimes as function of block.
    xlim = np.asarray([0.5, n_blocks + 0.5])
    fig, ax = plt.subplots(clear=True)
    ax.plot(np.arange(1, n_blocks + 1), lifetimes, color=colors[0], marker="o")
    if np.isfinite(lifetime_av):
        ax.axhline(
            lifetime_av,
            label="Mean",
            color=colors[1],
            linestyle="dashed",
            alpha=leap.plot.ALPHA,
        )
        ax.fill_between(
            xlim,
            y1=np.full(xlim.shape, lifetime_av + lifetime_se),
            y2=np.full(xlim.shape, lifetime_av - lifetime_se),
            color=colors[1],
            edgecolor=None,
            alpha=leap.plot.ALPHA / 2,
        )
    if np.isfinite(lifetime_acf_av):
        ax.axhline(
            lifetime_acf_av,
            label="Average ACF",
            color="red",
            linestyle="dashed",
            alpha=leap.plot.ALPHA,
        )
    if np.isfinite(lifetime_total):
        ax.axhline(
            lifetime_total,
            label="Full ACF",
            color="orange",
            linestyle="dotted",
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Trajectory Block", ylabel="Correlation Time / ns", xlim=xlim
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_tick_params(which="minor", bottom=False, top=False)
    legend = ax.legend(
        title=(
            legend_title_base
            + "\n"
            + r"Mean:      ${:.4f}$ ns".format(lifetime_av)
            + "\n"
            + r"Std. Err.: ${:.4f}$ ns".format(lifetime_se)
        ),
        loc="best",
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile_pdf))


print("Creating output file(s)...")
header = (
    "Coordination correlation times for each trajectory block.\n"
    + "\n"
    + "Average coordination correlation times are calculated by numerical\n"
    + "integration of the lifetime autocorrelation function.\n"
    + "\n"
    + "No. of       blocks:   {:d}\n".format(n_blocks)
    + "No. of valid blocks:   {:d}\n".format(n_lifetime_measurements)
    + "Mean correlation time: {:.9e} ns\n".format(lifetime_av)
    + "Standard error:        {:.9e} ns\n".format(lifetime_se)
    + "\n"
    + "Corr. time from average ACF: {:9e} ns\n".format(lifetime_acf_av)
    + "Corr. time from full    ACF: {:9e} ns\n".format(lifetime_total)
    + "\n"
    + "The columns contain:\n"
    + " 1 Block number\n"
    + " 2 Block start / ns\n"
    + " 3 Block end / ns\n"
    + " 4 {:s} correlation time / ns\n".format(args.cmp)
)
data = np.column_stack(
    [np.arange(1, n_blocks + 1), blocks[:-1], blocks[1:], lifetimes]
)
leap.io_handler.savetxt(outfile_txt, data, header=header)
print("Created {}".format(outfile_txt))
print("Done")
