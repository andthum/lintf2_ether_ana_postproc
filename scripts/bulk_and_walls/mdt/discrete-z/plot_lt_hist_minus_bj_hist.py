#!/usr/bin/env python3


"""
Plot the difference of the sum of the lifetime histograms of adjacent
bins obtained from the uncensored count method and the back-jump
histogram for each bin for a single simulation.

Additionally plot the quotient of the back-jump histogram and the sum of
the adjacent lifetime histograms.
"""


# Standard libraries
import argparse

# Third-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot difference of the sum of the lifetime histograms of adjacent"
        " states obtained from the uncensored count method and the back-jump"
        " histogram for each bin for a single simulation."
    )
)
parser.add_argument(
    "--system",
    type=str,
    required=True,
    help="Name of the simulated system, e.g. lintf2_g1_20-1_gra_q1_sc80.",
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
    required=False,
    default="Li",
    choices=("Li",),
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--intermittency",
    type=int,
    required=False,
    default=0,
    help=(
        "Maximum number of frames a compound is allowed to leave its state"
        " while still being considered to be in this state provided that it"
        " returns to this state after the given number of frames."
    ),
)
args = parser.parse_args()

# Use the uncensored lifetime histogram.
uncensored = True
# Use the continuous back-jump probability.
continuous = True

analysis = "discrete-z"  # Analysis name.
tool = "mdt"  # Analysis software.
outfile = (
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis
    + "_"
    + args.cmp
    + "_lt_hist"
)
if uncensored:
    outfile += "_uncensored"
if args.intermittency > 0:
    outfile += "_intermittency_%d" % args.intermittency
outfile += "_minus_bj_hist"
if continuous:
    outfile += "_continuous"
outfile += ".pdf"

# Time conversion factor to convert from trajectory steps to ns.
time_conv = 2e-3


print("Creating Simulation instance(s)...")
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    top_path = "q%g" % surfq
else:
    surfq = None
    top_path = "bulk"
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, top_path)


print("Reading data...")
file_suffix = analysis + "_" + args.cmp + "_dtrj.npz"
infile = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
dtrj = mdt.fh.load_dtrj(infile)
n_frames = dtrj.shape[-1]


print("Calculating back-jump histograms")
bj_probs, norms = mdt.dtrj.back_jump_prob_discrete(
    dtrj, dtrj, continuous=continuous, return_norm=True, verbose=True
)
n_states = bj_probs.shape[0]
if bj_probs.shape[1] + 1 != n_frames:
    raise ValueError(
        "`bj_probs.shape[1] + 1` ({}) != `n_frames`"
        " ({})".format(bj_probs.shape[1] + 1, n_frames)
    )
# Convert back-jump probabilities to the absolute number of compounds
# that return to their initial state after a given lag time.
bj_counts = bj_probs * norms
del bj_probs, norms
if not np.allclose(
    bj_counts, np.round(bj_counts), equal_nan=True, rtol=0, atol=1e-9
):
    raise ValueError("`bj_counts` is not an integer array")
bj_counts = np.round(bj_counts, out=bj_counts)
# Discard lag time 0, because a corresponding lifetime of 0 does not
# exist and the back-jump count at lag time 0 is zero anyway.
bj_counts = bj_counts[:, 1:]
# Create array of corresponding lag times in trajectory steps.
times = np.arange(1, n_frames, dtype=np.uint32)

# Bins for re-binning the back-jump counts.
# Linear bins.
# step = 1
# bins = np.arange(1, n_frames, step, dtype=np.float64)
# Logarithmic bins.
stop = int(np.ceil(np.log2(n_frames)))
bins = np.logspace(0, stop, stop + 1, base=2, dtype=np.float64)
unit_bins = True if np.allclose(np.diff(bins), 1) else False
bins -= 0.5
bin_mids = bins[1:] - np.diff(bins) / 2
if not unit_bins:
    # Combine multiple lag times in one bin.
    # Binning is done in trajectory steps.
    if np.any(times < bins[0]) or np.any(times > bins[-1]):
        raise ValueError(
            "At least one lag time lies outside the binned region"
        )
    bin_ix = np.digitize(times, bins)
    slices = np.flatnonzero(np.diff(bin_ix, prepend=0) != 0)
    bj_counts = np.add.reduceat(bj_counts, slices, axis=-1)
    del bin_ix, slices
del times


if args.intermittency > 0:
    print("Correcting for intermittency...")
    dtrj = mdt.dyn.correct_intermittency(
        dtrj.T, args.intermittency, inplace=True, verbose=True
    )
    dtrj = dtrj.T


print("Calculating lifetime histograms...")
lts_per_state, states = mdt.dtrj.lifetimes_per_state(
    dtrj, uncensored=uncensored, return_states=True
)
del dtrj
if len(states) != n_states:
    raise ValueError(
        "`len(states)` ({}) != `n_states` ({})".format(len(states), n_states)
    )
# Use `n_states + 2` for the size of the first dimension of `hists` to
# add a histogram containing only zeros at the beginning and end of
# `hists`.  This is important for the calculation of `hists_sum`.
hists = np.zeros((n_states + 2, len(bins) - 1), dtype=np.uint32)
for state_ix, lts_state in enumerate(lts_per_state, start=1):
    if np.any(lts_state < bins[0]) or np.any(lts_state > bins[-1]):
        raise ValueError(
            "At least one lifetime lies outside the binned region"
        )
    hists[state_ix], _bins = np.histogram(lts_state, bins=bins, density=False)
    if not np.allclose(_bins, bins, rtol=0):
        raise ValueError("`_bins` != `bins`.  This should not have happened")
    if np.sum(hists[state_ix]) != len(lts_state):
        raise ValueError(
            "The sum of all histogram values ({}) is not equal to the number"
            " of samples ({})".format(np.sum(hists[state_ix]), len(lts_state))
        )
if np.any(hists[0] != 0):
    print("hists[0] =", hists[0])
    raise ValueError("np.any(hists[0] != 0)")
if np.any(hists[-1] != 0):
    print("hists[-1] =", hists[-1])
    raise ValueError("np.any(hists[-1] != 0)")
del lts_per_state, lts_state, _bins

# Let each histogram be the sum of its preceding and following
# histogram.  Only the summed lifetime histograms can be compared to the
# back-jump counts, because the maximum possible number of back jumps at
# a given lag time is the number compounds in the neighboring states
# that have a lifetime equal to the given lag time.
hists_sum = np.zeros((n_states, len(bins) - 1), dtype=np.uint32)
for state_ix in range(n_states):
    hists_sum[state_ix] = hists[state_ix] + hists[state_ix + 2]
if not np.array_equal(hists_sum[0], hists[2]):
    print("hists_sum[0] =", hists_sum[0])
    print("hists[2]     =", hists[2])
    raise ValueError("`hists_sum[0]` != `hists[2]`")
if not np.array_equal(hists_sum[-1], hists[-3]):
    print("hists_sum[-1] =", hists_sum[-1])
    print("hists[-3]     =", hists[-3])
    raise ValueError("`hists_sum[-1]` != `hists[-3]`")
del hists


print("Fitting power law...")
# Select a state from the center of the simulation box.
state_ix_fit = n_states // 2
fit_start_ix = 0
_, fit_stop_ix = mdt.nph.find_nearest(bin_mids, 50, return_index=True)
fit_stop_ix += 1

# Fit back-jump counts.
bj_count_fit = bj_counts[state_ix_fit][fit_start_ix:fit_stop_ix]
bin_mids_fit = bin_mids[fit_start_ix:fit_stop_ix]  # This is a view!
popt_bj_count, pcov_bj_count = curve_fit(
    f=leap.misc.straight_line,
    xdata=np.log(bin_mids_fit),
    ydata=np.log(bj_count_fit),
    p0=(-1.5, np.log(bj_count_fit[0])),
)
# perr_bj_count = np.sqrt(np.diag(pcov_bj_count))
bj_count_fit = leap.misc.power_law(
    bin_mids_fit, popt_bj_count[0], np.exp(popt_bj_count[1])
)

# Fit lifetime histogram.
if args.intermittency == 0:
    hist_fit = hists_sum[state_ix_fit][fit_start_ix:fit_stop_ix]
    bin_mids_fit = bin_mids[fit_start_ix:fit_stop_ix]  # This is a view!
    popt_lt_hist, pcov_lt_hist = curve_fit(
        f=leap.misc.straight_line,
        xdata=np.log(bin_mids_fit),
        ydata=np.log(hist_fit),
        p0=(-1.5, np.log(hist_fit[0])),
    )
    # perr_lt_hist = np.sqrt(np.diag(pcov_lt_hist))
    hist_fit = leap.misc.power_law(
        bin_mids_fit, popt_lt_hist[0], np.exp(popt_lt_hist[1])
    )


print("Creating plots...")
bins *= time_conv
bin_mids *= time_conv
# Don't do the following, because `bin_mids_fit` is a view of `bin_mids`
# bin_mids_fit *= time_conv

xmin_xlin = 0
xmin_xlog = 1 * time_conv
xmax_ylin = 200
xmax_ylog = (n_frames - 1) * time_conv
ymin_ylin = 0
ymin_ylog = 0.5
ymax_ylin = 3e5
ymax_ylog = 5e5

if surfq is None:
    legend_title = ""
else:
    legend_title = r"$\sigma_s = \pm %.2f$ $e$/nm$^2$" % surfq + "\n"
legend_title = (
    legend_title
    + r"$n_{EO} = %d$, " % Sim.O_per_chain
    + r"$r = %.4f$" % Sim.Li_O_ratio
    + "\n"
    + "Bin Number"
)
n_legend_cols = 1 + n_states // (5 + 1)

cmap = plt.get_cmap()
c_vals = np.arange(n_states)
c_norm = max(1, n_states - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot the sum of the neighboring lifetime histograms.
    xlabel = "Lifetime / ns"
    ylabel = "Sum of Adjacent LT Hists"
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for state_ix, state_num in enumerate(states):
        if not unit_bins:
            ax.stairs(
                hists_sum[state_ix],
                bins,
                fill=False,
                label=r"$%d$" % (state_num + 1),
                alpha=leap.plot.ALPHA,
                rasterized=False,
            )
        else:
            ax.plot(
                bin_mids,
                hists_sum[state_ix],
                label=r"$%d$" % (state_num + 1),
                alpha=leap.plot.ALPHA,
                rasterized=False,
            )
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(xmin_xlin, xmax_ylin),
        ylim=(ymin_ylin, ymax_ylin),
    )
    legend = ax.legend(
        title=legend_title,
        loc="upper right",
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()

    # Log scale x.
    ax.set_xlim(xmin_xlog, xmax_ylin)
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()

    # Log scale y.
    ax.relim()
    ax.autoscale()
    ax.set_xscale("linear")
    ax.set_xlim(xmin_xlin, xmax_ylog)
    ax.set_ylim(ymin_ylog, ymax_ylog)
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()

    # Log scale xy.
    if args.intermittency == 0:
        ax.plot(
            bin_mids_fit,
            hist_fit,
            color="black",
            linestyle="dashed",
            alpha=leap.plot.ALPHA,
        )
        ax.text(
            bin_mids_fit[-1],
            hist_fit[-1] * 1.2,
            r"$\propto t^{%.2f}$" % popt_lt_hist[0],
            rotation=np.rad2deg(np.arctan(popt_lt_hist[0])) / 1.5,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize="small",
        )
    ax.set_xlim(xmin_xlog, xmax_ylog)
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot back-jump histograms.
    xlabel = "Lag Time / ns"
    ylabel = "Back-Jump Histogram"
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for state_ix, state_num in enumerate(states):
        if not unit_bins:
            ax.stairs(
                bj_counts[state_ix],
                bins,
                fill=False,
                label=r"$%d$" % (state_num + 1),
                alpha=leap.plot.ALPHA,
                rasterized=False,
            )
        else:
            ax.plot(
                bin_mids,
                bj_counts[state_ix],
                label=r"$%d$" % (state_num + 1),
                alpha=leap.plot.ALPHA,
                rasterized=True,
            )
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(xmin_xlin, xmax_ylin),
        ylim=(ymin_ylin, ymax_ylin),
    )
    legend = ax.legend(
        title=legend_title,
        loc="upper right",
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()

    # Log scale x.
    ax.set_xlim(xmin_xlog, xmax_ylin)
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()

    # Log scale y.
    ax.relim()
    ax.autoscale()
    ax.set_xscale("linear")
    ax.set_xlim(xmin_xlin, xmax_ylog)
    ax.set_ylim(ymin_ylog, ymax_ylog)
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()

    # Log scale xy.
    ax.plot(
        bin_mids_fit,
        bj_count_fit,
        color="black",
        linestyle="dashed",
        alpha=leap.plot.ALPHA,
    )
    ax.text(
        bin_mids_fit[-1],
        bj_count_fit[-1] * 1.2,
        r"$\propto t^{%.2f}$" % popt_bj_count[0],
        rotation=np.rad2deg(np.arctan(popt_bj_count[0])) / 1.5,
        rotation_mode="anchor",
        transform_rotates_text=False,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize="small",
    )
    ax.set_xlim(xmin_xlog, xmax_ylog)
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot the difference of the sum of the neighboring lifetime
    # histograms and the back-jump histogram.
    ydata = hists_sum - bj_counts
    xlabel = "Time / ns"
    ylabel = r"($\Sigma$ Adj. LT Hists) - BJ Hist"
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for state_ix, state_num in enumerate(states):
        if not unit_bins:
            ax.stairs(
                ydata[state_ix],
                bins,
                fill=False,
                label=r"$%d$" % (state_num + 1),
                alpha=leap.plot.ALPHA,
                rasterized=False,
            )
        else:
            ax.plot(
                bin_mids,
                ydata[state_ix],
                label=r"$%d$" % (state_num + 1),
                alpha=leap.plot.ALPHA,
                rasterized=True,
            )
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(xmin_xlin, xmax_ylin),
        ylim=(ymin_ylin, ymax_ylin),
    )
    legend = ax.legend(
        title=legend_title,
        loc="upper right",
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    # Log scale x.
    ax.set_xlim(xmin_xlog, xmax_ylin)
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale y.
    ax.relim()
    ax.autoscale()
    ax.set_xscale("linear")
    ax.set_xlim(xmin_xlin, xmax_ylog)
    ax.set_ylim(ymin_ylog, ymax_ylog)
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale xy.
    ax.set_xlim(xmin_xlog, xmax_ylog)
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot the quotient of the back-jump histogram and the sum of the
    # neighboring lifetime histograms.
    ydata = bj_counts / hists_sum
    xlabel = "Time / ns"
    ylabel = r"BJ Hist / ($\Sigma$ Adj. LT Hists)"
    ymin_ylog = np.min(ydata[ydata > 0]) / 2
    ymax_ylin = 2
    ymax_ylog = 2
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for state_ix, state_num in enumerate(states):
        if not unit_bins:
            ax.stairs(
                ydata[state_ix],
                bins,
                fill=False,
                label=r"$%d$" % (state_num + 1),
                alpha=leap.plot.ALPHA,
                rasterized=False,
            )
        else:
            ax.plot(
                bin_mids,
                ydata[state_ix],
                label=r"$%d$" % (state_num + 1),
                alpha=leap.plot.ALPHA,
                rasterized=True,
            )
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(xmin_xlin, xmax_ylin),
        ylim=(ymin_ylin, ymax_ylin),
    )
    legend = ax.legend(
        title=legend_title,
        loc="upper right",
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    # Log scale x.
    ax.set_xlim(xmin_xlog, xmax_ylin)
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale y.
    ax.relim()
    ax.autoscale()
    ax.set_xscale("linear")
    ax.set_xlim(xmin_xlin, xmax_ylog)
    ax.set_ylim(ymin_ylog, ymax_ylog)
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale xy.
    ax.set_xlim(xmin_xlog, xmax_ylog)
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    legend = ax.legend(
        title=legend_title,
        loc="lower left",
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
