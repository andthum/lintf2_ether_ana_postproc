#!/usr/bin/env python3


"""
Plot the lifetime histogram obtained from the count method for each bin
for a single simulation.
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
        "Plot the lifetime histogram obtained from the count method for each"
        " bin for a single simulation."
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
    "--uncensored",
    required=False,
    default=False,
    action="store_true",
    help=(
        "Use the 'uncensored' counting method, i.e. discard truncated"
        " lifetimes at the trajectory edges."
    ),
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
    + "_lifetime_hist"
)
if args.uncensored:
    outfile += "_uncensored"
if args.intermittency > 0:
    outfile += "_intermittency_%d" % args.intermittency
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


if args.intermittency > 0:
    print("Correcting for intermittency...")
    dtrj = mdt.dyn.correct_intermittency(
        dtrj.T, args.intermittency, inplace=True, verbose=True
    )
    dtrj = dtrj.T


print("Calculating histograms...")
lts_per_state, states = mdt.dtrj.lifetimes_per_state(
    dtrj, uncensored=args.uncensored, return_states=True
)
n_states = len(states)
del dtrj

# Binning is done in trajectory steps.
# Linear bins.
# step = 1
# bins = np.arange(1, n_frames, step, dtype=np.float64)
# Logarithmic bins.
stop = int(np.ceil(np.log2(n_frames))) + 1
bins = np.logspace(0, stop, stop + 1, base=2, dtype=np.float64)
unit_bins = True if np.allclose(np.diff(bins), 1) else False
bins -= 0.5
bin_mids = bins[1:] - np.diff(bins) / 2
hists = np.full((n_states, len(bins) - 1), np.nan, dtype=np.float32)
for state_ix, lts_state in enumerate(lts_per_state):
    if np.any(lts_state < bins[0]) or np.any(lts_state > bins[-1]):
        raise ValueError(
            "At least one lifetime lies outside the binned region"
        )
    hists[state_ix], _bins = np.histogram(lts_state, bins=bins, density=True)
    if not np.allclose(_bins, bins, rtol=0):
        raise ValueError("`_bins` != `bins`.  This should not have happened")
    if not np.isclose(np.sum(hists[state_ix] * np.diff(bins)), 1):
        raise ValueError(
            "The integral of the histogram ({}) is not close to"
            " one".format(np.sum(hists[state_ix] * np.diff(bins)))
        )
del lts_per_state, lts_state, _bins


if args.intermittency == 0:
    print("Fitting power law...")
    state_ix_fit = n_states // 2
    fit_start_ix = 0
    _, fit_stop_ix = mdt.nph.find_nearest(bin_mids, 50, return_index=True)
    fit_stop_ix += 1
    hist_fit = hists[state_ix_fit][fit_start_ix:fit_stop_ix]
    bin_mids_fit = bin_mids[fit_start_ix:fit_stop_ix]  # This is a view!
    popt, pcov = curve_fit(
        f=leap.misc.straight_line,
        xdata=np.log(bin_mids_fit),
        ydata=np.log(hist_fit),
        p0=(-1.5, np.log(hist_fit[0])),
    )
    # perr = np.sqrt(np.diag(pcov))
    hist_fit = leap.misc.power_law(bin_mids_fit, popt[0], np.exp(popt[1]))


print("Creating plots...")
bins *= time_conv
bin_mids *= time_conv
# Don't do the following, because `bin_mids_fit` is a view of `bin_mids`
# bin_mids_fit *= time_conv

xlabel = "Lifetime / ns"
ylabel = "PDF"
xmin_xlin = 0
xmin_xlog = 1 * time_conv
xmax_ylin = 0.2
xmax_ylog = 200
ymin_ylin = 0
ymin_ylog = np.min(hists[hists > 0]) / 2
ymax_ylin = 0.5
ymax_ylog = 0.6

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
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for state_ix, state_num in enumerate(states):
        if not unit_bins:
            ax.stairs(
                hists[state_ix],
                bins,
                fill=False,
                label=r"$%d$" % (state_num + 1),
                alpha=leap.plot.ALPHA,
                rasterized=False,
            )
        else:
            ax.plot(
                bin_mids,
                hists[state_ix],
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
            r"$\propto t^{%.2f}$" % popt[0],
            rotation=np.rad2deg(np.arctan(popt[0])) / 1.6,
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

print("Created {}".format(outfile))
print("Done")
