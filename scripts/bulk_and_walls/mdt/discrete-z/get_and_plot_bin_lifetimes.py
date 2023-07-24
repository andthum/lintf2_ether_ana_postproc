#!/usr/bin/env python3


"""
Calculate bin residence times / lifetimes.

For a single simulation, calculate the average time that a given
compound stays in a given bin directly from the discrete trajectory
(Method 1-2) and from the corresponding remain probability function
(Method 3-4).
"""


# Standard libraries
import argparse

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from scipy.special import gamma

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "For a single simulation, calculate the average time that a given"
        " compound stays in a given bin."
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
    choices=("Li",),  # ("Li", "NBT", "OBT", "OE"),
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--continuous",
    required=False,
    default=False,
    action="store_true",
    help="Use the 'continuous' definition of the remain probability function.",
)
parser.add_argument(
    "--int-thresh",
    type=float,
    required=False,
    default=0.01,
    help=(
        "Only calculate the lifetime by directly integrating the remain"
        " probability if the remain probability decayed below the given"
        " threshold.  Default:  %(default)s."
    ),
)
parser.add_argument(
    "--end-fit",
    type=float,
    required=False,
    default=None,
    help=(
        "Last lag time (in ns) to include when fitting the remain probability."
        "  Default:  %(default)s (this means end at 90%% of the lag times)."
    ),
)
parser.add_argument(
    "--stop-fit",
    type=float,
    required=False,
    default=0.01,
    help=(
        "Stop fitting the remain probability as soon as it falls below this"
        " threshold.  The fitting is stopped by whatever happens earlier:"
        " --end-fit or --stop-fit.  Default: %(default)s"
    ),
)
args = parser.parse_args()

if args.continuous:
    con = "_continuous"
else:
    con = ""

analysis = "discrete-z"  # Analysis name.
# Common file suffix of analysis input files.
file_suffix_common = analysis + "_" + args.cmp
tool = "mdt"  # Analysis software.
outfile_base = (
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis
    + "_"
    + args.cmp
    + "_lifetimes"
    + con
)
outfile_txt = outfile_base + ".txt.gz"
outfile_pdf = outfile_base + ".pdf"

time_conv = 2e-3  # Trajectory steps -> ns.


print("Creating Simulation instance(s)...")
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    path_key = "q%g" % surfq
else:
    surfq = None
    path_key = "bulk"
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, path_key)


print("Reading data and calculating lifetimes (Method 1-2)...")
# Discrete trajectory.
file_suffix = file_suffix_common + "_dtrj.npz"
infile_dtrj = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
dtrj = mdt.fh.load_dtrj(infile_dtrj)
n_frames = dtrj.shape[1]

# Method 1: Calculate the average lifetime by counting the number of
# frames that a given compound stays in a given state.
lifetimes_cnt, states_cnt = mdt.dtrj.lifetimes_per_state(
    dtrj, return_states=True
)
lifetimes_cnt = [lts * time_conv for lts in lifetimes_cnt]
lifetimes_cnt_mom1 = np.array([np.nanmean(lts) for lts in lifetimes_cnt])
lifetimes_cnt_mom2 = np.array([np.nanmean(lts**2) for lts in lifetimes_cnt])
lifetimes_cnt_mom3 = np.array([np.nanmean(lts**3) for lts in lifetimes_cnt])
del lifetimes_cnt

# Method 2: Calculate the transition rate as the number of transitions
# leading out of a given state divided by the number of frames that
# compounds have spent in this state.  The average lifetime is
# calculated as the inverse transition rate.
rates, states_k = mdt.dtrj.trans_rate_per_state(dtrj, return_states=True)
lifetimes_k = time_conv / rates
if not np.array_equal(states_k, states_cnt):
    raise ValueError(
        "`states_k` ({}) != `states_cnt` ({})".format(states_k, states_cnt)
    )
del dtrj, rates, states_k


print("Reading data and calculating lifetimes (Method 3-5)...")
# Read remain probability functions (one for each bin).
file_suffix = file_suffix_common + "_state_lifetime_discrete" + con + ".txt.gz"
infile_rp = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
remain_props = np.loadtxt(infile_rp)
states = remain_props[0, 1:]  # State/Bin indices.
times = remain_props[1:, 0]  # Lag times in trajectory steps.
remain_props = remain_props[1:, 1:]  # Remain probability functions.
if np.any(remain_props < 0) or np.any(remain_props > 1):
    raise ValueError(
        "Some values of the remain probability lie outside the interval [0, 1]"
    )
if not np.array_equal(times, np.arange(n_frames)):
    print("`n_frames` =", n_frames)
    print("`times` =")
    print(times)
    raise ValueError("`times` != `np.arange(n_frames)`")
times *= time_conv  # Trajectory steps -> ns.
if np.any(np.modf(states)[0] != 0):
    raise ValueError(
        "Some state indices are not integers but floats.  states ="
        " {}".format(states)
    )
if not np.array_equal(states, states_cnt):
    raise ValueError(
        "`states` ({}) != `states_cnt` ({})".format(states, states_cnt)
    )
del states_cnt
states = states.astype(np.int32)
n_states = len(states)

# Method 3: Set the lifetime to the lag time at which the remain
# probability crosses 1/e.
thresh = 1 / np.e
ix_thresh = np.nanargmax(remain_props <= thresh, axis=0)
lifetimes_e = np.full(n_states, np.nan, dtype=np.float64)
for i, rp in enumerate(remain_props.T):
    if rp[ix_thresh[i]] > thresh:
        # The remain probability never falls below the threshold.
        lifetimes_e[i] = np.nan
    elif ix_thresh[i] < 1:
        # The remain probability immediately falls below the threshold.
        lifetimes_e[i] = 0
    elif rp[ix_thresh[i] - 1] < thresh:
        raise ValueError(
            "The threshold ({}) does not lie within the remain probability"
            " interval ([{}, {}]) at the found indices ({}, {}).  This should"
            " not have happened.".format(
                thresh,
                rp[ix_thresh[i]],
                rp[ix_thresh[i] - 1],
                ix_thresh[i],
                ix_thresh[i] - 1,
            )
        )
    else:
        lifetimes_e[i] = leap.misc.line_inv(
            y=thresh,
            xp1=times[ix_thresh[i] - 1],
            xp2=times[ix_thresh[i]],
            fp1=rp[ix_thresh[i] - 1],
            fp2=rp[ix_thresh[i]],
        )
        if (
            times[ix_thresh[i] - 1] > lifetimes_e[i]
            or times[ix_thresh[i]] < lifetimes_e[i]
        ):
            raise ValueError(
                "The lifetime ({}) does not lie within the time interval"
                " ([{}, {}]) at the found indices ({}, {}).  This should not"
                " have happened.".format(
                    lifetimes_e[i],
                    times[ix_thresh[i] - 1],
                    times[ix_thresh[i]],
                    ix_thresh[i] - 1,
                    ix_thresh[i],
                )
            )

# Method 4: Calculate the lifetime as the integral of the remain
# probability.
lifetimes_int_mom1 = np.full(n_states, np.nan, dtype=np.float64)
lifetimes_int_mom2 = np.full(n_states, np.nan, dtype=np.float64)
lifetimes_int_mom3 = np.full(n_states, np.nan, dtype=np.float64)
for i, rp in enumerate(remain_props.T):
    valid = ~np.isnan(rp)
    lifetimes_int_mom1[i] = np.trapz(y=rp[valid], x=times[valid])
    lifetimes_int_mom2[i] = np.trapz(
        y=rp[valid] * times[valid], x=times[valid]
    )
    lifetimes_int_mom3[i] = np.trapz(
        y=rp[valid] * (times[valid] ** 2), x=times[valid]
    )
    lifetimes_int_mom3[i] /= 2
invalid = np.all(remain_props > args.int_thresh, axis=0)
lifetimes_int_mom1[invalid] = np.nan
lifetimes_int_mom2[invalid] = np.nan
lifetimes_int_mom3[invalid] = np.nan
del valid, invalid

# Method 5: Fit the remain probability with a stretched exponential and
# calculate the lifetime as the integral of this stretched exponential.
if args.end_fit is None:
    end_fit = int(0.9 * len(times))
else:
    _, end_fit = mdt.nph.find_nearest(times, args.end_fit, return_index=True)
end_fit += 1  # Make `end_fit` inclusive.
fit_start = np.zeros(n_states, dtype=np.uint32)  # Inclusive.
fit_stop = np.zeros(n_states, dtype=np.uint32)  # Exclusive.

# Initial guesses for `tau0` and `beta`.
init_guess = np.column_stack([lifetimes_e, np.ones(n_states)])
init_guess[np.isnan(init_guess)] = 1.5 * times[-1]

popt = np.full((n_states, 2), np.nan, dtype=np.float64)
perr = np.full((n_states, 2), np.nan, dtype=np.float64)
fit_r2 = np.full(n_states, np.nan, dtype=np.float64)
fit_mse = np.full(n_states, np.nan, dtype=np.float64)
for i, rp in enumerate(remain_props.T):
    stop_fit = np.nanargmax(rp < args.stop_fit)
    if stop_fit == 0 and rp[stop_fit] >= args.stop_fit:
        stop_fit = len(rp)
    elif stop_fit < 2:
        stop_fit = 2
    fit_stop[i] = min(end_fit, stop_fit)
    times_fit = times[fit_start[i] : fit_stop[i]]
    rp_fit = rp[fit_start[i] : fit_stop[i]]
    popt[i], perr[i] = mdt.func.fit_kww(
        xdata=times_fit, ydata=rp_fit, p0=init_guess[i], method="trf"
    )
    if np.any(np.isnan(popt[i])):
        fit_mse[i] = np.nan
        fit_r2[i] = np.nan
    else:
        fit = mdt.func.kww(times_fit, *popt[i])
        # Residual sum of squares.
        ss_res = np.nansum((rp_fit - fit) ** 2)
        # Mean squared error / mean squared residuals.
        fit_mse[i] = ss_res / len(fit)
        # Total sum of squares
        ss_tot = np.nansum((rp_fit - np.nanmean(rp_fit)) ** 2)
        # (Pseudo) coefficient of determination (R^2).
        # https://www.r-bloggers.com/2021/03/the-r-squared-and-nonlinear-regression-a-difficult-marriage/
        fit_r2[i] = 1 - (ss_res / ss_tot)
tau0, beta = popt.T
tau0_sd, beta_sd = perr.T
lifetimes_exp_mom1 = tau0 / beta * gamma(1 / beta)
lifetimes_exp_mom2 = tau0**2 / beta * gamma(2 / beta)
lifetimes_exp_mom3 = tau0**3 / beta * gamma(3 / beta) / 2


print("Creating output file(s)...")
# Read bin edges.
file_suffix = file_suffix_common + "_bins" + ".txt.gz"
infile_bins = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
bins = np.loadtxt(infile_bins)

Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK
box_z = Sim.box[2]

header = (
    "Bin residence times.\n"
    + "Average time that a given compound stays in a given bin calculated\n"
    + "either directly from the discrete trajectory (Method 1-2) or from the\n"
    + "corresponding remain probability function (Method 3-4).\n"
    + "\n"
    + "System:              {:s}\n".format(args.system)
    + "Settings:            {:s}\n".format(args.settings)
    + "Bin edges:           {:s}\n".format(infile_bins)
    + "Discrete trajectory: {:s}\n".format(infile_dtrj)
    + "Remain probability:  {:s}\n".format(infile_rp)
    + "\n"
    + "Compound:                      {:s}\n".format(args.cmp)
)
if surfq is not None:
    header += "Surface charge:                {:.2f} e/nm^2\n".format(surfq)
header += (
    "Lithium-to-ether-oxygen ratio: {:.4f}\n".format(Sim.Li_O_ratio)
    + "Ether oxygens per PEO chain:   {:d}\n".format(Sim.O_per_chain)
    + "\n"
    + "\n"
    + "Residence times are calculated using five different methods:\n"
    + "\n"
    + "1) The residence time <tau_cnt> is calculated by counting how many\n"
    + "   frames a given compound stays in a given bin.  Note that residence\n"
    + "   times calculated in this way can at maximum be as long as the\n"
    + "   trajectory and are usually biased to lower values because of edge\n"
    + "   effects.\n"
    + "\n"
    + "2) The average transition rate <k> is calculated as the number of\n"
    + "   transitions leading out of a given bin divided by the number of\n"
    + "   frames that compounds have spent in this bin.  The average\n"
    + "   lifetime <tau_k> is calculated as the inverse transition rate:\n"
    + "     <tau_k> = 1 / <k>\n"
    + "\n"
    + "3) The residence time <tau_e> is set to the lag time at which the\n"
    + "   remain probability function p(t) crosses 1/e.  If this never\n"
    + "   happens, <tau_e> is set to NaN.\n"
    + "\n"
    + "4) According to Equations (12) and (14) of Reference [1], the n-th\n"
    + "   moment of the residence time <tau_int^n> is calculated as the\n"
    + "   integral of the remain probability function p(t) times t^{n-1}:\n"
    + "     <tau_int^n> = 1/(n-1)! int_0^inf t^{n-1} p(t) dt\n"
    + "   If p(t) does not decay below the given threshold of\n"
    + "   {:.4f}, <tau_int^n> is set to NaN.\n".format(args.int_thresh)
    + "\n"
    + "5) The remain probability function p(t) is fitted by a stretched\n"
    + "   exponential function using the 'Trust Region Reflective' method of\n"
    + "   scipy.optimize.curve_fit:\n"
    + "     f(t) = exp[-(t/tau0)^beta]\n"
    + "   Thereby, tau0 is confined to positive values and beta is confined\n"
    + "   to the interval [0, 1].  The remain probability is fitted until it\n"
    + "   decays below a given threshold or until a given lag time is\n"
    + "   reached (whatever happens earlier).  The n-th moment of the\n"
    + "   residence time <tau_exp^n> is calculated according to Equations\n"
    + "   (12) and (14) of Reference [1] and Equation (16) of Reference [2]\n"
    + "   as the integral of f(t) times t^{n-1}:\n"
    + "     <tau_exp^n> = 1/(n-1)! int_0^infty t^{n-1} f(t) dt\n"
    + "                 = tau0^n/beta * Gamma(1/beta)/Gamma(n)\n"
    + "   where Gamma(x) is the gamma function.\n"
    + "\n"
    + "Note that the moments calculated by method 4 and 5 are the moments of\n"
    + "an assumed underlying, continuous distribution of time constants tau.\n"
    + "They are related to the moments of the underlying distribution of\n"
    + "decay times t by\n"
    + "  <t^n> = n! * <tau^{n+1}> / <tau>\n"
    + "Compare Equation (14) of Reference [1].\n"
    + "\n"
    + "Reference [1]:\n"
    + "  M. N. Berberan-Santos, E. N. Bodunov, B. Valeur,\n"
    + "  Mathematical functions for the analysis of luminescence decays with\n"
    + "  underlying distributions 1. Kohlrausch decay function (stretched\n"
    + "  exponential),\n"
    + "  Chemical Physics, 2005, 315, 171-182.\n"
    + "Reference [2]:\n"
    + "  D. C. Johnston,\n"
    + "  Stretched exponential relaxation arising from a continuous sum of\n"
    + "  exponential decays,\n"
    + "  Physical Review B, 2006, 74, 184430.\n"
    + "\n"
    + "Box edges:          {:>16.9e}, {:>16.9e} A\n".format(0, box_z)
    + "Electrode surfaces: {:>16.9e}, {:>16.9e} A\n".format(
        elctrd_thk, box_z - elctrd_thk
    )
    + "int_thresh = {:.4f}\n".format(args.int_thresh)
    + "\n"
    + "\n"
    + "The columns contain:\n"
    + "  1 State/Bin indices (zero based)\n"
    + "  2 Lower bin edges / A\n"
    + "  3 Upper bin edges / A\n"
    + "  4 Distance of the lower bin edges to the left electrode surface / A\n"
    + "  5 Distance of the lower bin edges to the right electrode surface / A"
    + "\n"
    + "  6 Distance of the upper bin edges to the left electrode surface / A\n"
    + "  7 Distance of the upper bin edges to the right electrode surface / A"
    + "\n"
    + "\n"
    + "  Residence times from Method 1 (counting)\n"
    + "  8 1st moment <tau_cnt> / ns\n"
    + "  9 2nd moment <tau_cnt^2> / ns^2\n"
    + " 10 3rd moment <tau_cnt^3> / ns^3\n"
    + "\n"
    + "  Residence times from Method 2 (inverse transition rate)\n"
    + " 11 <tau_k> / ns\n"
    + "\n"
    + "  Residence times from Method 3 (1/e criterion)\n"
    + " 12 <tau_e> / ns\n"
    + "\n"
    + "  Residence times from Method 4 (direct integral)\n"
    + " 13 1st moment <tau_int> / ns\n"
    + " 14 2nd moment <tau_int^2> / ns^2\n"
    + " 15 3rd moment <tau_int^3> / ns^3\n"
    + "\n"
    + "  Residence times from Method 5 (integral of the fit)\n"
    + " 16 1st moment <tau_exp> / ns\n"
    + " 17 2nd moment <tau_exp^2> / ns^2\n"
    + " 18 3rd moment <tau_exp^3> / ns^3\n"
    + " 19 Fit parameter tau0 / ns\n"
    + " 20 Standard deviation of tau0 / ns\n"
    + " 21 Fit parameter beta\n"
    + " 22 Standard deviation of beta\n"
    + " 23 Coefficient of determination of the fit (R^2 value)\n"
    + " 24 Mean squared error of the fit (mean squared residuals) / ns^2\n"
    + " 25 Start of fit region (inclusive) / ns\n"
    + " 26 End of fit region (exclusive) / ns\n"
    + "\n"
    + "Column number:\n"
)
header += "{:>14d}".format(1)
for i in range(2, 27):
    header += " {:>16d}".format(i)

bins_low = bins[states]  # Lower bin edges.
bins_up = bins[states + 1]  # Upper bin edges.
# Distance of the bins to the left/right electrode surface.
bins_low_el = bins_low - elctrd_thk
bins_up_el = box_z - elctrd_thk - bins_up
data = np.column_stack(
    [
        states,  # 1
        bins_low,  # 2
        bins_up,  # 3
        bins_low - elctrd_thk,  # 4
        box_z - elctrd_thk - bins_low,  # 5
        bins_up - elctrd_thk,  # 6
        box_z - elctrd_thk - bins_up,  # 7
        #
        lifetimes_cnt_mom1,  # 8
        lifetimes_cnt_mom2,  # 9
        lifetimes_cnt_mom3,  # 10
        #
        lifetimes_k,  # 11
        #
        lifetimes_e,  # 12
        #
        lifetimes_int_mom1,  # 13
        lifetimes_int_mom2,  # 14
        lifetimes_int_mom3,  # 15
        #
        lifetimes_exp_mom1,  # 16
        lifetimes_exp_mom2,  # 17
        lifetimes_exp_mom3,  # 18
        tau0,  # 19
        tau0_sd,  # 20
        beta,  # 21
        beta_sd,  # 22
        fit_r2,  # 23
        fit_mse,  # 24
        fit_start * time_conv,  # 25
        fit_stop * time_conv,  # 26
    ]
)
leap.io_handler.savetxt(outfile_txt, data, header=header)
print("Created {}".format(outfile_txt))


print("Creating plot(s)...")
elctrd_thk /= 10  # A -> nm.
box_z /= 10  # A -> nm.
bins /= 10  # A -> nm.
bin_mids = bins_up - (bins_up - bins_low) / 2
bin_mids /= 10  # A -> nm.

xlabel = r"$z$ / nm"
xlim = (0, box_z)

if surfq is None:
    legend_title = ""
else:
    legend_title = r"$\sigma_s = \pm %.2f$ $e$/nm$^2$" % surfq + "\n"
legend_title = (
    legend_title
    + r"$n_{EO} = %d$, " % Sim.O_per_chain
    + r"$r = %.4f$" % Sim.Li_O_ratio
)

height_ratios = (0.2, 1)

lifetime_min = np.nanmin(
    [
        lifetimes_cnt_mom1,
        lifetimes_k,
        lifetimes_e,
        lifetimes_int_mom1,
        lifetimes_exp_mom1,
    ]
)

cmap = plt.get_cmap()
c_vals = np.arange(n_states)
c_norm = n_states - 1
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile_pdf)
with PdfPages(outfile_pdf) as pdf:
    # Plot residence times vs. bins ("ensemble average").
    fig, axs = plt.subplots(
        clear=True,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.set_figheight(fig.get_figheight() * sum(height_ratios))
    ax_profile, ax = axs
    leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
    if surfq is not None:
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
    # Method 5 (integral of the fit).
    ax.errorbar(
        bin_mids,
        lifetimes_exp_mom1,
        yerr=np.sqrt(lifetimes_exp_mom2 - lifetimes_exp_mom1**2),
        label="Fit",
        color="tab:cyan",
        marker="D",
        alpha=leap.plot.ALPHA,
    )
    # Method 4 (direct integral)
    ax.errorbar(
        bin_mids,
        lifetimes_int_mom1,
        yerr=np.sqrt(lifetimes_int_mom2 - lifetimes_int_mom1**2),
        label="Area",
        color="tab:blue",
        marker="d",
        alpha=leap.plot.ALPHA,
    )
    # Method 3 (1/e criterion).
    ax.plot(
        bin_mids,
        lifetimes_e,
        label=r"$1/e$",
        color="tab:purple",
        marker="s",
        alpha=leap.plot.ALPHA,
    )
    # Method 2 (inverse transition rate).
    ax.plot(
        bin_mids,
        lifetimes_k,
        label="Rate",
        color="tab:red",
        marker="h",
        alpha=leap.plot.ALPHA,
    )
    # Method 1 (counting).
    ax.errorbar(
        bin_mids,
        lifetimes_cnt_mom1,
        yerr=np.sqrt(lifetimes_cnt_mom2 - lifetimes_cnt_mom1**2),
        label="Count",
        color="tab:orange",
        marker="H",
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel=xlabel,
        ylabel=r"Residence Time $\langle \tau \rangle$ / ns",
        xlim=xlim,
    )
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    leap.plot.bins(ax, bins=bins)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Sort legend entries.  By default, lines plotted with `ax.plot`
    # come before lines plotted with `ax.errorbar`.
    handles, labels = ax.get_legend_handles_labels()
    legend_order = (2, 3, 0, 1, 4)
    handles = [handles[leg_ord] for leg_ord in legend_order]
    labels = [labels[leg_ord] for leg_ord in legend_order]
    legend = ax.legend(
        handles,
        labels,
        title=legend_title,
        loc="upper center",
        ncol=2,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    # Set y axis to log scale
    # (residence times vs. bins, "ensemble average").
    # Round y limits to next lower and higher power of ten.
    ylim = ax.get_ylim()
    ymin = 10 ** np.floor(np.log10(lifetime_min))
    ymax = 10 ** np.ceil(np.log10(ylim[1]))
    ax.set_ylim(ymin if not np.isnan(ymin) else None, ymax)
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot residence times vs. bins ("time average").
    fig, axs = plt.subplots(
        clear=True,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.set_figheight(fig.get_figheight() * sum(height_ratios))
    ax_profile, ax = axs
    leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
    if surfq is not None:
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
    # Method 5 (integral of the fit).
    ax.errorbar(
        bin_mids,
        lifetimes_exp_mom2 / lifetimes_exp_mom1,
        yerr=np.sqrt(2 * lifetimes_exp_mom3 / lifetimes_exp_mom1),
        label="Fit",
        color="tab:cyan",
        marker="D",
        alpha=leap.plot.ALPHA,
    )
    # Method 4 (direct integral).
    ax.errorbar(
        bin_mids,
        lifetimes_int_mom2 / lifetimes_int_mom1,
        yerr=np.sqrt(2 * lifetimes_int_mom3 / lifetimes_int_mom1),
        label="Area",
        color="tab:blue",
        marker="d",
        alpha=leap.plot.ALPHA,
    )
    # Method 3 (1/e criterion).
    ax.plot(
        bin_mids,
        lifetimes_e,
        label=r"$1/e$",
        color="tab:purple",
        marker="s",
        alpha=leap.plot.ALPHA,
    )
    # Method 2 (inverse transition rate).
    ax.plot(
        bin_mids,
        lifetimes_k,
        label="Rate",
        color="tab:red",
        marker="h",
        alpha=leap.plot.ALPHA,
    )
    # Method 1 (counting).
    ax.errorbar(
        bin_mids,
        lifetimes_cnt_mom1,
        yerr=np.sqrt(lifetimes_cnt_mom2 - lifetimes_cnt_mom1**2),
        label="Count",
        color="tab:orange",
        marker="H",
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel=xlabel,
        ylabel=r"Residence Time $\bar{\tau}$ / ns",
        xlim=xlim,
    )
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    leap.plot.bins(ax, bins=bins)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Sort legend entries.  By default, lines plotted with `ax.plot`
    # come before lines plotted with `ax.errorbar`.
    handles, labels = ax.get_legend_handles_labels()
    legend_order = (2, 3, 0, 1, 4)
    handles = [handles[leg_ord] for leg_ord in legend_order]
    labels = [labels[leg_ord] for leg_ord in legend_order]
    legend = ax.legend(
        handles,
        labels,
        title=legend_title,
        loc="upper center",
        ncol=2,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    # Set y axis to log scale
    # (residence times vs. bins, "time average").
    # Round y limits to next lower and higher power of ten.
    ylim = ax.get_ylim()
    ymin = 10 ** np.floor(np.log10(lifetime_min))
    ymax = 10 ** np.ceil(np.log10(ylim[1]))
    ax.set_ylim(ymin if not np.isnan(ymin) else None, ymax)
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot fit parameter tau0.
    fig, axs = plt.subplots(
        clear=True,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.set_figheight(fig.get_figheight() * sum(height_ratios))
    ax_profile, ax = axs
    leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
    if surfq is not None:
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
    ax.errorbar(bin_mids, tau0, yerr=tau0_sd, marker="o")
    ax.set(xlabel=xlabel, ylabel=r"Fit Parameter $\tau_0$ / ns", xlim=xlim)
    leap.plot.bins(ax, bins=bins)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend = ax.legend(title=legend_title, loc="upper center")
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    if not np.all(np.isnan(tau0)):
        # Set y axis to log scale (fit parameter tau0).
        # Round y limits to next lower and higher power of ten.
        ylim = ax.get_ylim()
        ymin = 10 ** np.floor(np.log10(np.nanmin(tau0)))
        ymax = 10 ** np.ceil(np.log10(ylim[1]))
        ax.set_ylim(ymin if not np.isnan(ymin) else None, ymax)
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        pdf.savefig()
    plt.close()

    # Plot fit parameter beta.
    fig, axs = plt.subplots(
        clear=True,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.set_figheight(fig.get_figheight() * sum(height_ratios))
    ax_profile, ax = axs
    leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
    if surfq is not None:
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
    ax.errorbar(bin_mids, beta, yerr=beta_sd, marker="o")
    ax.set(
        xlabel=xlabel,
        ylabel=r"Fit Parameter $\beta$",
        xlim=xlim,
        ylim=(0, 1.05),
    )
    leap.plot.bins(ax, bins=bins)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend = ax.legend(title=legend_title, loc="upper center")
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot R^2 value of the fits.
    fig, axs = plt.subplots(
        clear=True,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.set_figheight(fig.get_figheight() * sum(height_ratios))
    ax_profile, ax = axs
    leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
    if surfq is not None:
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
    ax.plot(bin_mids, fit_r2, marker="o")
    ax.set(
        xlabel=xlabel,
        ylabel=r"Coeff. of Determ. $R^2$",
        xlim=xlim,
        ylim=(0, 1.05),
    )
    leap.plot.bins(ax, bins=bins)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend = ax.legend(title=legend_title, loc="lower center")
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot mean squared error.
    fig, axs = plt.subplots(
        clear=True,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.set_figheight(fig.get_figheight() * sum(height_ratios))
    ax_profile, ax = axs
    leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
    if surfq is not None:
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
    ax.plot(bin_mids, fit_mse, marker="o")
    ax.set(xlabel=xlabel, ylabel=r"Mean Squared Error / ns$^2$", xlim=xlim)
    leap.plot.bins(ax, bins=bins)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend = ax.legend(title=legend_title, loc="lower center")
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    if not np.all(np.isnan(fit_mse)):
        # Set y axis to log scale (mean squared error).
        # Round y limits to next lower and higher power of ten.
        ylim = ax.get_ylim()
        ymin = 10 ** np.floor(np.log10(np.nanmin(fit_mse[fit_mse > 0])))
        ymax = 10 ** np.ceil(np.log10(ylim[1]))
        ax.set_ylim(ymin if not np.isnan(ymin) else None, ymax)
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        pdf.savefig()
    plt.close()

    # Plot fitted region.
    fig, axs = plt.subplots(
        clear=True,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.set_figheight(fig.get_figheight() * sum(height_ratios))
    ax_profile, ax = axs
    leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
    if surfq is not None:
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
    ax.plot(bin_mids, fit_start * time_conv, label="Start", marker="^")
    ax.plot(bin_mids, fit_stop * time_conv, label="End", marker="v")
    ax.set(xlabel=xlabel, ylabel="Fitted Region / ns", xlim=xlim)
    leap.plot.bins(ax, bins=bins)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend = ax.legend(title=legend_title, loc="upper center", ncol=2)
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot remain probabilities and fits for each bin.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, rp in enumerate(remain_props.T):
        times_fit = times[fit_start[i] : fit_stop[i]]
        fit = mdt.func.kww(times_fit, *popt[i])
        lines = ax.plot(
            times, rp, label=r"$%d$" % (states[i] + 1), linewidth=1
        )
        ax.plot(times_fit, fit, linestyle="dashed", color=lines[0].get_color())
    ax.set(
        xlabel="Lag Time / ns",
        ylabel="Decay Law",
        xlim=(times[1], times[-1]),
        ylim=(0, 1),
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    legend = ax.legend(
        title=legend_title + "\nBin Number",
        loc="upper right",
        ncol=3,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot fit residuals for each bin.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, rp in enumerate(remain_props.T):
        times_fit = times[fit_start[i] : fit_stop[i]]
        fit = mdt.func.kww(times_fit, *popt[i])
        res = rp[fit_start[i] : fit_stop[i]] - fit
        ax.plot(times_fit, res, label=r"$%d$" % (states[i] + 1))
    ax.set(
        xlabel="Lag Time / ns",
        ylabel="Fit Residuals",
        xlim=(times[1], times[-1]),
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    legend = ax.legend(
        title=legend_title + "\nBin Number",
        loc="lower right",
        ncol=3,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile_pdf))
print("Done")
