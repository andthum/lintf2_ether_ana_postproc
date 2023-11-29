#!/usr/bin/env python3

"""
Calculate bin residence times / lifetimes for different intermittency
values.

For a single simulation, calculate the average time that a given
compound stays in a given bin directly from the discrete trajectory for
different intermittency values.
"""


# Standard libraries
import argparse
import glob
import os

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from scipy import stats

# First-party libraries
import lintf2_ether_ana_postproc as leap


def dist_characs(a, axis=-1, n_moms=4):
    """
    Estimate distribution characteristics from a given sample.

    Parameters
    ----------
    a : array_like
        Array of samples.
    axis : int, optional
        The axis along which to compute the distribution
        characteristics.
    n_moms : int, optional
        Number of raw moments to calculate.

    Returns
    -------
    characs : numpy.ndarray
        Array of shape ``(10 + n_moms-1, )`` containing the following
        distribution characteristics:

            1. Sample mean (unbiased 1st raw moment)
            2. Uncertainty of the sample mean (standard error)
            3. Corrected sample standard deviation
            4. Unbiased sample skewness (Fisher)
            5. Unbiased sample excess kurtosis (according to Fisher)
            6. Sample median
            7. Non-parametric skewness
            8. 2nd raw moment (biased estimate)
            9. 3rd raw moment (biased estimate)
            10. 4th raw moment (biased estimate)
            11. Sample minimum
            12. Sample maximum
            13. Number of samples

        The number of calculated raw moments depends on `n_moms`.  The
        first raw moment (mean) is always calculated.
    """
    a = np.asarray(a)
    nobs, min_max, mean, var, skew, kurt = stats.describe(
        a, axis=axis, ddof=1, bias=False
    )
    median = np.median(a, axis=axis)
    std = np.sqrt(var)
    skew_non_param = np.divide((mean - median), std)
    raw_moments = [np.mean(a**n) for n in range(2, n_moms + 1)]
    characs = np.array(
        [
            mean,  # Sample mean.
            np.divide(std, np.sqrt(nobs)),  # Uncertainty of sample mean
            std,  # Corrected sample standard deviation.
            skew,  # Unbiased sample skewness (Fisher).
            kurt,  # Unbiased sample excess kurtosis (Fisher).
            median,  # Median of the sample.
            skew_non_param,  # Non-parametric skewness.
            *raw_moments,  # 2nd to 4th raw moment (biased).
            *min_max,  # Minimum and maximum value of the sample.
            nobs,  # Number of observations (sample points).
        ]
    )
    return characs


def count_method(
    dtrj, uncensored=False, n_moms=4, time_conv=1, states_check=None
):
    """
    Estimate characteristics of the underlying lifetime distribution
    from a sample of lifetimes.

    Take a discrete trajectory and count the number of frames that a
    given compound stays in a given state.  Estimate characteristics of
    the underlying lifetime distribution from the obtained sample.

    Parameters
    ----------
    dtrj : array_like
        The discrete trajectory.  Array of shape ``(n, f)``, where ``n``
        is the number of compounds and ``f`` is the number of frames.
        The shape can also be ``(f,)``, in which case the array is
        expanded to shape ``(1, f)``.   The elements of `dtrj` are
        interpreted as the indices of the states in which a given
        compound is at a given frame.
    uncensored : bool, optional
        If ``True`` only take into account uncensored states, i.e.
        states whose start and end lie within the trajectory.  In other
        words, discard the truncated (censored) states at the beginning
        and end of the trajectory.  For these states the start/end time
        is unknown.
    n_moms : int, optional
        Number of raw moments to calculate.
    time_conv : scalar, optional
        Time conversion factor.  All lifetimes are multiplied by this
        factor.
    states_check : array_like or None, optional
        Expected state indices.  If provided, the state indices in the
        discrete trajectory are checked against the provided state
        indices.

    Returns
    -------
    characs : numpy.ndarray
        Estimated distribution characteristics.  See
        :func:`dist_characs` for more details.
    states : numpy.ndarray
        The state indices.

    Raises
    ------
    ValueError :
        If the state indices in the given discrete trajectory are not
        contained in the given array of state indices.
    """
    lts_per_state, states = mdt.dtrj.lifetimes_per_state(
        dtrj, uncensored=uncensored, return_states=True
    )
    lts_per_state = [lts * time_conv for lts in lts_per_state]
    if states_check is not None and not np.all(np.isin(states, states_check)):
        raise ValueError(
            "`states` ({}) is not fully contained in `states_check`"
            " ({})".format(states, states_check)
        )
    if states_check is not None:
        n_states = len(states_check)
    else:
        n_states = len(states)
    characs = np.full((n_states, 9 + n_moms), np.nan, dtype=np.float64)
    characs[:, -1] = 0  # Default number of observations.
    for i, lts in enumerate(lts_per_state):
        if len(lts) == 0:
            if not uncensored:
                raise ValueError(
                    "`len(lts) == 0` although `uncensored` is False"
                )
            continue
        else:
            characs[i] = dist_characs(lts, n_moms=n_moms)
    return characs, states


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "For a single simulation, calculate the average time that a given"
        " compound stays in a given bin for different intermittency values."
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
    "--method",
    type=str,
    required=False,
    default="count_censored",
    choices=("count_censored", "count_uncensored", "rate"),
    help=(
        "The method to use to calculate the residence times.  Default:"
        " %(default)s."
    ),
)
args = parser.parse_args()

analysis = "discrete-z"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
tool = "mdt"  # Analysis software.
infile_pattern = (
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis
    + analysis_suffix
    + "_dtrj_intermittency_[0-9]*.npz"
)
outfile = (
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis
    + analysis_suffix
    + "_lifetimes_intermittency_dependence_"
    + args.method
    + ".pdf"
)

# Time conversion factor to convert trajectory steps to ns.
time_conv = 2e-3
# Number of moments to calculate.
n_moms = 1
# Minimum and maximum intermittency value to consider
int_min = 0  # [Trajectory steps]
int_max = 512  # [Trajectory steps]


print("Creating Simulation instance(s)...")
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    path_key = "q%g" % surfq
else:
    surfq = None
    path_key = "bulk"
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, path_key)


print("Finding and sorting input files...")
file_suffix = analysis + analysis_suffix + "_dtrj.npz"
infile_int0 = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
states = np.unique(mdt.fh.load_dtrj(infile_int0))

path = os.path.dirname(os.path.abspath(infile_int0))
path = os.path.join(path, "lifetime_intermittency_evaluation_power")
if not os.path.isdir(path):
    raise FileNotFoundError("No such directory: '{}'".format(path))
infile_pattern = os.path.join(path, infile_pattern)
infiles = np.asarray(glob.glob(infile_pattern), dtype=str)
if len(infiles) == 0:
    raise ValueError(
        "Could not find any file matching the pattern"
        " '{}'".format(infile_pattern)
    )
intermittencies = np.zeros(len(infiles), dtype=np.float32)
for i, infile in enumerate(infiles):
    infile = os.path.splitext(os.path.basename(infile))[0]
    intermittency = infile.split("_intermittency_")[1]
    intermittencies[i] = np.uint16(intermittency)
sort_ix = np.argsort(intermittencies)
intermittencies, infiles = intermittencies[sort_ix], infiles[sort_ix]
intermittencies = np.insert(intermittencies, 0, np.uint16(0))
infiles = np.insert(infiles, 0, infile_int0)

valid = (intermittencies >= int_min) & (intermittencies <= int_max)
intermittencies = intermittencies[valid]
infiles = infiles[valid]
n_files = len(infiles)
intermittencies *= 2  # Convert trajectory frames to ps.


print("Calculating characteristics of the lifetime distributions...")
lts_characs_int = [None for i in intermittencies]
for i, infile in enumerate(infiles):
    dtrj = mdt.fh.load_dtrj(infile)
    if args.method == "count_censored":
        # Method 1: Censored counting.
        lts_characs, _states = count_method(
            dtrj,
            uncensored=False,
            n_moms=n_moms,
            time_conv=time_conv,
            states_check=states,
        )
    elif args.method == "count_uncensored":
        # Method 2: Uncensored counting.
        lts_characs, _states = count_method(
            dtrj,
            uncensored=True,
            n_moms=n_moms,
            time_conv=time_conv,
            states_check=states,
        )
    elif args.method == "rate":
        # Method 3: Calculate the transition rate as the number of
        # transitions leading out of a given state divided by the number
        # of frames that compounds have spent in this state.  The
        # average lifetime is calculated as the inverse transition rate.
        rates, _states = mdt.dtrj.trans_rate_per_state(
            dtrj, return_states=True
        )
        lts_characs = time_conv / rates
        # # Bring `lts_characs` to the same format as the output of the
        # # function `count_method`, i.e. one row for each state/bin.
        # lts_characs = lts_characs.reshape(len(lts_characs), 1)
        if not np.array_equal(_states, states):
            raise ValueError(
                "`_states` ({}) != `states` ({})".format(_states, states)
            )
        del rates
    lts_characs_int[i] = lts_characs
del dtrj, lts_characs, _states


print("Creating plot(s)...")
# Read bin edges.
file_suffix = analysis + analysis_suffix + "_bins.txt.gz"
infile_bins = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
bins = np.loadtxt(infile_bins)
bins /= 10  # A -> nm.
bins_low = bins[states]  # Lower bin edges.
bins_up = bins[states + 1]  # Upper bin edges.
bin_mids = bins_up - (bins_up - bins_low) / 2

Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm.
box_z = Sim.box[2] / 10  # A -> nm.

xlabel = r"$z$ / nm"
xlim = (0, box_z)
if args.method in ("count_censored", "count_uncensored"):
    ylabels = (
        "Residence Time / ns",
        "Std. Dev. / ns",
        "Skewness",
        "Excess Kurtosis",
        "Median / ns",
        "Non-Parametric Skewness",
        "Min. Lifetime / ns",
        "Max. Lifetime / ns",
        "No. of Samples",
    )
elif args.method == "rate":
    ylabels = ("Residence Time / ns",)
else:
    raise ValueError("Unknown --method ({})".format(args.method))
if surfq is None:
    legend_title = ""
else:
    legend_title = r"$\sigma_s = \pm %.2f$ $e$/nm$^2$" % surfq + "\n"
legend_title = (
    legend_title
    + r"$n_{EO} = %d$, " % Sim.O_per_chain
    + r"$r = %.4f$" % Sim.Li_O_ratio
    + "\n"
    + "Allowed Intermittence / ps"
)
height_ratios = (0.2, 1)

cmap = plt.get_cmap()
c_vals = np.arange(n_files)
c_norm = max(1, n_files - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for i, ylabel in enumerate(ylabels):
        if args.method in ("count_censored", "count_uncensored"):
            if i == 0:
                offset_i = 0
            else:
                offset_i = 1
        else:
            offset_i = 0
        fig, axs = plt.subplots(
            clear=True,
            nrows=2,
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        fig.set_figheight(fig.get_figheight() * sum(height_ratios))
        ax_profile, ax = axs
        ax.set_prop_cycle(color=colors)
        leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
        if surfq is not None:
            leap.plot.elctrds(
                ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
            )
        if i == 2:
            # Skewness of exponential distribution is 2.
            ax.axhline(
                y=2, color="tab:red", linestyle="dashed", label="Exp. Dist."
            )
        elif i == 3:
            # Excess kurtosis of exponential distribution is 6.
            ax.axhline(
                y=6, color="tab:red", linestyle="dashed", label="Exp. Dist."
            )
        elif i == 5:
            # Non-parametric skewness exponential distribution is
            # 1 - ln(2).
            ax.axhline(
                y=1 - np.log(2),
                color="tab:red",
                linestyle="dashed",
                label="Exp. Dist.",
            )
        for int_ix, lts_characs in enumerate(lts_characs_int):
            if args.method in ("count_censored", "count_uncensored"):
                ax.errorbar(
                    bin_mids,
                    lts_characs[:, i + offset_i],
                    yerr=lts_characs[:, i + 1] if i == 0 else None,
                    label=r"$%d$" % intermittencies[int_ix],
                    alpha=leap.plot.ALPHA,
                )
            else:
                ax.plot(
                    bin_mids,
                    lts_characs,
                    label=r"$%d$" % intermittencies[int_ix],
                    alpha=leap.plot.ALPHA,
                )
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax.get_ylim()
        if i not in (2, 3, 5) and ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        leap.plot.bins(ax, bins=bins)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend = fig.legend(
            title=legend_title,
            ncol=4,
            bbox_to_anchor=(0.58, 1.01),
            loc="upper center",
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        # Log scale y.
        ax.relim()
        ax.autoscale()
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        pdf.savefig()
        plt.close()

print("Created {}".format(outfile))
print("Done")
