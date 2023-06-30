#!/usr/bin/env python3


"""
Calculate bin residence times / lifetimes.

For a single simulation, calculate the average time that a given
compound stays in a given bin from the corresponding remain probability
function.
"""


# Standard libraries
import argparse

# Third-party libraries
import mdtools as mdt
import numpy as np
from scipy.special import gamma

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "For a single simulation, calculate the average time that a given"
        " compound stays in a given bin from the corresponding autocorrelation"
        " function."
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
    choices=("Li"),  # ("Li", "NBT", "OBT", "OE"),
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
tool = "mdt"  # Analysis software.
outfile = (
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis
    + "_"
    + args.cmp
    + "_state_lifetimes"
    + con
    + ".txt.gz"
)


print("Creating Simulation instance(s)...")
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    path_key = "q%g" % surfq
else:
    surfq = None
    path_key = "bulk"
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, path_key)


print("Reading data...")
# Read density profile.
ana_dens = "density-z"
file_suffix = ana_dens + "_number.xvg.gz"
infile_dens = leap.simulation.get_ana_file(Sim, ana_dens, "gmx", file_suffix)
cols_dens = (0, Sim.dens_file_cmp2col[args.cmp])
xdata, ydata = np.loadtxt(
    infile_dens, comments=["#", "@"], usecols=cols_dens, unpack=True
)
ydata = leap.misc.dens2free_energy(xdata, ydata, bulk_region=None)

# Read bin edges.
file_suffix_common = analysis + "_" + args.cmp
file_suffix = file_suffix_common + "_bins" + ".txt.gz"
infile_bins = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
bins = np.loadtxt(infile_bins)
bins /= 10  # A -> nm

# Read remain probability functions (one for each bin).
file_suffix = file_suffix_common + "_state_lifetime_discrete" + con + ".txt.gz"
infile_rp = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
remain_props = np.loadtxt(infile_rp)
states = remain_props[0:, 1]  # State/Bin numbers
times = remain_props[1:, 0] * 2e-3  # Trajectory step width -> ns.
remain_props = remain_props[1:, 1:]  # Remain probability functions.
if np.any(remain_props < 0) or np.any(remain_props > 1):
    raise ValueError(
        "Some values of the remain probability lie outside the interval [0, 1]"
    )
if np.any(np.modf(states)[0] != 0):
    raise ValueError("Some state indices are not integers but floats")
states = states.astype(np.int32)


print("Calculating lifetimes...")
# Method 1: Set the lifetime to the lag time at which the remain
# probability crosses 1/e.
thresh = 1 / np.e
ix_thresh = np.argmax(remain_props <= thresh, axis=0)
lifetimes_e = np.full(len(states), np.nan, dtype=np.float64)
for i, rp in enumerate(remain_props.T):
    if rp[ix_thresh[i]] > thresh:
        lifetimes_e[i] = np.nan
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
    elif ix_thresh[i] < 1:
        lifetimes_e[i] = 0
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

# Method 2: Directly calculate the integral of the remain probability.
lifetimes_int = np.trapz(y=remain_props, x=times, axis=0)
lifetimes_int_sd = np.trapz(y=remain_props * times, x=times, axis=0)
lifetimes_int_sd = np.sqrt(lifetimes_int_sd - lifetimes_int**2)
invalid = np.all(remain_props > args.int_thresh, axis=0)
lifetimes_int[invalid] = np.nan
lifetimes_int_sd[invalid] = np.nan

# Method 3: Fit the remain probability with a stretched exponential and
# calculate the lifetime as the integral of this stretched exponential.
if args.end_fit is None:
    end_fit = int(0.9 * len(times))
else:
    _, end_fit = mdt.nph.find_nearest(times, args.end_fit, return_index=True)
end_fit += 1  # Make `end_fit` inclusive.
fit_start = np.zeros(len(states), dtype=np.uint32)  # inclusive
fit_stop = np.zeros(len(states), dtype=np.uint32)  # Exclusive.

# Initial guesses for `tau` and `beta`.
init_guess = np.column_stack([lifetimes_int, np.ones(len(states))])
init_guess[np.isnan(init_guess)] = times[-1]

popt = np.full((len(states), 2), np.nan, dtype=np.float64)
perr = np.full((len(states), 2), np.nan, dtype=np.float64)
for i, rp in enumerate(remain_props.T):
    stop_fit = np.argmax(rp < args.stop_fit)
    if stop_fit == 0 and rp[stop_fit] >= args.stop_fit:
        stop_fit = len(rp)
    elif stop_fit < 2:
        stop_fit = 2
    fit_stop[i] = min(end_fit, stop_fit)
    popt[i], perr[i] = mdt.func.fit_kww(
        xdata=times[fit_start[i] : fit_stop[i]],
        ydata=rp[fit_start[i] : fit_stop[i]],
        p0=init_guess[i],
    )
del remain_props
tau, beta = popt.T
tau_sd, beta_sd = perr.T
lifetimes_exp = tau / beta * gamma(1 / beta)
# TODO: lifetimes_exp_sd = ??


print("Creating output file(s)...")
Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK
box_z = Sim.box[2]

bins_low = bins[states]  # Lower bin edges.
bins_up = bins[states + 1]  # Upper bin edges.
# Distance of the bins to the left/right electrode surface.
bins_low_el = bins_low - elctrd_thk
bins_up_el = box_z - elctrd_thk - bins_up

header = (
    "Bin residence times.\n"
    + "Average time that a given compound stays in a given bin calculated\n"
    + "from the corresponding remain probability function.\n"
    + "\n"
    + "System:             {:s}\n".format(args.system)
    + "Settings:           {:s}\n".format(args.settings)
    + "Density profile:    {:s}\n".format(infile_dens)
    + "Read Column(s):     {}\n".format(np.array(cols_dens) + 1)
    + "Bin edges:          {:s}\n".format(infile_bins)
    + "Remain probability: {:s}\n".format(infile_rp)
    + "\n"
    + "Compound:                      {:s}\n".format(args.cmp)
    + "Surface charge:                {:.2f} e/nm^2\n".format(surfq)
    + "Lithium-to-ether-oxygen ratio: {:.4f}\n".format(Sim.Li_O_ratio)
    + "Ether oxygens per PEO chain:   {:d}\n".format(Sim.O_per_chain)
    + "\n"
    + "\n"
    + "Residence times are calculated from the remain probability function\n"
    + "p(t) using three different methods:\n"
    + "\n"
    + "1) The residence time <tau_e> is set to the lag time at which p(t)\n"
    + "   crosses 1/e.  If this never happens, the <tau_e> is set to NaN.\n"
    + "\n"
    + "2) According to Equation (12) of Reference [1], the residence time\n"
    + "   <tau_int> is calculated as the integral of p(t):\n"
    + "     <tau_int> = int_0^inf p(t) dt\n"
    + "   However, if p(t) does not decay below the given threshold of\n"
    + "   {:.4f}, the residence time is set to NaN.  The standard\n".format(
        args.int_thresh
    )
    + "   deviation of the underlying distribution of residence times is\n"
    + "   estimated using Equation (14) of Reference [1] for the second\n"
    + "   moment:\n"
    + "     <tau_int^2> = int_0^infty t * I(t) dt\n"
    + "     tau_int_sd = <tau_int^2> - <tau_int>^2\n"
    + "\n"
    + "3) p(t) is fitted by a stretched exponential function:\n"
    + "     f(t) = exp[-(t/tau0)^beta]\n"
    + "   Thereby, tau0 is confined to positive values and beta is confined\n"
    + "   to the interval [0, 1].  The remain probability is fitted until it\n"
    + "   decays below {:.4f}\n or until a lag time of {:.4f} ns is\n".format(
        args.stop_fit, args.end_fit
    )
    + "   reached (whatever happens earlier).  The residence time <tau_exp>\n"
    + "   is calculated according to Equation (29) of Reference [1] as the\n"
    + "   integral of the stretched exponential fit function:\n"
    + "     <tau_exp> = int_0^infty f(t) dt = tau0/beta * Gamma(1/beta)\n"
    + "   where Gamma(x) is the gamma function.  Again, the standard\n"
    + "   deviation of the underlying distribution of residence times is\n"
    + "   estimated using Equation (14) of Reference [1] for the second\n"
    + "   moment:\n"
    + "     <tau_exp^2> = int_0^infty t * f(t) dt\n"
    + "     tau_exp_sd = <tau_exp^2> - <tau_exp>^2\n"
    + "\n"
    + "Reference [1]:\n"
    + "  M. N. Berberan-Santos, E. N. Bodunov, B. Valeur,\n"
    + "  Mathematical functions for the analysis of luminescence decays with\n"
    + "  underlying distributions 1. Kohlrausch decay function (stretched\n"
    + "  exponential),\n"
    + "  Chemical Physics, 2005, 315, 171-182\n"
    + "\n"
    + "Box edges:          {:>16.9e}, {:>16.9e} A\n".format(0, box_z)
    + "Electrode surfaces: {:>16.9e}, {:>16.9e} A\n".format(
        elctrd_thk, box_z - elctrd_thk
    )
    + "int_thresh = {:.4f}\n".format(args.int_thresh)
    + "end_fit    = {:.4f} ns\n".format(args.end_fit)
    + "stop_fit   = {:.4f}\n".format(args.stop_fit)
    + "\n"
    + "\n"
    + "The columns contain:\n"
    + "  1 Lower bin edges / A\n"
    + "  2 Upper bin edges / A\n"
    + "  3 Distance of the lower bin edges to the left electrode surface / A\n"
    + "  4 Distance of the lower bin edges to the right electrode surface / A"
    + "\n"
    + "  5 Distance of the upper bin edges to the left electrode surface / A\n"
    + "  6 Distance of the upper bin edges to the right electrode surface / A"
    + "\n"
    + "\n"
    + "  7 Residence times <tau_e> by the 1/e criterion / ns\n"
    + "\n"
    + "  8 Residence times <tau_int> by directly integrating the remain\n"
    + "    probability / ns\n"
    + "  9 Standard deviation of <tau_int> / ns\n"
    + "\n"
    + " 10 Residence times <tau_kww> by integrating the KWW fit / ns\n"
    + " 11 Standard deviation <tau_kww> / ns\n"
    + " 12 Fit parameter tau0 / ns\n"
    + " 13 Standard deviation of tau0 / ns\n"
    + " 14 Fit parameter beta\n"
    + " 15 Standard deviation of beta\n"
    + "\n"
    + "Column number:\n"
)
print("Created {}".format(outfile))

print("Done")
