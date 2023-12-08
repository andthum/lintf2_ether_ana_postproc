#!/usr/bin/env python3


"""
Calculate bin residence times / lifetimes.

For a single simulation, calculate the average time that a given
compound stays in a given bin directly from the discrete trajectory
(Method 1-3), from the corresponding remain probability function
(Method 4-7) of from the corresponding Kaplan-Meier estimate of the
survival function.
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

# Time conversion factor to convert trajectory steps to ns.
time_conv = 2e-3
# Number of moments to calculate.  For calculating the skewness, the 2nd
# and 3rd (central) moments are required, for the kurtosis the 2nd and
# 4th (central) moments are required.
n_moms = 4
# Fit method of `scipy.optimize.curve_fit` to use for all fits.
fit_method = "trf"


print("Creating Simulation instance(s)...")
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    path_key = "q%g" % surfq
else:
    surfq = None
    path_key = "bulk"
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, path_key)


print("Calculating lifetimes directly from `dtrj`...")
# Read discrete trajectory.
file_suffix = file_suffix_common + "_dtrj.npz"
infile_dtrj = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
dtrj = mdt.fh.load_dtrj(infile_dtrj)
n_frames = dtrj.shape[1]
# Method 1: Censored counting.
lts_cnt_cen_characs, states = leap.lifetimes.count_method(
    dtrj,
    uncensored=False,
    n_moms=n_moms,
    time_conv=time_conv,
    states_check=None,
)
n_states = len(states)
# Method 2: Uncensored counting.
lts_cnt_unc_characs, _states = leap.lifetimes.count_method(
    dtrj,
    uncensored=True,
    n_moms=n_moms,
    time_conv=time_conv,
    states_check=states,
)
del _states
# Method 3: Calculate the transition rate as the number of transitions
# leading out of a given state divided by the number of frames that
# compounds have spent in this state.  The average lifetime is
# calculated as the inverse transition rate.
rates, states_k = mdt.dtrj.trans_rate_per_state(dtrj, return_states=True)
lts_k = time_conv / rates
if not np.array_equal(states_k, states):
    raise ValueError(
        "`states_k` ({}) != `states` ({})".format(states_k, states)
    )
del dtrj, rates, states_k


print("Calculating lifetimes from the remain probability...")
# Read remain probabilities (one for each bin).
file_suffix = file_suffix_common + "_state_lifetime_discrete" + con + ".txt.gz"
infile_rp = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
remain_probs, times, _states = leap.simulation.read_time_state_matrix(
    infile_rp,
    time_conv=time_conv,
    amin=0,
    amax=1,
    n_rows_check=n_frames,
    states_check=states,
)
del _states
# Method 4: Numerical integration of the remain probability.
lts_rp_int_characs = leap.lifetimes.integral_method(
    remain_probs, times, n_moms=n_moms, int_thresh=args.int_thresh
)
# Get fit region for fitting methods.
fit_start_rp, fit_stop_rp = leap.lifetimes.get_fit_region(
    remain_probs, times, end_fit=args.end_fit, stop_fit=args.stop_fit
)
# Method 5: Weibull fit of the remain probability.
(
    lts_rp_wbl_characs,
    lts_rp_wbl_fit_goodness,
    popt_rp_wbl,
    perr_rp_wbl,
) = leap.lifetimes.weibull_fit_method(
    remain_probs,
    times,
    fit_start=fit_start_rp,
    fit_stop=fit_stop_rp,
    n_moms=n_moms,
    fit_method=fit_method,
)
tau0_rp_wbl, beta_rp_wbl = popt_rp_wbl.T
tau0_rp_wbl_sd, beta_rp_wbl_sd = perr_rp_wbl.T
# Method 6: Burr Type XII fit of the remain probability.
(
    lts_rp_brr_characs,
    lts_rp_brr_fit_goodness,
    popt_rp_brr,
    perr_rp_brr,
    popt_conv_rp_brr,
    perr_conv_rp_brr,
) = leap.lifetimes.burr12_fit_method(
    remain_probs,
    times,
    fit_start=fit_start_rp,
    fit_stop=fit_stop_rp,
    n_moms=n_moms,
    fit_method=fit_method,
)
tau0_rp_brr, beta_rp_brr, delta_rp_brr = popt_conv_rp_brr.T
tau0_rp_brr_sd, beta_rp_brr_sd, delta_rp_brr_sd = perr_conv_rp_brr.T

print("Calculating lifetimes from the Kaplan-Meier estimator...")
# Read Kaplan-Meier survival functions (one for each bin).
file_suffix = file_suffix_common + "_kaplan_meier_discrete_sf.txt.gz"
infile_km = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
file_suffix = file_suffix_common + "_kaplan_meier_discrete_sf_var.txt.gz"
infile_km_var = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
(
    km_surv_funcs,
    km_surv_funcs_var,
    times,
    _states,
) = leap.simulation.read_time_state_matrix(
    infile_km,
    fname_var=infile_km_var,
    time_conv=time_conv,
    amin=0,
    amax=1,
    n_rows_check=n_frames,
    states_check=states,
)
del _states
# Method 7: Numerical integration of the Kaplan-Meier estimator.
lts_km_int_characs = leap.lifetimes.integral_method(
    km_surv_funcs, times, n_moms=n_moms, int_thresh=args.int_thresh
)
# Get fit region for fitting methods.
fit_start_km, fit_stop_km = leap.lifetimes.get_fit_region(
    km_surv_funcs, times, end_fit=args.end_fit, stop_fit=args.stop_fit
)
# Method 8: Weibull fit of the Kaplan-Meier estimator.
(
    lts_km_wbl_characs,
    lts_km_wbl_fit_goodness,
    popt_km_wbl,
    perr_km_wbl,
) = leap.lifetimes.weibull_fit_method(
    km_surv_funcs,
    times,
    fit_start=fit_start_km,
    fit_stop=fit_stop_km,
    surv_funcs_var=km_surv_funcs_var,
    n_moms=n_moms,
    fit_method=fit_method,
)
tau0_km_wbl, beta_km_wbl = popt_km_wbl.T
tau0_km_wbl_sd, beta_km_wbl_sd = perr_km_wbl.T
# Method 9: Burr Type XII fit of the Kaplan-Meier estimator.
(
    lts_km_brr_characs,
    lts_km_brr_fit_goodness,
    popt_km_brr,
    perr_km_brr,
    popt_conv_km_brr,
    perr_conv_km_brr,
) = leap.lifetimes.burr12_fit_method(
    km_surv_funcs,
    times,
    fit_start=fit_start_km,
    fit_stop=fit_stop_km,
    surv_funcs_var=km_surv_funcs_var,
    n_moms=n_moms,
    fit_method=fit_method,
)
tau0_km_brr, beta_km_brr, delta_km_brr = popt_conv_km_brr.T
tau0_km_brr_sd, beta_km_brr_sd, delta_km_brr_sd = perr_conv_km_brr.T


print("Creating output file(s)...")
# Read bin edges.
file_suffix = file_suffix_common + "_bins" + ".txt.gz"
infile_bins = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
bins = np.loadtxt(infile_bins)

Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK
box_z = Sim.box[2]

bins_low = bins[states]  # Lower bin edges.
bins_up = bins[states + 1]  # Upper bin edges.
# Distance of the bins to the left/right electrode surface.
bins_low_el = bins_low - elctrd_thk
bins_up_el = box_z - elctrd_thk - bins_up

# Create text output.
data = np.column_stack(
    [
        states,  # 1
        bins_low,  # 2
        bins_up,  # 3
        bins_low - elctrd_thk,  # 4
        box_z - elctrd_thk - bins_low,  # 5
        bins_up - elctrd_thk,  # 6
        box_z - elctrd_thk - bins_up,  # 7
        # Method 1: Censored counting.
        lts_cnt_cen_characs,  # 8-21
        # Method 2: Uncensored counting.
        lts_cnt_unc_characs,  # 22-35
        # Method 3: Inverse transition rate.
        lts_k,  # 36
        # Method 4: Numerical integration of the remain probability.
        lts_rp_int_characs,  # 37-46
        # Method 5: Weibull fit of the remain probability.
        lts_rp_wbl_characs,  # 47-56
        tau0_rp_wbl,  # 57
        tau0_rp_wbl_sd,  # 58
        beta_rp_wbl,  # 59
        beta_rp_wbl_sd,  # 60
        lts_rp_wbl_fit_goodness,  # 61-62
        # Method 6: Burr Type XII fit of the remain probability.
        lts_rp_brr_characs,  # 63-72
        tau0_rp_brr,  # 73
        tau0_rp_brr_sd,  # 74
        beta_rp_brr,  # 75
        beta_rp_brr_sd,  # 76
        delta_rp_brr,  # 77
        delta_rp_brr_sd,  # 78
        lts_rp_brr_fit_goodness,  # 79-80
        # Fit region for the remain probability.
        fit_start_rp * time_conv,  # 81
        (fit_stop_rp - 1) * time_conv,  # 82
        # Method 7: Numerical integration of the Kaplan-Meier estimator.
        lts_km_int_characs,  # 83-92
        # Method 8: Weibull fit of the Kaplan-Meier estimator.
        lts_km_wbl_characs,  # 93-102
        tau0_km_wbl,  # 103
        tau0_km_wbl_sd,  # 104
        beta_km_wbl,  # 105
        beta_km_wbl_sd,  # 106
        lts_km_wbl_fit_goodness,  # 107-108
        # Method 9: Burr Type XII fit of the Kaplan-Meier estimator.
        lts_km_brr_characs,  # 109-118
        tau0_km_brr,  # 119
        tau0_km_brr_sd,  # 120
        beta_km_brr,  # 121
        beta_km_brr_sd,  # 122
        delta_km_brr,  # 123
        delta_km_brr_sd,  # 124
        lts_km_brr_fit_goodness,  # 125-126
        # Fit region for the Kaplan-Meier estimator.
        fit_start_km * time_conv,  # 127
        (fit_stop_km - 1) * time_conv,  # 128
    ]
)
header = (
    "Bin residence times (hereafter denoted state lifetimes).\n"
    + "Average time that a given compound stays in a given bin calculated\n"
    + "either directly from the discrete trajectory (Method 1-3) or from the\n"
    + "corresponding estimate of the survival function (Method 4-9).\n"
    + "\n"
    + "System:                   {:s}\n".format(args.system)
    + "Settings:                 {:s}\n".format(args.settings)
    + "Bin edges:                {:s}\n".format(infile_bins)
    + "Discrete trajectory:      {:s}\n".format(infile_dtrj)
    + "Autocorrelation function: {:s}\n".format(infile_rp)
    + "Kaplan-Meier estimator:   {:s}\n".format(infile_km)
    + "KM estimator variance:    {:s}\n".format(infile_km_var)
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
    + "Lifetimes are calculated using different methods:\n"
    + "\n"
    + "1) The average lifetime <t_cnt_cen> is calculated by counting how\n"
    + "   many frames a given compound stays in a given state including\n"
    + "   truncated states at the trajectory edges -> censored counting.\n"
    + "   Note that lifetimes calculated in this way are usually biased to\n"
    + "   lower values because of the limited length of the trajectory and\n"
    + "   because of truncation/censoring at the trajectory edges.\n"
    + "\n"
    + "2) The average lifetime <t_cnt_unc> is calculated by counting how\n"
    + "   many frames a given compound stays in a given state excluding\n"
    + "   truncated states at the trajectory edges -> uncensored counting.\n"
    + "   Note that lifetimes calculated in this way are usually biased to\n"
    + "   lower values because of the limited length of the trajectory.\n"
    + "   Uncensored counting might waste a significant amount of the\n"
    + "   trajectory.\n"
    + "\n"
    + "3) The average transition rate <k> is calculated as the number of\n"
    + "   transitions leading out of a given state divided by the number of\n"
    + "   frames that compounds have spent in this state.  The average\n"
    + "   lifetime <t_k> is calculated as the inverse transition rate:\n"
    + "     <t_k> = 1 / <k>\n"
    + "\n"
    + "4) The autocorrelation function (ACF) C(t) of the existence/lifetime\n"
    + "   operator is interpreted as the survival function (SF) of the\n"
    + "   underlying lifetime distribution.  Thus, the lifetime can be\n"
    + "   calculated according to the alternative expectation formula [1]:\n"
    + "     <t_int^n> = n * int_0^inf t^(n-1) C(t) dt\n"
    + "   If C(t) does not decay below the given threshold of\n"
    + "   {:.4f}, <t_acf_int^n> is set to NaN.\n".format(args.int_thresh)
    + "\n"
    + "5) The ACF C(t) is fitted by the SF of the Weibull distribution\n"
    + "   (stretched exponential):\n"
    + "     S_wbl(t) = exp[-(t/tau0_wbl)^beta_wbl]\n"
    + "   with tau0_wbl > 0 and beta_wbl > 0."
    + "   The average lifetime <t_acf_wbl^n> is calculated according to the\n"
    + "   alternative expectation formula [1]:\n"
    + "     <t_wbl^n> = n * int_0^inf t^(n-1) S_wbl(t) dt\n"
    + "               = tau0_wbl^n * Gamma(1 + n/beta_bwl)\n"
    + "   where Gamma(z) is the gamma function.\n"
    + "\n"
    + "6) The ACF C(t) is fitted by the SF of a Burr Type XII\n"
    + "   distribution:\n"
    + "     S_brr(t) = 1 / [1 + (t/tau0_brr)^beta_brr]^delta_brr\n"
    + "   with tau0_brr > 0, beta_brr > 0 and beta_brr*delta_brr > 1.\n"
    + "   The average lifetime <t_brr^n> is calculated according to the\n"
    + "   alternative expectation formula [1]:\n"
    + "     <t_brr^n> = n * int_0^inf t^(n-1) S_brr(t) dt\n"
    + "               = tau0_brr^n * Gamma(delta_brr - n/beta_brr) *\n"
    + "                 Gamma(1 + n/beta_brr) / Gamma(delta_brr)\n"
    + "   where Gamma(z) is the gamma function.\n"
    + "\n"
    + "7)-9) Like 4)-6) but instead of the ACF, the Kaplan-Meier estimate of\n"
    + "   the SF is used."
    + "\n"
    + "All fits are done using scipy.optimize.curve_fit with the 'Trust\n"
    + "Region Reflective' method.  The SF is always fitted until it decays\n"
    + "below the given threshold or until the given lag time is reached\n"
    + "(whatever happens earlier).\n"
    + "\n"
    + "int_thresh = {:.4f}\n".format(args.int_thresh)
    + "end_fit  = {}\n".format(args.end_fit)
    + "stop_fit = {:.4f}\n".format(args.stop_fit)
    + "Box edges:          {:>16.9e}, {:>16.9e} A\n".format(0, box_z)
    + "Electrode surfaces: {:>16.9e}, {:>16.9e} A\n".format(
        elctrd_thk, box_z - elctrd_thk
    )
    + "\n"
    + "Reference [1]:\n"
    + "  S. Chakraborti, F. Jardim, E. Epprecht,\n"
    + "  Higher-order moments using the survival function: The\n"
    + "  alternative expectation formula,\n"
    + "  The American Statistician, 2019, 73, 2, 191-194."
    + "\n"
    + "\n"
    + "The columns contain:\n"
    + "  1 State/Bin index (zero based)\n"
    + "  2 Lower bin edges / A\n"
    + "  3 Upper bin edges / A\n"
    + "  4 Distance of the lower bin edges to the left electrode surface / A\n"
    + "  5 Distance of the lower bin edges to the right electrode surface / A"
    + "\n"
    + "  6 Distance of the upper bin edges to the left electrode surface / A\n"
    + "  7 Distance of the upper bin edges to the right electrode surface / A"
    + "\n"
    + "\n"
    + "Methods based on counting frames:\n"
    + "  Method 1: Censored counting\n"
    + "  8 Sample mean (1st raw moment) / ns\n"
    + "  9 Uncertainty of the sample mean (standard error) / ns\n"
    + " 10 Corrected sample standard deviation / ns\n"
    + " 11 Corrected coefficient of variation\n"
    + " 12 Unbiased sample skewness (Fisher)\n"
    + " 13 Unbiased sample excess kurtosis (Fisher)\n"
    + " 14 Sample median / ns\n"
    + " 15 Non-parametric skewness\n"
    + " 16 2nd raw moment (biased estimate) / ns^2\n"
    + " 17 3rd raw moment (biased estimate) / ns^3\n"
    + " 18 4th raw moment (biased estimate) / ns^4\n"
    + " 19 Sample minimum / ns\n"
    + " 20 Sample maximum / ns\n"
    + " 21 Number of observations/samples\n"
    + "\n"
    + "  Method 2: Uncensored counting.\n"
    + " 22-35 As Method 1\n"
    + "\n"
    + "  Method 3: Inverse transition rate\n"
    + " 36 Mean lifetime / ns\n"
    + "\n"
    + "Methods based on the ACF:\n"
    + "  Method 4: Numerical integration of the ACF\n"
    + " 37 Mean lifetime (1st raw moment) / ns\n"
    + " 38 Standard deviation / ns\n"
    + " 39 Coefficient of variation"
    + " 40 Skewness (Fisher)\n"
    + " 41 Excess kurtosis (Fisher)\n"
    + " 42 Median / ns\n"
    + " 43 Non-parametric skewness\n"
    + " 44 2nd raw moment / ns^2\n"
    + " 45 3rd raw moment / ns^3\n"
    + " 46 4th raw moment / ns^4\n"
    + "\n"
    + "  Method 5: Weibull fit of the ACF\n"
    + " 47-56 As Method 4\n"
    + " 57 Fit parameter tau0_wbl / ns\n"
    + " 58 Standard deviation of tau0_wbl / ns\n"
    + " 59 Fit parameter beta_wbl\n"
    + " 60 Standard deviation of beta_wbl\n"
    + " 61 Coefficient of determination of the fit (R^2 value)\n"
    + " 62 Root-mean-square error (RMSE) of the fit\n"
    + "\n"
    + "  Method 6: Burr Type XII fit of the ACF\n"
    + " 63-76 As Method 5\n"
    + " 77 Fit parameter delta_brr\n"
    + " 78 Standard deviation of delta_brr\n"
    + " 79 Coefficient of determination of the fit (R^2 value)\n"
    + " 80 Root-mean-square error (RMSE) of the fit\n"
    + "\n"
    + "  Fit region for all ACF fitting methods\n"
    + " 81 Start of fit region (inclusive) / ns\n"
    + " 82 End of fit region (inclusive) / ns\n"
    + "\n"
    + "Methods based on the Kaplan-Meier estimator:\n"
    + "  Method 7: Numerical integration of the Kaplan-Meier estimator\n"
    + " 83-92 As Method 4\n"
    + "\n"
    + "  Method 8: Weibull fit of the Kaplan-Meier estimator\n"
    + " 93-108 As Method 5\n"
    + "\n"
    + "  Method 9: Burr Type XII fit of the Kaplan-Meier estimator\n"
    + " 109-126 As Method 6\n"
    + "\n"
    + "  Fit region for all Kaplan-Meier estimator fitting methods\n"
    + " 127 Start of fit region (inclusive) / ns\n"
    + " 128 End of fit region (inclusive) / ns\n"
    + "\n"
    + "Column number:\n"
)
header += "{:>14d}".format(1)
for i in range(2, data.shape[-1] + 1):
    header += " {:>16d}".format(i)
leap.io_handler.savetxt(outfile_txt, data, header=header)
print("Created {}".format(outfile_txt))


print("Creating plot(s)...")
elctrd_thk /= 10  # A -> nm.
box_z /= 10  # A -> nm.
bins /= 10  # A -> nm.
bin_mids = bins_up - (bins_up - bins_low) / 2
bin_mids /= 10  # A -> nm.

label_cnt_cen = "Cens."  # Count
label_cnt_unc = "Uncens."  # Count
label_k = "Rate"
label_rp_int = "ACF Num"
label_rp_wbl = "ACF Wbl"
label_rp_brr = "ACF Burr"
label_km_int = "KM Num"
label_km_wbl = "KM Wbl"
label_km_brr = "KM Burr"

color_cnt_cen = "tab:orange"
color_cnt_unc = "tab:red"
color_k = "tab:brown"
color_rp_int = "tab:purple"
color_rp_wbl = "tab:blue"
color_rp_brr = "tab:cyan"
color_km_int = "goldenrod"
color_km_wbl = "gold"
color_km_brr = "yellow"

marker_cnt_cen = "H"
marker_cnt_unc = "h"
marker_k = "p"
marker_rp_int = "^"
marker_rp_wbl = ">"
marker_rp_brr = "<"
marker_km_int = "s"
marker_km_wbl = "D"
marker_km_brr = "d"

color_exp = "tab:green"
linestyle_exp = "dashed"
label_exp = "Exp. Dist."

ylabel_acf = "Autocorrelation Function"
ylabel_km = "Kaplan-Meier Estimate"

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
legend_bbox = (0.58, 1.01)
legend_loc = "upper center"
height_ratios = (0.2, 1)
cmap = plt.get_cmap()
c_vals = np.arange(n_states)
c_norm = max(1, n_states - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile_pdf)
with PdfPages(outfile_pdf) as pdf:
    # Plot distribution characteristics vs. bins.
    ylabels = (
        "Residence Time / ns",
        "Std. Dev. / ns",
        "Coeff. of Variation",
        "Skewness",
        "Excess Kurtosis",
        "Median / ns",
        "Non-Parametric Skewness",
    )
    for i, ylabel in enumerate(ylabels):
        if i == 0:
            offset_i_cnt = 0
        else:
            offset_i_cnt = 1
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
        if ylabel == "Coeff. of Variation":
            y_exp = 1  # Coeff. of variation of exp. distribution.
        elif ylabel == "Skewness":
            y_exp = 2  # Skewness of exponential distribution.
        elif ylabel == "Excess Kurtosis":
            y_exp = 6  # Excess kurtosis of exponential distribution.
        elif ylabel == "Non-Parametric Skewness":
            y_exp = 1 - np.log(2)  # Non-param. skew. of exp. dist.
        else:
            y_exp = None
        if y_exp is not None:
            ax.axhline(
                y=y_exp,
                color=color_exp,
                linestyle=linestyle_exp,
                label=label_exp,
            )
        # Method 1: Censored counting.
        ax.errorbar(
            bin_mids,
            lts_cnt_cen_characs[:, i + offset_i_cnt],
            yerr=lts_cnt_cen_characs[:, i + 1] if i == 0 else None,
            label=label_cnt_cen,
            color=color_cnt_cen,
            marker=marker_cnt_cen,
            alpha=leap.plot.ALPHA,
        )
        # Method 2: Uncensored counting.
        ax.errorbar(
            bin_mids,
            lts_cnt_unc_characs[:, i + offset_i_cnt],
            yerr=lts_cnt_unc_characs[:, i + 1] if i == 0 else None,
            label=label_cnt_unc,
            color=color_cnt_unc,
            marker=marker_cnt_unc,
            alpha=leap.plot.ALPHA,
        )
        if i == 0:
            # Method 3: Inverse transition rate.
            ax.errorbar(
                bin_mids,
                lts_k,
                yerr=None,
                label=label_k,
                color=color_k,
                marker=marker_k,
                alpha=leap.plot.ALPHA,
            )
        # Method 4: Numerical integration of the remain probability.
        ax.errorbar(
            bin_mids,
            lts_rp_int_characs[:, i],
            yerr=None,
            label=label_rp_int,
            color=color_rp_int,
            marker=marker_rp_int,
            alpha=leap.plot.ALPHA,
        )
        # Method 5: Weibull fit of the remain probability.
        ax.errorbar(
            bin_mids,
            lts_rp_wbl_characs[:, i],
            yerr=None,
            label=label_rp_wbl,
            color=color_rp_wbl,
            marker=marker_rp_wbl,
            alpha=leap.plot.ALPHA,
        )
        # Method 6: Burr Type XII fit of the remain probability.
        ax.errorbar(
            bin_mids,
            lts_rp_brr_characs[:, i],
            yerr=None,
            label=label_rp_brr,
            color=color_rp_brr,
            marker=marker_rp_brr,
            alpha=leap.plot.ALPHA,
        )
        # Method 7: Numerical integration of the KM estimator.
        ax.errorbar(
            bin_mids,
            lts_km_int_characs[:, i],
            yerr=None,
            label=label_km_int,
            color=color_km_int,
            marker=marker_km_int,
            alpha=leap.plot.ALPHA,
        )
        # Method 8: Weibull fit of the Kaplan-Meier estimator.
        ax.errorbar(
            bin_mids,
            lts_km_wbl_characs[:, i],
            yerr=None,
            label=label_km_wbl,
            color=color_km_wbl,
            marker=marker_km_wbl,
            alpha=leap.plot.ALPHA,
        )
        # Method 9: Burr Type XII fit of the Kaplan-Meier estimator.
        ax.errorbar(
            bin_mids,
            lts_km_brr_characs[:, i],
            yerr=None,
            label=label_km_brr,
            color=color_km_brr,
            marker=marker_km_brr,
            alpha=leap.plot.ALPHA,
        )
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax.get_ylim()
        if ylim[0] < 0 and ylabel not in (
            "Skewness",
            "Excess Kurtosis",
            "Non-Parametric Skewness",
        ):
            ax.set_ylim(0, ylim[1])
        leap.plot.bins(ax, bins=bins)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend = fig.legend(
            title=legend_title,
            ncol=3,
            bbox_to_anchor=legend_bbox,
            loc=legend_loc,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
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

    # Plot min, max and number of samples for count methods.
    ylabels = (
        "Min. Residence Time / ns",
        "Max. Residence Time / ns",
        "No. of Samples",
    )
    for i, ylabel in enumerate(ylabels):
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
        # Method 1: Censored counting.
        ax.plot(
            bin_mids,
            lts_cnt_cen_characs[:, 11 + i],
            label=label_cnt_cen,
            color=color_cnt_cen,
            marker=marker_cnt_cen,
            alpha=leap.plot.ALPHA,
        )
        # Method 2: Uncensored counting.
        ax.plot(
            bin_mids,
            lts_cnt_unc_characs[:, 11 + i],
            label=label_cnt_unc,
            color=color_cnt_unc,
            marker=marker_cnt_unc,
            alpha=leap.plot.ALPHA,
        )
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax.get_ylim()
        if ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        leap.plot.bins(ax, bins=bins)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend = ax.legend(
            title=legend_title,
            ncol=2,
            loc="lower center" if i == 2 else "upper center",
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
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

    # Plot fit parameters tau0, beta and delta.
    ylabels = (
        r"Fit Parameter $\tau_0$ / ns",
        r"Fit Parameter $\beta$",
        r"Fit Parameter $\delta$",
    )
    for i, ylabel in enumerate(ylabels):
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
        if i < 2:
            # Method 5: Weibull fit of the remain probability.
            ax.errorbar(
                bin_mids,
                popt_rp_wbl[:, i],
                yerr=perr_rp_wbl[:, i],
                label=label_rp_wbl,
                color=color_rp_wbl,
                marker=marker_rp_wbl,
                alpha=leap.plot.ALPHA,
            )
        # Method 6: Burr Type XII fit of the remain probability.
        ax.errorbar(
            bin_mids,
            popt_conv_rp_brr[:, i],
            yerr=perr_conv_rp_brr[:, i],
            label=label_rp_brr,
            color=color_rp_brr,
            marker=marker_rp_brr,
            alpha=leap.plot.ALPHA,
        )
        if i < 2:
            # Method 8: Weibull fit of the Kaplan-Meier estimator.
            ax.errorbar(
                bin_mids,
                popt_km_wbl[:, i],
                yerr=perr_km_wbl[:, i],
                label=label_km_wbl,
                color=color_km_wbl,
                marker=marker_km_wbl,
                alpha=leap.plot.ALPHA,
            )
        # Method 9: Burr Type XII fit of the Kaplan-Meier estimator.
        ax.errorbar(
            bin_mids,
            popt_conv_km_brr[:, i],
            yerr=perr_conv_km_brr[:, i],
            label=label_km_brr,
            color=color_km_brr,
            marker=marker_km_brr,
            alpha=leap.plot.ALPHA,
        )
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax.get_ylim()
        if ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        leap.plot.bins(ax, bins=bins)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend = fig.legend(
            title=legend_title,
            ncol=2,
            bbox_to_anchor=legend_bbox,
            loc=legend_loc,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
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

    # Plot goodness of fit quantities.
    ylabels = (r"Coeff. of Determ. $R^2$", "RMSE")
    for i, ylabel in enumerate(ylabels):
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
        # Method 5: Weibull fit of the remain probability.
        ax.plot(
            bin_mids,
            lts_rp_wbl_fit_goodness[:, i],
            label=label_rp_wbl,
            color=color_rp_wbl,
            marker=marker_rp_wbl,
            alpha=leap.plot.ALPHA,
        )
        # Method 6: Burr Type XII fit of the remain probability.
        ax.plot(
            bin_mids,
            lts_rp_brr_fit_goodness[:, i],
            label=label_rp_brr,
            color=color_rp_brr,
            marker=marker_rp_brr,
            alpha=leap.plot.ALPHA,
        )
        # Method 8: Weibull fit of the Kaplan-Meier estimator.
        ax.plot(
            bin_mids,
            lts_km_wbl_fit_goodness[:, i],
            label=label_km_wbl,
            color=color_km_wbl,
            marker=marker_km_wbl,
            alpha=leap.plot.ALPHA,
        )
        # Method 9: Burr Type XII fit of the Kaplan-Meier estimator.
        ax.plot(
            bin_mids,
            lts_km_brr_fit_goodness[:, i],
            label=label_km_brr,
            color=color_km_brr,
            marker=marker_km_brr,
            alpha=leap.plot.ALPHA,
        )
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax.get_ylim()
        if ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        leap.plot.bins(ax, bins=bins)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend = fig.legend(
            title=legend_title,
            ncol=2,
            bbox_to_anchor=legend_bbox,
            loc=legend_loc,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
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

    # Plot end of fit region.
    ylabel = "End of Fit Region / ns"
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
    # Fit of remain probability.
    ax.plot(
        bin_mids,
        (fit_stop_rp - 1) * time_conv,
        label="ACF",
        color=color_rp_wbl,
        marker=marker_rp_wbl,
    )
    # Fit of Kaplan-Meier estimator.
    ax.plot(
        bin_mids,
        (fit_stop_km - 1) * time_conv,
        label="KM",
        color=color_km_wbl,
        marker=marker_km_wbl,
    )
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    leap.plot.bins(ax, bins=bins)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend = fig.legend(
        title=legend_title,
        ncol=2,
        bbox_to_anchor=legend_bbox,
        loc=legend_loc,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
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

    # Plot remain probabilities and Weibull fits for each state.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, rp in enumerate(remain_probs.T):
        times_fit = times[fit_start_rp[i] : fit_stop_rp[i]]
        fit = mdt.func.kww(times_fit, *popt_rp_wbl[i])
        lines = ax.plot(
            times,
            rp,
            label=r"$%d$" % (states[i] + 1),
            linewidth=1,
            alpha=leap.plot.ALPHA,
        )
        ax.plot(
            times_fit,
            fit,
            label="Wbl Fit" if i == len(remain_probs.T) - 1 else None,
            linestyle="dashed",
            color=lines[0].get_color(),
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Time / ns",
        ylabel=ylabel_acf,
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
    # Plot Weibull fit residuals (remain probability) for each state.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, rp in enumerate(remain_probs.T):
        times_fit = times[fit_start_rp[i] : fit_stop_rp[i]]
        fit = mdt.func.kww(times_fit, *popt_rp_wbl[i])
        res = rp[fit_start_rp[i] : fit_stop_rp[i]] - fit
        ax.plot(
            times_fit,
            res,
            label=r"$%d$" % (states[i] + 1),
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Time / ns",
        ylabel="ACF Weibull Fit Res.",
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

    # Plot Kaplan-Meier estimates and Weibull fits for each state.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, km in enumerate(km_surv_funcs.T):
        times_fit = times[fit_start_km[i] : fit_stop_km[i]]
        fit = mdt.func.kww(times_fit, *popt_km_wbl[i])
        lines = ax.plot(
            times,
            km,
            label=r"$%d$" % (states[i] + 1),
            linewidth=1,
            alpha=leap.plot.ALPHA,
        )
        ax.plot(
            times_fit,
            fit,
            label="Wbl Fit" if i == len(km_surv_funcs.T) - 1 else None,
            linestyle="dashed",
            color=lines[0].get_color(),
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Time / ns",
        ylabel=ylabel_km,
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
    # Plot Weibull fit residuals (Kaplan-Meier) for each state.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, km in enumerate(km_surv_funcs.T):
        times_fit = times[fit_start_km[i] : fit_stop_km[i]]
        fit = mdt.func.kww(times_fit, *popt_km_wbl[i])
        res = km[fit_start_km[i] : fit_stop_km[i]] - fit
        ax.plot(
            times_fit,
            res,
            label=r"$%d$" % (states[i] + 1),
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Time / ns",
        ylabel="KM Weibull Fit Res.",
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

    # Plot remain probabilities and Burr fits for each state.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, rp in enumerate(remain_probs.T):
        times_fit = times[fit_start_rp[i] : fit_stop_rp[i]]
        fit = mdt.func.burr12_sf_alt(times_fit, *popt_rp_brr[i])
        lines = ax.plot(
            times,
            rp,
            label=r"$%d$" % (states[i] + 1),
            linewidth=1,
            alpha=leap.plot.ALPHA,
        )
        ax.plot(
            times_fit,
            fit,
            label="Burr Fit" if i == len(remain_probs.T) - 1 else None,
            linestyle="dashed",
            color=lines[0].get_color(),
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Time / ns",
        ylabel=ylabel_acf,
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
    # Plot Burr fit residuals (remain probability) for each state.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, rp in enumerate(remain_probs.T):
        times_fit = times[fit_start_rp[i] : fit_stop_rp[i]]
        fit = mdt.func.burr12_sf_alt(times_fit, *popt_rp_brr[i])
        res = rp[fit_start_rp[i] : fit_stop_rp[i]] - fit
        ax.plot(
            times_fit,
            res,
            label=r"$%d$" % (states[i] + 1),
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Time / ns",
        ylabel="ACF Burr Fit Res.",
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

    # Plot Kaplan-Meier estimates and Burr fits for each state.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, km in enumerate(km_surv_funcs.T):
        times_fit = times[fit_start_km[i] : fit_stop_km[i]]
        fit = mdt.func.burr12_sf_alt(times_fit, *popt_km_brr[i])
        lines = ax.plot(
            times,
            km,
            label=r"$%d$" % (states[i] + 1),
            linewidth=1,
            alpha=leap.plot.ALPHA,
        )
        ax.plot(
            times_fit,
            fit,
            label="Burr Fit" if i == len(km_surv_funcs.T) - 1 else None,
            linestyle="dashed",
            color=lines[0].get_color(),
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Time / ns",
        ylabel=ylabel_km,
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
    # Plot Burr fit residuals (Kaplan-Meier) for each state.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, km in enumerate(km_surv_funcs.T):
        times_fit = times[fit_start_km[i] : fit_stop_km[i]]
        fit = mdt.func.burr12_sf_alt(times_fit, *popt_km_brr[i])
        res = km[fit_start_km[i] : fit_stop_km[i]] - fit
        ax.plot(
            times_fit,
            res,
            label=r"$%d$" % (states[i] + 1),
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Time / ns",
        ylabel="KM Burr Fit Res.",
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
