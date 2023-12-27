#!/usr/bin/env python3


"""
Calculate and plot the renewal time for a given pair of compounds as
function of the PEO chain length.

Calculate the time until the PEO chain (or more generally the ligand)
that was bound the longest to the central ion detaches form the central
ion.  The renewal time is calculated directly from the discrete
trajectory (Method 1-3), from the corresponding remain probability
function (Method 4-6) of from the corresponding Kaplan-Meier estimate of
the survival function (Method 7-9).

See
:file:`scripts/bulk_and_walls/mdt/discrete-z/get_and_plot_bin_lifetimes.py`
for more details about the calculation methods.
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

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Calculate and plot the renewal time for a given pair of compounds as"
        " function of the PEO chain length."
    )
)
parser.add_argument(
    "--cmp",
    type=str,
    required=True,
    choices=("Li-ether", "Li-NTf2"),
    help="Compounds for which to calculate the renewal times.",
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
        "Only calculate the renewal time by directly integrating the remain"
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
cmp1, cmp2 = args.cmp.split("-")

if args.continuous:
    con = "_continuous"
else:
    con = ""

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "renewal_events"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
ana_path = os.path.join(analysis, analysis + analysis_suffix)
tool = "mdt"  # Analysis software.
outfile_base = (  # Output file name.
    settings + "_lintf2_peoN_20-1_sc80_" + analysis + analysis_suffix + con
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
sys_pat = "lintf2_[gp]*[0-9]*_20-1_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="O_per_chain"
)


print("Calculating renewal times directly from `dtrj`...")
lts_cnt_cen_characs = np.full(
    (Sims.n_sims, 10 + n_moms), np.nan, dtype=np.float32
)
lts_cnt_unc_characs = np.full_like(lts_cnt_cen_characs, np.nan)
lts_k = np.full(Sims.n_sims, np.nan, dtype=np.float32)
for sim_ix, Sim in enumerate(Sims.sims):
    # Read discrete trajectory.
    file_suffix = analysis + analysis_suffix + "_dtrj.npz"
    try:
        infile = leap.simulation.get_ana_file(Sim, ana_path, tool, file_suffix)
    except FileNotFoundError:
        continue
    dtrj = mdt.fh.load_dtrj(infile)
    # Method 1: Censored counting.
    lts_cnt_cen_characs[sim_ix] = leap.lifetimes.count_method_state_average(
        dtrj,
        n_moms=n_moms,
        time_conv=time_conv,
        uncensored=False,
        discard_neg_start=True,
    )
    # Method 2: Uncensored counting.
    lts_cnt_unc_characs[sim_ix] = leap.lifetimes.count_method_state_average(
        dtrj,
        n_moms=n_moms,
        time_conv=time_conv,
        uncensored=True,
        discard_neg_start=True,
    )
    # Method 3: Calculate the transition rate as the number of
    # transitions leading out of a given state divided by the number of
    # frames that compounds have spent in this state.  The average
    # renewal time is calculated as the inverse transition rate.
    rate = mdt.dtrj.trans_rate_tot(dtrj, discard_neg_start=True)
    lts_k[sim_ix] = time_conv / rate
del dtrj, rate


print("Calculating renewal times from the remain probability...")
times_prev = None
remain_probs = [None for Sim in Sims.sims]

lts_rp_int_characs = np.full(
    (Sims.n_sims, 6 + n_moms), np.nan, dtype=np.float32
)

fit_start_rp = np.zeros(Sims.n_sims, dtype=np.uint32)
fit_stop_rp = np.zeros_like(fit_start_rp)

lts_rp_wbl_characs = np.full_like(lts_rp_int_characs, np.nan)
lts_rp_wbl_fit_goodness = np.full((Sims.n_sims, 2), np.nan, dtype=np.float32)
popt_rp_wbl = np.full((Sims.n_sims, 2), np.nan, dtype=np.float32)
perr_rp_wbl = np.full_like(popt_rp_wbl, np.nan)
tau0_rp_wbl = np.full(Sims.n_sims, np.nan, dtype=np.float32)
tau0_rp_wbl_sd = np.full_like(tau0_rp_wbl, np.nan)
beta_rp_wbl = np.full_like(tau0_rp_wbl, np.nan)
beta_rp_wbl_sd = np.full_like(tau0_rp_wbl, np.nan)

lts_rp_brr_characs = np.full_like(lts_rp_int_characs, np.nan)
lts_rp_brr_fit_goodness = np.full_like(lts_rp_wbl_fit_goodness, np.nan)
popt_rp_brr = np.full((Sims.n_sims, 3), np.nan, dtype=np.float32)
perr_rp_brr = np.full_like(popt_rp_brr, np.nan)
popt_conv_rp_brr = np.full_like(popt_rp_brr, np.nan)
perr_conv_rp_brr = np.full_like(popt_rp_brr, np.nan)
tau0_rp_brr = np.full(Sims.n_sims, np.nan, dtype=np.float32)
tau0_rp_brr_sd = np.full_like(tau0_rp_brr, np.nan)
beta_rp_brr = np.full_like(tau0_rp_brr, np.nan)
beta_rp_brr_sd = np.full_like(tau0_rp_brr, np.nan)
delta_rp_brr = np.full_like(tau0_rp_brr, np.nan)
delta_rp_brr_sd = np.full_like(tau0_rp_brr, np.nan)

for sim_ix, Sim in enumerate(Sims.sims):
    # Read remain probability.
    file_suffix = (
        analysis
        + analysis_suffix
        + "_state_lifetime_discard-neg-start"
        + con
        + ".txt.gz"
    )
    try:
        infile = leap.simulation.get_ana_file(Sim, ana_path, tool, file_suffix)
    except FileNotFoundError:
        continue
    times, remain_prob_sim = np.loadtxt(
        infile, usecols=(0, 1), unpack=True, dtype=np.float32
    )
    if times_prev is not None and not np.array_equal(times, times_prev):
        # Saving the lag times for all simulations would be quite memory
        # consuming.  Because in my case all simulations should have the
        # same lag times, I am just saving one lag time array.
        raise ValueError("The lag times of the different simulations differ")
    times_prev = np.copy(times)
    times *= time_conv
    remain_probs[sim_ix] = remain_prob_sim
    # Add an additional dimension to `remain_prob_sim`, because the used
    # functions from `lintf2_ether_ana_postproc.lifetimes` expect an
    # array of survival functions.
    remain_prob_sim = remain_prob_sim.reshape(remain_prob_sim.shape + (1,))
    # Method 4: Numerical integration of the remain probability.
    lts_rp_int_characs_sim = leap.lifetimes.integral_method(
        remain_prob_sim, times, n_moms=n_moms, int_thresh=args.int_thresh
    )
    lts_rp_int_characs[sim_ix] = np.squeeze(lts_rp_int_characs_sim)
    # Get fit region for fitting methods.
    fit_start_rp_sim, fit_stop_rp_sim = leap.lifetimes.get_fit_region(
        remain_prob_sim, times, end_fit=args.end_fit, stop_fit=args.stop_fit
    )
    fit_start_rp[sim_ix] = fit_start_rp_sim[0]
    fit_stop_rp[sim_ix] = fit_stop_rp_sim[0]
    # Method 5: Weibull fit of the remain probability.
    (
        lts_rp_wbl_characs_sim,
        lts_rp_wbl_fit_goodness_sim,
        popt_rp_wbl_sim,
        perr_rp_wbl_sim,
    ) = leap.lifetimes.weibull_fit_method(
        remain_prob_sim,
        times,
        fit_start=fit_start_rp_sim,
        fit_stop=fit_stop_rp_sim,
        n_moms=n_moms,
        fit_method=fit_method,
    )
    lts_rp_wbl_characs[sim_ix] = np.squeeze(lts_rp_wbl_characs_sim)
    lts_rp_wbl_fit_goodness[sim_ix] = np.squeeze(lts_rp_wbl_fit_goodness_sim)
    popt_rp_wbl[sim_ix] = np.squeeze(popt_rp_wbl_sim)
    perr_rp_wbl[sim_ix] = np.squeeze(perr_rp_wbl_sim)
    tau0_rp_wbl[sim_ix], beta_rp_wbl[sim_ix] = popt_rp_wbl[sim_ix].T
    tau0_rp_wbl_sd[sim_ix], beta_rp_wbl_sd[sim_ix] = perr_rp_wbl[sim_ix].T
    # Method 6: Burr Type XII fit of the remain probability.
    (
        lts_rp_brr_characs_sim,
        lts_rp_brr_fit_goodness_sim,
        popt_rp_brr_sim,
        perr_rp_brr_sim,
        popt_conv_rp_brr_sim,
        perr_conv_rp_brr_sim,
    ) = leap.lifetimes.burr12_fit_method(
        remain_prob_sim,
        times,
        fit_start=fit_start_rp_sim,
        fit_stop=fit_stop_rp_sim,
        n_moms=n_moms,
        fit_method=fit_method,
    )
    lts_rp_brr_characs[sim_ix] = np.squeeze(lts_rp_brr_characs_sim)
    lts_rp_brr_fit_goodness[sim_ix] = np.squeeze(lts_rp_brr_fit_goodness_sim)
    popt_rp_brr[sim_ix] = np.squeeze(popt_rp_brr_sim)
    perr_rp_brr[sim_ix] = np.squeeze(perr_rp_brr_sim)
    popt_conv_rp_brr[sim_ix] = np.squeeze(popt_conv_rp_brr_sim)
    perr_conv_rp_brr[sim_ix] = np.squeeze(perr_conv_rp_brr_sim)
    (
        tau0_rp_brr[sim_ix],
        beta_rp_brr[sim_ix],
        delta_rp_brr[sim_ix],
    ) = popt_conv_rp_brr[sim_ix].T
    (
        tau0_rp_brr_sd[sim_ix],
        beta_rp_brr_sd[sim_ix],
        delta_rp_brr_sd[sim_ix],
    ) = perr_conv_rp_brr[sim_ix].T
del remain_prob_sim
del lts_rp_int_characs_sim
del fit_start_rp_sim, fit_stop_rp_sim
del (
    lts_rp_wbl_characs_sim,
    lts_rp_wbl_fit_goodness_sim,
    popt_rp_wbl_sim,
    perr_rp_wbl_sim,
)
del (
    lts_rp_brr_characs_sim,
    lts_rp_brr_fit_goodness_sim,
    popt_rp_brr_sim,
    perr_rp_brr_sim,
    popt_conv_rp_brr_sim,
    perr_conv_rp_brr_sim,
)


print("Calculating renewal times from the Kaplan-Meier estimator...")
times_prev = None
km_surv_funcs = [None for Sim in Sims.sims]

lts_km_int_characs = np.full(
    (Sims.n_sims, 6 + n_moms), np.nan, dtype=np.float32
)

fit_start_km = np.zeros(Sims.n_sims, dtype=np.uint32)
fit_stop_km = np.zeros_like(fit_start_km)

lts_km_wbl_characs = np.full_like(lts_km_int_characs, np.nan)
lts_km_wbl_fit_goodness = np.full((Sims.n_sims, 2), np.nan, dtype=np.float32)
popt_km_wbl = np.full((Sims.n_sims, 2), np.nan, dtype=np.float32)
perr_km_wbl = np.full_like(popt_km_wbl, np.nan)
tau0_km_wbl = np.full(Sims.n_sims, np.nan, dtype=np.float32)
tau0_km_wbl_sd = np.full_like(tau0_km_wbl, np.nan)
beta_km_wbl = np.full_like(tau0_km_wbl, np.nan)
beta_km_wbl_sd = np.full_like(tau0_km_wbl, np.nan)

lts_km_brr_characs = np.full_like(lts_km_int_characs, np.nan)
lts_km_brr_fit_goodness = np.full_like(lts_km_wbl_fit_goodness, np.nan)
popt_km_brr = np.full((Sims.n_sims, 3), np.nan, dtype=np.float32)
perr_km_brr = np.full_like(popt_km_brr, np.nan)
popt_conv_km_brr = np.full_like(popt_km_brr, np.nan)
perr_conv_km_brr = np.full_like(popt_km_brr, np.nan)
tau0_km_brr = np.full(Sims.n_sims, np.nan, dtype=np.float32)
tau0_km_brr_sd = np.full_like(tau0_km_brr, np.nan)
beta_km_brr = np.full_like(tau0_km_brr, np.nan)
beta_km_brr_sd = np.full_like(tau0_km_brr, np.nan)
delta_km_brr = np.full_like(tau0_km_brr, np.nan)
delta_km_brr_sd = np.full_like(tau0_km_brr, np.nan)

for sim_ix, Sim in enumerate(Sims.sims):
    # Read Kaplan-Meier survival function.
    file_suffix = (
        analysis + analysis_suffix + "_kaplan_meier_discard-neg-start.txt.gz"
    )
    try:
        infile = leap.simulation.get_ana_file(Sim, ana_path, tool, file_suffix)
    except FileNotFoundError:
        continue
    times, km_surv_func_sim, km_surv_func_var_sim = np.loadtxt(
        infile, unpack=True, dtype=np.float32
    )
    if times_prev is not None and not np.array_equal(times, times_prev):
        # Saving the lag times for all simulations would be quite memory
        # consuming.  Because in my case all simulations should have the
        # same lag times, I am just saving one lag time array.
        raise ValueError("The lag times of the different simulations differ")
    times_prev = np.copy(times)
    times *= time_conv
    km_surv_funcs[sim_ix] = km_surv_func_sim
    # Add an additional dimension to `km_surv_func_sim`, because the
    # used functions from `lintf2_ether_ana_postproc.lifetimes` expect
    # an array of survival functions.
    km_surv_func_sim = km_surv_func_sim.reshape(km_surv_func_sim.shape + (1,))
    km_surv_func_var_sim = km_surv_func_var_sim.reshape(
        km_surv_func_var_sim.shape + (1,)
    )
    # Method 7: Numerical integration of the Kaplan-Meier estimator.
    lts_km_int_characs_sim = leap.lifetimes.integral_method(
        km_surv_func_sim, times, n_moms=n_moms, int_thresh=args.int_thresh
    )
    lts_km_int_characs[sim_ix] = np.squeeze(lts_km_int_characs_sim)
    # Get fit region for fitting methods.
    fit_start_km_sim, fit_stop_km_sim = leap.lifetimes.get_fit_region(
        km_surv_func_sim, times, end_fit=args.end_fit, stop_fit=args.stop_fit
    )
    fit_start_km[sim_ix] = fit_start_km_sim[0]
    fit_stop_km[sim_ix] = fit_stop_km_sim[0]
    # Method 8: Weibull fit of the Kaplan-Meier estimator.
    (
        lts_km_wbl_characs_sim,
        lts_km_wbl_fit_goodness_sim,
        popt_km_wbl_sim,
        perr_km_wbl_sim,
    ) = leap.lifetimes.weibull_fit_method(
        km_surv_func_sim,
        times,
        fit_start=fit_start_km_sim,
        fit_stop=fit_stop_km_sim,
        surv_funcs_var=km_surv_func_var_sim,
        n_moms=n_moms,
        fit_method=fit_method,
    )
    lts_km_wbl_characs[sim_ix] = np.squeeze(lts_km_wbl_characs_sim)
    lts_km_wbl_fit_goodness[sim_ix] = np.squeeze(lts_km_wbl_fit_goodness_sim)
    popt_km_wbl[sim_ix] = np.squeeze(popt_km_wbl_sim)
    perr_km_wbl[sim_ix] = np.squeeze(perr_km_wbl_sim)
    tau0_km_wbl[sim_ix], beta_km_wbl[sim_ix] = popt_km_wbl[sim_ix].T
    tau0_km_wbl_sd[sim_ix], beta_km_wbl_sd[sim_ix] = perr_km_wbl[sim_ix].T
    # Method 9: Burr Type XII fit of the Kaplan-Meier estimator.
    (
        lts_km_brr_characs_sim,
        lts_km_brr_fit_goodness_sim,
        popt_km_brr_sim,
        perr_km_brr_sim,
        popt_conv_km_brr_sim,
        perr_conv_km_brr_sim,
    ) = leap.lifetimes.burr12_fit_method(
        km_surv_func_sim,
        times,
        fit_start=fit_start_km_sim,
        fit_stop=fit_stop_km_sim,
        surv_funcs_var=km_surv_func_var_sim,
        n_moms=n_moms,
        fit_method=fit_method,
    )
    lts_km_brr_characs[sim_ix] = np.squeeze(lts_km_brr_characs_sim)
    lts_km_brr_fit_goodness[sim_ix] = np.squeeze(lts_km_brr_fit_goodness_sim)
    popt_km_brr[sim_ix] = np.squeeze(popt_km_brr_sim)
    perr_km_brr[sim_ix] = np.squeeze(perr_km_brr_sim)
    popt_conv_km_brr[sim_ix] = np.squeeze(popt_conv_km_brr_sim)
    perr_conv_km_brr[sim_ix] = np.squeeze(perr_conv_km_brr_sim)
    (
        tau0_km_brr[sim_ix],
        beta_km_brr[sim_ix],
        delta_km_brr[sim_ix],
    ) = popt_conv_km_brr[sim_ix].T
    (
        tau0_km_brr_sd[sim_ix],
        beta_km_brr_sd[sim_ix],
        delta_km_brr_sd[sim_ix],
    ) = perr_conv_km_brr[sim_ix].T
del km_surv_func_sim, km_surv_func_var_sim
del lts_km_int_characs_sim
del fit_start_km_sim, fit_stop_km_sim
del (
    lts_km_wbl_characs_sim,
    lts_km_wbl_fit_goodness_sim,
    popt_km_wbl_sim,
    perr_km_wbl_sim,
)
del (
    lts_km_brr_characs_sim,
    lts_km_brr_fit_goodness_sim,
    popt_km_brr_sim,
    perr_km_brr_sim,
    popt_conv_km_brr_sim,
    perr_conv_km_brr_sim,
)


print("Creating output file(s)...")
xdata = Sims.O_per_chain
data = np.column_stack(
    [
        xdata,  # 1
        # Method 1: Censored counting.
        lts_cnt_cen_characs,  # 2-15
        # Method 2: Uncensored counting.
        lts_cnt_unc_characs,  # 16-29
        # Method 3: Inverse transition rate.
        lts_k,  # 30
        # Method 4: Numerical integration of the remain probability.
        lts_rp_int_characs,  # 31-40
        # Method 5: Weibull fit of the remain probability.
        lts_rp_wbl_characs,  # 41-50
        tau0_rp_wbl,  # 51
        tau0_rp_wbl_sd,  # 52
        beta_rp_wbl,  # 53
        beta_rp_wbl_sd,  # 54
        lts_rp_wbl_fit_goodness,  # 55-56
        # Method 6: Burr Type XII fit of the remain probability.
        lts_rp_brr_characs,  # 57-66
        tau0_rp_brr,  # 67
        tau0_rp_brr_sd,  # 68
        beta_rp_brr,  # 69
        beta_rp_brr_sd,  # 70
        delta_rp_brr,  # 71
        delta_rp_brr_sd,  # 72
        lts_rp_brr_fit_goodness,  # 73-74
        # Fit region for the remain probability.
        fit_start_rp * time_conv,  # 75
        (fit_stop_rp - 1) * time_conv,  # 76
        # Method 7: Numerical integration of the Kaplan-Meier estimator.
        lts_km_int_characs,  # 77-86
        # Method 8: Weibull fit of the Kaplan-Meier estimator.
        lts_km_wbl_characs,  # 87-96
        tau0_km_wbl,  # 97
        tau0_km_wbl_sd,  # 98
        beta_km_wbl,  # 99
        beta_km_wbl_sd,  # 100
        lts_km_wbl_fit_goodness,  # 101-102
        # Method 9: Burr Type XII fit of the Kaplan-Meier estimator.
        lts_km_brr_characs,  # 103-112
        tau0_km_brr,  # 113
        tau0_km_brr_sd,  # 114
        beta_km_brr,  # 115
        beta_km_brr_sd,  # 116
        delta_km_brr,  # 117
        delta_km_brr_sd,  # 118
        lts_km_brr_fit_goodness,  # 119-120
        # Fit region for the Kaplan-Meier estimator.
        fit_start_km * time_conv,  # 121
        (fit_stop_km - 1) * time_conv,  # 122
    ]
)
header = (
    "{} renewal times.\n".format(args.cmp)
    + "The renewal time is the average time after which the selection\n"
    + "compound ('{}') that was continuously bound the longest to\n".format(
        cmp2
    )
    + "a reference compound ('{}') dissociates from it.\n".format(cmp1)
    + "\n"
    + "Lithium-to-ether-oxygen ratio: {:.2f}\n".format(Sims.Li_O_ratios[0])
    + "int_thresh = {:.4f}\n".format(args.int_thresh)
    + "end_fit  = {}\n".format(args.end_fit)
    + "stop_fit = {:.4f}\n".format(args.stop_fit)
    + "\n"
    + "\n"
    + "The columns contain:\n"
    + "  1 Number of ether oxygens per PEO chain\n"
    + "\n"
    + "Methods based on counting frames:\n"
    + "  Method 1: Censored counting\n"
    + "  2 Sample mean (1st raw moment) / ns\n"
    + "  3 Uncertainty of the sample mean (standard error) / ns\n"
    + "  4 Corrected sample standard deviation / ns\n"
    + "  5 Corrected coefficient of variation\n"
    + "  6 Unbiased sample skewness (Fisher)\n"
    + "  7 Unbiased sample excess kurtosis (Fisher)\n"
    + "  8 Sample median / ns\n"
    + "  9 Non-parametric skewness\n"
    + " 10 2nd raw moment (biased estimate) / ns^2\n"
    + " 11 3rd raw moment (biased estimate) / ns^3\n"
    + " 12 4th raw moment (biased estimate) / ns^4\n"
    + " 13 Sample minimum / ns\n"
    + " 14 Sample maximum / ns\n"
    + " 15 Number of observations/samples\n"
    + "\n"
    + "  Method 2: Uncensored counting.\n"
    + " 16-29 As Method 1\n"
    + "\n"
    + "  Method 3: Inverse transition rate\n"
    + " 30 Mean renewal time / ns\n"
    + "\n"
    + "Methods based on the ACF:\n"
    + "  Method 4: Numerical integration of the ACF\n"
    + " 31 Mean renewal time (1st raw moment) / ns\n"
    + " 32 Standard deviation / ns\n"
    + " 33 Coefficient of variation"
    + " 34 Skewness (Fisher)\n"
    + " 35 Excess kurtosis (Fisher)\n"
    + " 36 Median / ns\n"
    + " 37 Non-parametric skewness\n"
    + " 38 2nd raw moment / ns^2\n"
    + " 39 3rd raw moment / ns^3\n"
    + " 40 4th raw moment / ns^4\n"
    + "\n"
    + "  Method 5: Weibull fit of the ACF\n"
    + " 41-50 As Method 4\n"
    + " 51 Fit parameter tau0_wbl / ns\n"
    + " 52 Standard deviation of tau0_wbl / ns\n"
    + " 53 Fit parameter beta_wbl\n"
    + " 54 Standard deviation of beta_wbl\n"
    + " 55 Coefficient of determination of the fit (R^2 value)\n"
    + " 56 Root-mean-square error (RMSE) of the fit\n"
    + "\n"
    + "  Method 6: Burr Type XII fit of the ACF\n"
    + " 57-70 As Method 5\n"
    + " 71 Fit parameter delta_brr\n"
    + " 72 Standard deviation of delta_brr\n"
    + " 73 Coefficient of determination of the fit (R^2 value)\n"
    + " 74 Root-mean-square error (RMSE) of the fit\n"
    + "\n"
    + "  Fit region for all ACF fitting methods\n"
    + " 75 Start of fit region (inclusive) / ns\n"
    + " 76 End of fit region (inclusive) / ns\n"
    + "\n"
    + "Methods based on the Kaplan-Meier estimator:\n"
    + "  Method 7: Numerical integration of the Kaplan-Meier estimator\n"
    + " 77-86 As Method 4\n"
    + "\n"
    + "  Method 8: Weibull fit of the Kaplan-Meier estimator\n"
    + " 87-102 As Method 5\n"
    + "\n"
    + "  Method 9: Burr Type XII fit of the Kaplan-Meier estimator\n"
    + " 103-120 As Method 6\n"
    + "\n"
    + "  Fit region for all Kaplan-Meier estimator fitting methods\n"
    + " 121 Start of fit region (inclusive) / ns\n"
    + " 122 End of fit region (inclusive) / ns\n"
    + "\n"
    + "Column number:\n"
)
header += "{:>14d}".format(1)
for i in range(2, data.shape[-1] + 1):
    header += " {:>16d}".format(i)
leap.io_handler.savetxt(outfile_txt, data, header=header)
print("Created {}".format(outfile_txt))


print("Creating plot(s)...")
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

label_sm = None
color_sm = None
marker_sm = "o"

ylabel_acf = "Autocorrelation Function"
ylabel_km = "Kaplan-Meier Estimate"

xlabel = r"Ether Oxygens per Chain $n_{EO}$"
xlim = (1, 200)
legend_title = (
    r"$"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
    + "-"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp2]
    + r"$"
    + "\n"
    + r"$r = %.2f$" % Sims.Li_O_ratios[0]
)
legend_title_sf = legend_title + "\n" + r"$n_{EO}$"
n_legend_col_sf = 1 + Sims.n_sims // (4 + 1)

cmap = plt.get_cmap()
c_vals = np.arange(Sims.n_sims)
c_norm = max(1, Sims.n_sims - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile_pdf)
with PdfPages(outfile_pdf) as pdf:
    # Plot distribution characteristics vs. chain length.
    ylabels = (
        "Renewal Time / ns",
        "Std. Dev. / ns",
        "Coeff. of Variation",
        "Skewness",
        "Excess Kurtosis",
        "Median / ns",
        "Non-Parametric Skewness",
    )
    for i, ylabel in enumerate(ylabels):
        # Figure containing all methods.
        fig, ax = plt.subplots(clear=True)
        # Figure containing only the finally chosen method.
        fig_single_method, ax_sm = plt.subplots(clear=True)
        if i == 0:
            offset_i_cnt = 0
        else:
            offset_i_cnt = 1
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
            ax_sm.axhline(
                y=y_exp,
                color=color_exp,
                linestyle=linestyle_exp,
                label=label_exp,
            )
        # Method 1: Censored counting.
        ax.errorbar(
            xdata,
            lts_cnt_cen_characs[:, i + offset_i_cnt],
            yerr=lts_cnt_cen_characs[:, i + 1] if i == 0 else None,
            label=label_cnt_cen,
            color=color_cnt_cen,
            marker=marker_cnt_cen,
            alpha=leap.plot.ALPHA,
        )
        # Method 2: Uncensored counting.
        ax.errorbar(
            xdata,
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
                xdata,
                lts_k,
                yerr=None,
                label=label_k,
                color=color_k,
                marker=marker_k,
                alpha=leap.plot.ALPHA,
            )
            valid = np.isfinite(lts_k)
            if np.any(valid):
                ax_sm.plot(
                    xdata[valid],
                    lts_k[valid],
                    label=label_sm,
                    color=color_sm,
                    marker=marker_sm,
                )
        # Method 4: Numerical integration of the remain probability.
        ax.errorbar(
            xdata,
            lts_rp_int_characs[:, i],
            yerr=None,
            label=label_rp_int,
            color=color_rp_int,
            marker=marker_rp_int,
            alpha=leap.plot.ALPHA,
        )
        # Method 5: Weibull fit of the remain probability.
        ax.errorbar(
            xdata,
            lts_rp_wbl_characs[:, i],
            yerr=None,
            label=label_rp_wbl,
            color=color_rp_wbl,
            marker=marker_rp_wbl,
            alpha=leap.plot.ALPHA,
        )
        # Method 6: Burr Type XII fit of the remain probability.
        ax.errorbar(
            xdata,
            lts_rp_brr_characs[:, i],
            yerr=None,
            label=label_rp_brr,
            color=color_rp_brr,
            marker=marker_rp_brr,
            alpha=leap.plot.ALPHA,
        )
        # Method 7: Numerical integration of the KM estimator.
        ax.errorbar(
            xdata,
            lts_km_int_characs[:, i],
            yerr=None,
            label=label_km_int,
            color=color_km_int,
            marker=marker_km_int,
            alpha=leap.plot.ALPHA,
        )
        # Method 8: Weibull fit of the Kaplan-Meier estimator.
        ax.errorbar(
            xdata,
            lts_km_wbl_characs[:, i],
            yerr=None,
            label=label_km_wbl,
            color=color_km_wbl,
            marker=marker_km_wbl,
            alpha=leap.plot.ALPHA,
        )
        # Method 9: Burr Type XII fit of the Kaplan-Meier estimator.
        ax.errorbar(
            xdata,
            lts_km_brr_characs[:, i],
            yerr=None,
            label=label_km_brr,
            color=color_km_brr,
            marker=marker_km_brr,
            alpha=leap.plot.ALPHA,
        )
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax.get_ylim()
        if ylim[0] < 0 and ylabel not in (
            "Skewness",
            "Excess Kurtosis",
            "Non-Parametric Skewness",
        ):
            ax.set_ylim(0, ylim[1])
        legend = ax.legend(
            title=legend_title, ncol=3, **mdtplt.LEGEND_KWARGS_XSMALL
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig)
        yd_min, yd_max = leap.plot.get_ydata_min_max(ax)
        if np.any(np.greater(yd_min, 0)):
            # Log scale y.
            ax.relim()
            ax.autoscale()
            ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax.set_xlim(xlim)
            pdf.savefig(fig)
        plt.close(fig)
        # The finally chosen method.
        ax_sm.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax_sm.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax_sm.get_ylim()
        if ylim[0] < 0 and ylabel not in (
            "Skewness",
            "Excess Kurtosis",
            "Non-Parametric Skewness",
        ):
            ax_sm.set_ylim(0, ylim[1])
        legend = ax_sm.legend(
            title=legend_title, ncol=3, **mdtplt.LEGEND_KWARGS_XSMALL
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig_single_method)
        yd_min, yd_max = leap.plot.get_ydata_min_max(ax_sm)
        if np.any(np.greater(yd_min, 0)):
            # Log scale y.
            ax_sm.relim()
            ax_sm.autoscale()
            ax_sm.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax_sm.set_xlim(xlim)
            pdf.savefig(fig_single_method)
        plt.close(fig_single_method)

    # Plot min, max and number of samples for count methods.
    ylabels = (
        "Min. Renewal Time / ns",
        "Max. Renewal Time / ns",
        "No. of Renewal Events",
        "Renewal Events per Li Ion",
    )
    for i, ylabel in enumerate(ylabels):
        if ylabel == "Renewal Events per Li Ion":
            divisor = Sims.res_nums["cation"]
            i = 2
        else:
            divisor = 1
        # Figure containing all methods.
        fig, ax = plt.subplots(clear=True)
        # Figure containing only the finally chosen method.
        fig_single_method, ax_sm = plt.subplots(clear=True)
        # Method 1: Censored counting.
        ax.plot(
            xdata,
            lts_cnt_cen_characs[:, 11 + i] / divisor,
            label=label_cnt_cen,
            color=color_cnt_cen,
            marker=marker_cnt_cen,
            alpha=leap.plot.ALPHA,
        )
        valid = np.isfinite(lts_cnt_cen_characs[:, 11 + i])
        if np.any(valid):
            if ylabel == "Renewal Events per Li Ion":
                div = divisor[valid]
            else:
                div = divisor
            ax_sm.plot(
                xdata[valid],
                lts_cnt_cen_characs[:, 11 + i][valid] / div,
                label=label_sm,
                color=color_sm,
                marker=marker_sm,
            )
        # Method 2: Uncensored counting.
        ax.plot(
            xdata,
            lts_cnt_unc_characs[:, 11 + i] / divisor,
            label=label_cnt_unc,
            color=color_cnt_unc,
            marker=marker_cnt_unc,
            alpha=leap.plot.ALPHA,
        )
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax.get_ylim()
        if ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        legend = ax.legend(
            title=legend_title, ncol=2, **mdtplt.LEGEND_KWARGS_XSMALL
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig)
        yd_min, yd_max = leap.plot.get_ydata_min_max(ax)
        if np.any(np.greater(yd_min, 0)):
            # Log scale y.
            ax.relim()
            ax.autoscale()
            ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax.set_xlim(xlim)
            pdf.savefig(fig)
        plt.close(fig)
        # The finally chosen method.
        ax_sm.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax_sm.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax_sm.get_ylim()
        if ylim[0] < 0:
            ax_sm.set_ylim(0, ylim[1])
        legend = ax_sm.legend(
            title=legend_title, ncol=2, **mdtplt.LEGEND_KWARGS_XSMALL
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig_single_method)
        yd_min, yd_max = leap.plot.get_ydata_min_max(ax_sm)
        if np.any(np.greater(yd_min, 0)):
            # Log scale y.
            ax_sm.relim()
            ax_sm.autoscale()
            ax_sm.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax_sm.set_xlim(xlim)
            pdf.savefig(fig_single_method)
        plt.close(fig_single_method)

    # Plot fit parameters tau0, beta and delta.
    ylabels = (
        r"Fit Parameter $\tau_0$ / ns",
        r"Fit Parameter $\beta$",
        r"Fit Parameter $\delta$",
    )
    for i, ylabel in enumerate(ylabels):
        fig, ax = plt.subplots(clear=True)
        if i < 2:
            # Method 5: Weibull fit of the remain probability.
            ax.errorbar(
                xdata,
                popt_rp_wbl[:, i],
                yerr=perr_rp_wbl[:, i],
                label=label_rp_wbl,
                color=color_rp_wbl,
                marker=marker_rp_wbl,
                alpha=leap.plot.ALPHA,
            )
        # Method 6: Burr Type XII fit of the remain probability.
        ax.errorbar(
            xdata,
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
                xdata,
                popt_km_wbl[:, i],
                yerr=perr_km_wbl[:, i],
                label=label_km_wbl,
                color=color_km_wbl,
                marker=marker_km_wbl,
                alpha=leap.plot.ALPHA,
            )
        # Method 9: Burr Type XII fit of the Kaplan-Meier estimator.
        ax.errorbar(
            xdata,
            popt_conv_km_brr[:, i],
            yerr=perr_conv_km_brr[:, i],
            label=label_km_brr,
            color=color_km_brr,
            marker=marker_km_brr,
            alpha=leap.plot.ALPHA,
        )
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax.get_ylim()
        if ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        legend = ax.legend(
            title=legend_title, ncol=2, **mdtplt.LEGEND_KWARGS_XSMALL
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
        fig, ax = plt.subplots(clear=True)
        # Method 5: Weibull fit of the remain probability.
        ax.plot(
            xdata,
            lts_rp_wbl_fit_goodness[:, i],
            label=label_rp_wbl,
            color=color_rp_wbl,
            marker=marker_rp_wbl,
            alpha=leap.plot.ALPHA,
        )
        # Method 6: Burr Type XII fit of the remain probability.
        ax.plot(
            xdata,
            lts_rp_brr_fit_goodness[:, i],
            label=label_rp_brr,
            color=color_rp_brr,
            marker=marker_rp_brr,
            alpha=leap.plot.ALPHA,
        )
        # Method 8: Weibull fit of the Kaplan-Meier estimator.
        ax.plot(
            xdata,
            lts_km_wbl_fit_goodness[:, i],
            label=label_km_wbl,
            color=color_km_wbl,
            marker=marker_km_wbl,
            alpha=leap.plot.ALPHA,
        )
        # Method 9: Burr Type XII fit of the Kaplan-Meier estimator.
        ax.plot(
            xdata,
            lts_km_brr_fit_goodness[:, i],
            label=label_km_brr,
            color=color_km_brr,
            marker=marker_km_brr,
            alpha=leap.plot.ALPHA,
        )
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax.get_ylim()
        if ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        legend = ax.legend(
            title=legend_title, ncol=2, **mdtplt.LEGEND_KWARGS_XSMALL
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
    fig, ax = plt.subplots(clear=True)
    # Fit of remain probability.
    ax.plot(
        xdata,
        (fit_stop_rp - 1) * time_conv,
        label="ACF",
        color=color_rp_wbl,
        marker=marker_rp_wbl,
    )
    # Fit of Kaplan-Meier estimator.
    ax.plot(
        xdata,
        (fit_stop_km - 1) * time_conv,
        label="KM",
        color=color_km_wbl,
        marker=marker_km_wbl,
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    legend = ax.legend(
        title=legend_title, ncol=2, **mdtplt.LEGEND_KWARGS_XSMALL
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

    # Plot remain probabilities and Weibull fits.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        rp = remain_probs[sim_ix]
        if rp is None or not np.any(np.isfinite(rp)):
            continue
        times_fit = times[fit_start_rp[sim_ix] : fit_stop_rp[sim_ix]]
        fit = mdt.func.kww(times_fit, *popt_rp_wbl[sim_ix])
        lines = ax.plot(
            times,
            rp,
            label=r"$%d$" % Sim.O_per_chain,
            linewidth=1,
            alpha=leap.plot.ALPHA,
        )
        ax.plot(
            times_fit,
            fit,
            label="Wbl Fit" if sim_ix == Sims.n_sims - 1 else None,
            linestyle="dashed",
            color=lines[0].get_color(),
            alpha=leap.plot.ALPHA,
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel="Time / ns",
        ylabel=ylabel_acf,
        xlim=(times[1], times[-1]),
        ylim=(0, 1),
    )
    legend = ax.legend(
        title=legend_title_sf,
        ncol=n_legend_col_sf,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()
    # Plot Weibull fit residuals (remain probability).
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        rp = remain_probs[sim_ix]
        if rp is None or not np.any(np.isfinite(rp)):
            continue
        times_fit = times[fit_start_rp[sim_ix] : fit_stop_rp[sim_ix]]
        fit = mdt.func.kww(times_fit, *popt_rp_wbl[sim_ix])
        res = rp[fit_start_rp[sim_ix] : fit_stop_rp[sim_ix]] - fit
        ax.plot(
            times_fit,
            res,
            label=r"$%d$" % Sim.O_per_chain,
            alpha=leap.plot.ALPHA,
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel="Time / ns",
        ylabel="ACF Weibull Fit Res.",
        xlim=(times[1], times[-1]),
    )
    legend = ax.legend(
        title=legend_title_sf,
        ncol=n_legend_col_sf,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot Kaplan-Meier estimates and Weibull fits.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        km = km_surv_funcs[sim_ix]
        if km is None or not np.any(np.isfinite(km)):
            continue
        times_fit = times[fit_start_km[sim_ix] : fit_stop_km[sim_ix]]
        fit = mdt.func.kww(times_fit, *popt_km_wbl[sim_ix])
        lines = ax.plot(
            times,
            km,
            label=r"$%d$" % Sim.O_per_chain,
            linewidth=1,
            alpha=leap.plot.ALPHA,
        )
        ax.plot(
            times_fit,
            fit,
            label="Wbl Fit" if sim_ix == Sims.n_sims - 1 else None,
            linestyle="dashed",
            color=lines[0].get_color(),
            alpha=leap.plot.ALPHA,
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel="Time / ns",
        ylabel=ylabel_km,
        xlim=(times[1], times[-1]),
        ylim=(0, 1),
    )
    legend = ax.legend(
        title=legend_title_sf,
        ncol=n_legend_col_sf,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()
    # Plot Weibull fit residuals (Kaplan-Meier).
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        km = km_surv_funcs[sim_ix]
        if km is None or not np.any(np.isfinite(km)):
            continue
        times_fit = times[fit_start_km[sim_ix] : fit_stop_km[sim_ix]]
        fit = mdt.func.kww(times_fit, *popt_km_wbl[sim_ix])
        res = km[fit_start_km[sim_ix] : fit_stop_km[sim_ix]] - fit
        ax.plot(
            times_fit,
            res,
            label=r"$%d$" % Sim.O_per_chain,
            alpha=leap.plot.ALPHA,
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel="Time / ns",
        ylabel="KM Weibull Fit Res.",
        xlim=(times[1], times[-1]),
    )
    legend = ax.legend(
        title=legend_title_sf,
        ncol=n_legend_col_sf,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot remain probabilities and Burr fits.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        rp = remain_probs[sim_ix]
        if rp is None or not np.any(np.isfinite(rp)):
            continue
        times_fit = times[fit_start_rp[sim_ix] : fit_stop_rp[sim_ix]]
        fit = mdt.func.burr12_sf_alt(times_fit, *popt_rp_brr[sim_ix])
        lines = ax.plot(
            times,
            rp,
            label=r"$%d$" % Sim.O_per_chain,
            linewidth=1,
            alpha=leap.plot.ALPHA,
        )
        ax.plot(
            times_fit,
            fit,
            label="Burr Fit" if sim_ix == Sims.n_sims - 1 else None,
            linestyle="dashed",
            color=lines[0].get_color(),
            alpha=leap.plot.ALPHA,
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel="Time / ns",
        ylabel=ylabel_acf,
        xlim=(times[1], times[-1]),
        ylim=(0, 1),
    )
    legend = ax.legend(
        title=legend_title_sf,
        ncol=n_legend_col_sf,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()
    # Plot Burr fit residuals (remain probability).
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        rp = remain_probs[sim_ix]
        if rp is None or not np.any(np.isfinite(rp)):
            continue
        times_fit = times[fit_start_rp[sim_ix] : fit_stop_rp[sim_ix]]
        fit = mdt.func.burr12_sf_alt(times_fit, *popt_rp_brr[sim_ix])
        res = rp[fit_start_rp[sim_ix] : fit_stop_rp[sim_ix]] - fit
        ax.plot(
            times_fit,
            res,
            label=r"$%d$" % Sim.O_per_chain,
            alpha=leap.plot.ALPHA,
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel="Time / ns",
        ylabel="ACF Burr Fit Res.",
        xlim=(times[1], times[-1]),
    )
    legend = ax.legend(
        title=legend_title_sf,
        ncol=n_legend_col_sf,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot Kaplan-Meier estimates and Burr fits for each state.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        km = km_surv_funcs[sim_ix]
        if km is None or not np.any(np.isfinite(km)):
            continue
        times_fit = times[fit_start_km[sim_ix] : fit_stop_km[sim_ix]]
        fit = mdt.func.burr12_sf_alt(times_fit, *popt_km_brr[sim_ix])
        lines = ax.plot(
            times,
            km,
            label=r"$%d$" % Sim.O_per_chain,
            linewidth=1,
            alpha=leap.plot.ALPHA,
        )
        ax.plot(
            times_fit,
            fit,
            label="Burr Fit" if sim_ix == Sims.n_sims - 1 else None,
            linestyle="dashed",
            color=lines[0].get_color(),
            alpha=leap.plot.ALPHA,
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel="Time / ns",
        ylabel=ylabel_km,
        xlim=(times[1], times[-1]),
        ylim=(0, 1),
    )
    legend = ax.legend(
        title=legend_title_sf,
        ncol=n_legend_col_sf,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()
    # Plot Burr fit residuals (Kaplan-Meier) for each state.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        km = km_surv_funcs[sim_ix]
        if km is None or not np.any(np.isfinite(km)):
            continue
        times_fit = times[fit_start_km[sim_ix] : fit_stop_km[sim_ix]]
        fit = mdt.func.burr12_sf_alt(times_fit, *popt_km_brr[sim_ix])
        res = km[fit_start_km[sim_ix] : fit_stop_km[sim_ix]] - fit
        ax.plot(
            times_fit,
            res,
            label=r"$%d$" % Sim.O_per_chain,
            alpha=leap.plot.ALPHA,
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel="Time / ns",
        ylabel="KM Burr Fit Res.",
        xlim=(times[1], times[-1]),
    )
    legend = ax.legend(
        title=legend_title_sf,
        ncol=n_legend_col_sf,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile_pdf))
print("Done")
