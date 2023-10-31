#!/usr/bin/env python3

# python3 -m pip install --upgrade pynumdiff CVXPY PYCHEBFUN

"""
.. warning::

    This script is work in progress! The hyperparameters for finding the
    diffusive regime need further fine tuning.

Identify the diffusive regime of the mean squared displacement (MSD) and
extract the self-diffusion coefficient by fitting a straight line to it.
"""


# Standard libraries
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

# Third-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
import psutil
import pynumdiff
import pynumdiff.optimize as pynumdiffopt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# First-party libraries
import lintf2_ether_ana_postproc as leap


# pynumdiff produces logs by default.  The following line effectively
# disables all logging calls.
logging.disable(logging.CRITICAL)


def einstein_msd(times, diff_coeff, n_dim=3):
    r"""
    Calculate the Einstein MSD

    .. math::

        \langle r^2 \rangle = 2d \cdot D t

    Parameters
    ----------
    times : array_like
        Times :math:`t` at which to evaluate the Einstein MSD.
    diff_coeff : float
        The diffusion coefficient :math:`D`.
    n_dim : int, optional
        Number of dimensions :math:`d` in which the diffusive process
        takes place.

    Returns
    -------
    msd : numpy.ndarray
        Mean squared displacement.
    """
    times = np.asarray(times)
    return 2 * n_dim * diff_coeff * times


def estimate_tvgamma(cutoff_frequency, dt):
    """
    Estimate the hyperparameter `tvgamma` using the heuristic described
    in the paper by van Bruegel et al.
    [#]_

    The hyperparameter is required for optimizing the smoothing
    parameters using :func:`pynumdiff.optimize.linear_model.savgoldiff`.

    Parameters
    ----------
    cutoff_frequency : float
        Cutoff frequency in inverse time units.  The cutoff frequency
        should be the highest frequency content of the signal in the
        data.  All frequencies above the cutoff frequency are regarded
        as noise.  As general rule, choose the cutoff frequency as the
        frequency where the power spectrum of the data starts to
        decrease and the noise of the spectrum starts to increase.

        Note that the time between restarting points used to calculate
        the MSD is 1 ns for my simulations.  Therefore, the cutoff
        frequency should probably be somewhat lower than 1 ns^{-1}.
    dt : float
        Time step between recorded data points.

    Returns
    -------
    tvgamma :float
        Hyperparameter.  The larger the value of `tvgamma`, the smoother
        the derivative but consequently the greater the deviation
        between the true data and the smoothed data.

    See Also
    --------
    :func:`power_spectrum` :
        Calculate the power spectrum of the data

    References
    ----------
    .. [#] F. van Bruegel, J. N. Kutz, B. W. Brunton,
        `Numerical Differentiation of Noisy Data: A Unifying
        Multi-Objective Optimization Framework
        <https://doi.org/10.1109/ACCESS.2020.3034077>`_,
        IEEE Access, 2020, 8, 196865-196877.
    """
    log_gamma = -1.6 * np.log(cutoff_frequency) - 0.71 * np.log(dt) - 5.1
    tvgamma = np.exp(log_gamma)
    return tvgamma


timer_tot = datetime.now()  # Initiate monitoring of wall-clock time.
proc = psutil.Process()
proc.cpu_percent()  # Initiate monitoring of CPU usage.

# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Identify the diffusive regime of the mean squared displacement (MSD)"
        " and extract the self-diffusion coefficient by fitting a straight"
        " line to it."
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
    choices=("Li", "NTf2", "ether", "NBT", "OBT", "OE"),
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--msd-component",
    type=str,
    required=False,
    default="tot",
    choices=("tot", "x", "y", "z"),
    help="The MSD component to use for the analysis.  Default: %(default)s",
)
args = parser.parse_args()

analysis = "msd"  # Analysis name.
tool = "mdt"  # Analysis software.
outfile_base = (
    args.settings + "_" + args.system + "_" + analysis + "_" + args.cmp
)
outfile_txt = outfile_base + ".txt.gz"
outfile_pdf = outfile_base + ".pdf"

# Columns to read from the MSD file.
cols = (0,)  # Diffusion times [ps].
if args.msd_component == "tot":
    cols += (1,)
    # Number of dimensions in which the diffusive process takes place.
    n_dim = 3
elif args.msd_component == "x":
    cols += (2,)
    n_dim = 1
elif args.msd_component == "y":
    cols += (3,)
    n_dim = 1
elif args.msd_component == "z":
    cols += (4,)
    n_dim = 1
else:
    raise ValueError("Unknown --msd-component ({})".format(args.msd_component))

# Number of frames between restarting points used for calculating the
# MSD.
restart_interval = 500  # [frames].
# Restarting frequency used for calculating the MSD.
restart_frequency = 1 / (restart_interval * 2e-3)  # [ns^-1].
# Cutoff frequency for estimating `tvgamma`.
cutoff_frequency = restart_frequency / 2  # [ns^-1].
# Calculate the standard deviation of the derivative of MSD/t between
# `start_deriv_sd_pct` and `stop_deriv_sd_pct` percent of the data.
start_deriv_sd_pct = 0.8
stop_deriv_sd_pct = 0.9
# Regard the derivative of MSD/t as zero if it lies within +/- the
# calculated standard deviation times `deriv_sd_factor`.
deriv_sd_factor = 4


print("Creating Simulation instance(s)...")
timer = datetime.now()
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    top_path = "q%g" % surfq
else:
    surfq = None
    top_path = "bulk"
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, top_path)
print("Elapsed time:         {}".format(datetime.now() - timer))
print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))


print("\n")
print("Reading data...")
timer = datetime.now()
file_suffix = analysis + "_" + args.cmp + ".txt.gz"
infile = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
times, msd = np.loadtxt(infile, usecols=[0, 1], unpack=True)
times *= 1e-3  # ps -> ns
msd *= 1e-2  # A^2 -> nm^2

time_diffs = np.diff(times)
time_diff = time_diffs[0]
if not np.all(time_diffs >= 0):
    raise ValueError("The diffusion times are not sorted")
if not np.allclose(time_diffs, time_diff, rtol=0):
    raise ValueError("The diffusion times are not evenly spaced")
del time_diffs

# Divide the MSD by the diffusion time.
# => The MSD becomes constant in the diffusive regime.
# Omit `times[0]`, because it is zero.
msd_t = msd[1:] / times[1:]
times_t = times[1:]
print("Elapsed time:         {}".format(datetime.now() - timer))
print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))


print("\n")
print("Calculating optimized smoothing parameters...")
timer = datetime.now()
tvgamma = estimate_tvgamma(cutoff_frequency=cutoff_frequency, dt=time_diff)
tvgamma *= 100
# Initial guesses of the smoothing parameters.
param_guesses = [
    [
        1,  # Order of the polynomial.
        restart_interval // n,  # Size of the sliding window.
        restart_interval // n,  # Window size for Gaussian smoothing.
    ]
    for n in (0.1, 0.2, 1, 5, 10)
]
params, val = pynumdiffopt.linear_model.savgoldiff(
    msd_t, dt=time_diff, params=param_guesses, tvgamma=tvgamma
)
print("tvgamma = {}".format(tvgamma))
print("params  = {}".format(params))
print("val     = {}".format(val))  # Optimal value of objective function
print("Elapsed time:         {}".format(datetime.now() - timer))
print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))


print("\n")
print("Smoothing data and calculating derivative...")
timer = datetime.now()
msd_t_smooth, msd_t_deriv = pynumdiff.linear_model.savgoldiff(
    msd_t, dt=time_diff, params=params
)
print("Elapsed time:         {}".format(datetime.now() - timer))
print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))


print("\n")
print("Identifying diffusive regime and calculating diffusion coefficient...")
timer = datetime.now()
start_deriv_sd_tot = int(start_deriv_sd_pct * len(msd_t_deriv))
stop_deriv_sd_tot = int(stop_deriv_sd_pct * len(msd_t_deriv))
msd_t_deriv_sd = np.std(
    msd_t_deriv[start_deriv_sd_tot:stop_deriv_sd_tot], ddof=1
)
deriv_tol = deriv_sd_factor * msd_t_deriv_sd
diffusive = (msd_t_deriv > -deriv_tol) & (msd_t_deriv < deriv_tol)
if not np.any(diffusive):
    fit_start, fit_stop = -1, -1
    diff_coeff, diff_coeff_sd = np.nan, np.nan
    times_fit, msd_fit = np.array([]), np.array([])
else:
    first_true = np.argmax(diffusive)
    fit_start, length, _value = mdt.nph.find_const_seq_long(
        diffusive[first_true:], tol=0.1
    )
    fit_start += first_true
    fit_stop = fit_start + length
    diff_coeff = np.mean(msd_t[fit_start:fit_stop]) / (2 * n_dim)
    diff_coeff_sd = np.std(msd_t[fit_start:fit_stop], ddof=1)
    diff_coeff_sd /= 2 * n_dim
    times_fit = times_t[fit_start:fit_stop]
    msd_fit = einstein_msd(times_fit, diff_coeff, n_dim)
print("Elapsed time:         {}".format(datetime.now() - timer))
print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))


print("\n")
print("Creating output...")
timer = datetime.now()
header = ""
print("Elapsed time:         {}".format(datetime.now() - timer))
print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))


print("\n")
print("Creating plots...")
timer = datetime.now()
label_orig = "Original"
label_smooth = "Smoothed"
label_fit = "Fit"
color_orig = "tab:blue"
color_smooth = "tab:orange"
color_fit = "black"
color_thresh = "tab:red"
ls_orig = "solid"
ls_smooth = "dotted"
ls_fit = "dashed"
lw_orig = None
lw_smooth = None
lw_fit = None

mdt.fh.backup(outfile_pdf)
with PdfPages(outfile_pdf) as pdf:
    # Plot MSD vs time.
    fig, ax = plt.subplots(clear=True)
    ax.plot(
        times,
        msd,
        label=label_orig,
        color=color_orig,
        linestyle=ls_orig,
        linewidth=lw_orig,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t,
        msd_t_smooth * times_t,
        label=label_smooth,
        color=color_smooth,
        linestyle=ls_smooth,
        linewidth=lw_smooth,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_fit,
        msd_fit,
        label=label_fit,
        color=color_fit,
        linestyle=ls_fit,
        linewidth=lw_fit,
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel="Diffusion Time / ns",
        ylabel=r"MSD / nm$^2$",
        xlim=(times[0], times[-1]),
        ylim=(0, None),
    )
    ax.legend(loc="upper left")
    pdf.savefig()
    # Log scale x.
    ax.set_xlim(times[1], times[-1])
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale xy.
    ax.relim()
    ax.autoscale()
    ax.set_xlim(times[1], times[-1])
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot MSD/time vs time.
    fig, ax = plt.subplots(clear=True)
    ax.plot(
        times_t,
        msd_t,
        label=label_orig,
        color=color_orig,
        linestyle=ls_orig,
        linewidth=lw_orig,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t,
        msd_t_smooth,
        label=label_smooth,
        color=color_smooth,
        linestyle=ls_smooth,
        linewidth=lw_smooth,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_fit,
        msd_fit / times_fit,
        label=label_fit,
        color=color_fit,
        linestyle=ls_fit,
        linewidth=lw_fit,
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel=r"Diffusion Time $t$ / ns",
        ylabel=r"MSD$(t)/t$ / nm$^2$/ns",
        xlim=(times_t[0], times_t[-1]),
    )
    ax.legend(loc="upper right")
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale xy.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot derivative of MSD/time vs time.
    fig, ax = plt.subplots(clear=True)
    ax.plot(
        times_t[:fit_start],
        msd_t_deriv[:fit_start],
        color=color_orig,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t[fit_stop:],
        msd_t_deriv[fit_stop:],
        color=color_orig,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_fit,
        msd_t_deriv[fit_start:fit_stop],
        color=color_fit,
        label="Fit Region",
        alpha=leap.plot.ALPHA,
    )
    ax.axvline(
        times_t[start_deriv_sd_tot],
        color="tab:green",
        label="Std Region",
        alpha=leap.plot.ALPHA,
    )
    ax.axvline(
        times_t[stop_deriv_sd_tot], color="tab:green", alpha=leap.plot.ALPHA
    )
    ax.set(
        xlabel=r"Diffusion Time $t$ / ns",
        ylabel=r"d$/$d$t$ MSD$(t)/t$ / nm$^2$/ns$^2$",
        xlim=(times_t[0], times_t[-1]),
    )
    ax.legend(loc="lower right")
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Linear scale, zoom to 0 +/- 2*tolerance.
    ax.axhline(
        -deriv_tol,
        label=r"Tolerance ($%.1f \sigma$)" % deriv_sd_factor,
        color=color_thresh,
        alpha=leap.plot.ALPHA,
    )
    ax.axhline(deriv_tol, color=color_thresh, alpha=leap.plot.ALPHA)
    ax.set_xscale("linear")
    ax.set_ylim(-3 * deriv_tol, 3 * deriv_tol)
    ax.legend(loc="upper left")
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot power spectrum of MSD/time.
    frequencies, amplitudes = leap.misc.power_spectrum(msd_t, dt=time_diff)
    fig, ax = plt.subplots(clear=True)
    ax.plot(frequencies, amplitudes, alpha=leap.plot.ALPHA)
    ax.axvline(
        cutoff_frequency,
        label="Cutoff",
        color=color_thresh,
        alpha=leap.plot.ALPHA,
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel=r"Frequency / ns$^{-1}$",
        ylabel=r"Pow. Spec. of MSD$(t)/t$",
        xlim=(frequencies[1], frequencies[-1]),
        ylim=(np.min(amplitudes[-2]), np.max(amplitudes)),
    )
    ax.legend(loc="upper right")
    pdf.savefig()
    plt.close()
print("Created {}".format(outfile_pdf))
print("Elapsed time:         {}".format(datetime.now() - timer))
print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))

# Remove the log file created by default by pynumdiff.
pynumdiff_debug_log = "debug.log"
if os.path.isfile(pynumdiff_debug_log):
    os.remove(pynumdiff_debug_log)

print("\n")
print("{} done".format(os.path.basename(sys.argv[0])))
print("Totally elapsed time: {}".format(datetime.now() - timer_tot))
_cpu_time = timedelta(seconds=sum(proc.cpu_times()[:4]))
print("CPU time:             {}".format(_cpu_time))
print("CPU usage:            {:.2f} %".format(proc.cpu_percent()))
print("Current memory usage: {:.2f} MiB".format(mdt.rti.mem_usage(proc)))
