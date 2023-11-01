#!/usr/bin/env python3

"""
Identify the diffusive regime of the mean squared displacement (MSD) and
extract the self-diffusion coefficient by fitting a straight line to it.
"""


# Standard libraries
import argparse
import warnings

# Third-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# First-party libraries
import lintf2_ether_ana_postproc as leap


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
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis
    + "_"
    + args.cmp
    + "_"
    + args.msd_component
)
outfile_txt = outfile_base + "_diff_coeff" + ".txt.gz"
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
restart_interval = 500
# Window size for calculating the moving average of MSD/t.
movav_wsize = 2 * restart_interval  # [data points].
if movav_wsize % 2 == 0:
    # Ensure that `movav_wsize` is odd.
    movav_wsize += 1
# Regard the derivative of the moving average of MSD/t as zero if it
# lies within +/- the minimum uncertainty of the derivative times
# `deriv_sd_factor`.
# 0.674490 sigma = 50% confidence interval of the normal distribution.
# 1.0 sigma = 68.2689492% confidence interval of the normal dist.
# 1.17741 sigma = Full width at half maximum of the normal distribution
# 1.644854 sigma = 90% confidence interval of the normal distribution.
deriv_sd_factor = 1.0
# Stop fitting the MSD at `fit_stop_pct` percent of the data.
fit_stop_pct = 0.9


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


print("Identifying diffusive regime and calculating diffusion coefficient...")
# Time shift of the moving average to the original times.
# To get the central moving average, set the time shift to half of the
# window size.
movav_t_shift = (movav_wsize - 1) // 2
times_t_movav = times_t[movav_t_shift : len(times_t) - movav_t_shift]

# Calculate the moving average of MSD/t.
msd_t_movav = mdt.stats.movav(msd_t, wlen=movav_wsize)

# Calculate the derivative of the moving average.
# Use `np.diff` instead of `numpy.gradient`, because otherwise the
# propagation of uncertainty is unclear.
times_t_movav_grad = times_t_movav[1:]
msd_t_movav_grad = np.diff(msd_t_movav)
msd_t_movav_grad /= time_diff

# Calculate the variance (squared uncertainty) of the moving average.
msd_t_movav_var = mdt.stats.movav(msd_t**2, wlen=movav_wsize)
msd_t_movav_var -= msd_t_movav**2
msd_t_movav_var /= movav_wsize - 1
if np.any(msd_t_movav_var < 0):
    raise ValueError(
        "At least one element of `msd_t_movav_var` is less than zero.  This"
        " should not have happened"
    )


# Calculate the uncertainty of the derivative of the moving average.
# Propagation of uncertainty:
# https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
# f = c * (X - Y)
# Var[f] = c^2 * (Var[X] + Var[Y] - 2 Cov[X, Y])
msd_t_movav_cov = mdt.stats.movav(msd_t[:-1] * msd_t[1:], wlen=movav_wsize)
msd_t_movav_cov -= msd_t_movav[:-1] * msd_t_movav[1:]
msd_t_movav_cov /= movav_wsize - 1
msd_t_movav_grad_sd = msd_t_movav_var[:-1] + msd_t_movav_var[1:]
msd_t_movav_grad_sd -= 2 * msd_t_movav_cov
if np.any(msd_t_movav_grad_sd < 0):
    warnings.warn(
        "At least one element of `msd_t_movav_grad_sd` is less than zero."
        "  This might happen, because the formula for the propagation of"
        " uncertainty is just an approximation.",
        RuntimeWarning,
        stacklevel=2,
    )
msd_t_movav_grad_sd = np.sqrt(msd_t_movav_grad_sd, out=msd_t_movav_grad_sd)
msd_t_movav_grad_sd /= time_diff
msd_t_movav_grad_sd_min = np.min(
    msd_t_movav_grad_sd[np.isfinite(msd_t_movav_grad_sd)]
)

# Calculate the uncertainty of the moving average.
msd_t_movav_sd = np.sqrt(msd_t_movav_var, out=msd_t_movav_var)
del msd_t_movav_var

# Identify the diffusive regime.
diffusive = msd_t_movav_grad >= -deriv_sd_factor * msd_t_movav_grad_sd
diffusive &= msd_t_movav_grad <= deriv_sd_factor * msd_t_movav_grad_sd

if not np.any(diffusive):
    fit_start, fit_stop = -1, -1
else:
    # Find the longest sequence of ``True`` in `diffusive`.
    # Get the start, length and value of all consecutive sequences of
    # ``False`` and ``True`` in `diffusive`.
    seq_starts, seq_lengths, vals = mdt.nph.get_const_seqs(diffusive, tol=0.5)
    # Discard all sequences of ``False``
    valid = np.flatnonzero(vals)
    seq_starts, seq_lengths = seq_starts[valid], seq_lengths[valid]
    # Find the longest sequence of ``True``.
    ix_longest = np.argmax(seq_lengths)
    fit_start = seq_starts[ix_longest]
    fit_stop = fit_start + seq_lengths[ix_longest]
    # Acount for the shift of indices: The times of the moving average,
    # `times_t_movav` (and consequently `diffusive`), are shifted to the
    # original times `time_t` by `movav_t_shift`.
    fit_start += movav_t_shift
    fit_stop += movav_t_shift
    # Reduce the fit region by half of the window size as security
    # buffer.
    fit_start += (movav_wsize - 1) // 2
    fit_stop -= (movav_wsize - 1) // 2
    del seq_starts, seq_lengths, vals, valid

fit_stop_tot = int(fit_stop_pct * len(msd_t))
fit_stop_tot -= (movav_wsize - 1) // 2  # Security buffer.
if fit_start >= fit_stop_tot:
    fit_start, fit_stop = -1, -1
if fit_stop > fit_stop_tot:
    fit_stop = fit_stop_tot

if fit_start > 0 and fit_stop > 0:
    diff_coeff = np.mean(msd_t[fit_start:fit_stop]) / (2 * n_dim)
    diff_coeff_sd = np.std(msd_t[fit_start:fit_stop], ddof=1)
    diff_coeff_sd /= 2 * n_dim
    times_fit = times_t[fit_start:fit_stop]
    msd_fit = einstein_msd(times_fit, diff_coeff, n_dim)
    msd_fit_sd = einstein_msd(times_fit, diff_coeff_sd, n_dim)
    # Fit residuals.
    msd_fit_res = msd[fit_start + 1 : fit_stop + 1] - msd_fit
    r2, rmse = leap.misc.fit_goodness(
        data=msd[fit_start + 1 : fit_stop + 1], fit=msd_fit
    )
else:
    diff_coeff, diff_coeff_sd = np.nan, np.nan
    times_fit, msd_fit = np.array([]), np.array([])
    msd_fit_sd, msd_fit_res = np.array([]), np.array([])
    r2, rmse = np.nan, np.nan


print("Creating output...")
if fit_start > 0 and fit_stop > 0:
    data = np.array(
        [
            diff_coeff,
            diff_coeff_sd,
            times_t[fit_start],
            times_t[fit_stop],
            r2,
            rmse,
        ]
    )
else:
    data = np.array([diff_coeff, diff_coeff_sd, np.nan, np.nan, r2, rmse])
data = data.reshape(1, data.size)
header = (
    "Diffusion coefficient\n"
    + "\n"
    + "System:       {:s}\n".format(args.system)
    + "Settings:     {:s}\n".format(args.settings)
    + "Input file:   {:s}\n".format(infile)
    + "Read Columns: {}\n".format(cols)
    + "\n"
    + "\n"
    + "The diffusion coefficient D is calculated according to the Einstein\n"
    + "relation\n"
    + "  MSD(t) = 2*d * D * t\n"
    + "by calculating the mean of MSD(t)/t within the diffusive regime and\n"
    + "dividing it by 2*d, where d is the number of dimensions in which the\n"
    + "diffusive process takes place.\n"
    + "\n"
    + "The diffusive regime is identified in the following way:\n"
    + "\n"
    + "1.) Calculate the function D(t) = MSD(t)/t.\n"
    + "\n"
    + "2.) Smooth D(t) by calculating the central moving average with a\n"
    + "window size of {:d} data points ({:f} ns).\n".format(
        movav_wsize, movav_wsize * time_diff
    )
    + "\n"
    + "3.) Estimate the derivative of the moving average using finite\n"
    + "differences.\n"
    + "\n"
    + "4.) Identify the diffusive regime as the largest continuous sequence\n"
    + "where the derivative is zero within {:f} times of its\n".format(
        deriv_sd_factor
    )
    + "uncertainty (which is calculated from the uncertainty of the moving\n"
    + "average using standard propagation of uncertainty).\n"
    + "\n"
    + "6.) Reduce the diffusive regime on both sides by half of the\n"
    + "moving-average window size as security buffer.  Furthermore, if the\n"
    + "diffusive regime exceeds {:f} percent of the available D(t)\n".format(
        fit_stop_pct
    )
    + "data minus half of the moving-average window size, cut it at this\n"
    + "point, because D(t) tends to get quite noisy beyond it.\n"
    + "\n"
    + "5.) Calculate the diffusion coefficient as the mean of D(t) within\n"
    + "the diffusive regime.\n"
    + "\n"
    + "movav_wsize     = {:d} data points ({:f} ns)\n".format(
        movav_wsize, movav_wsize * time_diff
    )
    + "deriv_sd_factor = {:f}\n".format(deriv_sd_factor)
    + "fit_stop_pct    = {:f}\n".format(fit_stop_pct)
    + "\n"
    + "\n"
    + "Compound:       {:s}\n".format(args.cmp)
    + "MSD component:  {:s}\n".format(args.msd_component)
    + "No. dimensions: {:d}\n".format(n_dim)
    + "\n"
    + "The columns contain:\n"
    + "  1 Diffusion coefficient in nm^2/ns\n"
    + "  2 Standard deviation of D determined from the fit in nm^2/ns\n"
    + "  3 Start of the fitting/averaging region in ns\n"
    + "  4 End of the fitting/averaging region in ns\n"
    + "  5 Coefficient of determination of the fit\n"
    + "  6 Root-mean-square error (RMSE) of the fit / nm^2\n"
    + "{:>14d}".format(1)
)
for col_num in range(2, data.shape[-1] + 1):
    header += " {:>16d}".format(col_num)
leap.io_handler.savetxt(outfile_txt, data, header=header)
print("Created {}".format(outfile_txt))


print("Creating plots...")
label_orig = "Original"
label_movav = "Mov. Avg."
label_fit = "Fit (Orig.)"
color_orig = "tab:blue"
color_movav = "tab:orange"
color_fit = "black"
color_thresh = "tab:red"
color_fit_stop_tot = "tab:grey"
ls_orig = "solid"
ls_movav = "dotted"
ls_fit = "dashed"

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
        alpha=leap.plot.ALPHA,
    )
    ax.fill_between(
        times_t_movav,
        y1=(msd_t_movav + msd_t_movav_sd) * times_t_movav,
        y2=(msd_t_movav - msd_t_movav_sd) * times_t_movav,
        color=color_movav,
        edgecolor=None,
        alpha=leap.plot.ALPHA / 2,
        rasterized=True,
    )
    ax.plot(
        times_t_movav,
        msd_t_movav * times_t_movav,
        label=label_movav,
        color=color_movav,
        linestyle=ls_movav,
        alpha=leap.plot.ALPHA,
    )
    ax.fill_between(
        times_fit,
        y1=msd_fit + msd_fit_sd,
        y2=msd_fit - msd_fit_sd,
        color=color_fit,
        edgecolor=None,
        alpha=leap.plot.ALPHA / 2,
        rasterized=True,
    )
    ax.plot(
        times_fit,
        msd_fit,
        label=label_fit,
        color=color_fit,
        linestyle=ls_fit,
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
    # Log scale y.
    ax.relim()
    ax.autoscale()
    ax.set_xscale("linear")
    ax.set_xlim(times[0], times[-1])
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    legend = ax.legend(loc="lower right")
    pdf.savefig()
    # Log scale xy.
    ax.set_xlim(times[1], times[-1])
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.legend(loc="upper left")
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
        alpha=leap.plot.ALPHA,
    )
    ax.fill_between(
        times_t_movav,
        y1=msd_t_movav + msd_t_movav_sd,
        y2=msd_t_movav - msd_t_movav_sd,
        color=color_movav,
        edgecolor=None,
        alpha=leap.plot.ALPHA / 2,
        rasterized=True,
    )
    ax.plot(
        times_t_movav,
        msd_t_movav,
        label=label_movav,
        color=color_movav,
        linestyle=ls_movav,
        alpha=leap.plot.ALPHA,
    )
    ax.fill_between(
        times_fit,
        y1=(msd_fit + msd_fit_sd) / times_fit,
        y2=(msd_fit - msd_fit_sd) / times_fit,
        color=color_fit,
        edgecolor=None,
        alpha=leap.plot.ALPHA / 2,
        rasterized=True,
    )
    ax.plot(
        times_fit,
        msd_fit / times_fit,
        label=label_fit,
        color=color_fit,
        linestyle=ls_fit,
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel=r"Diffusion Time $t$ / ns",
        ylabel=r"MSD$(t)/t$ / nm$^2$ ns$^{-1}$",
        xlim=(times_t[0], times_t[-1]),
    )
    ax.legend(loc="upper right")
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale y.
    ax.set_xscale("linear")
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale xy.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot fit residuals vs time.
    fig, ax = plt.subplots(clear=True)
    ax.plot(times_fit, msd_fit_res, color=color_orig, alpha=leap.plot.ALPHA)
    ax.set(
        xlabel="Diffusion Time / ns",
        ylabel=r"Fit Residuals / nm$^2$",
        xlim=(times_fit[0], times_fit[-1]),
    )
    pdf.savefig()
    # Log scale x.
    ax.set_xlim(times_fit[1], times[-1])
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot fit residuals / t vs time.
    fig, ax = plt.subplots(clear=True)
    ax.plot(
        times_fit,
        msd_fit_res / times_fit,
        color=color_orig,
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel=r"Diffusion Time $t$ / ns",
        ylabel=r"Fit Res. / $t$ / nm$^2$ ns$^{-1}$",
        xlim=(times_fit[0], times_fit[-1]),
    )
    pdf.savefig()
    # Log scale x.
    ax.set_xlim(times_fit[1], times[-1])
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot fit residuals / msd vs time.
    fig, ax = plt.subplots(clear=True)
    ax.plot(
        times_fit,
        msd_fit_res / msd[fit_start + 1 : fit_stop + 1],
        color=color_orig,
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel="Diffusion Time / ns",
        ylabel="Fit Residuals / MSD",
        xlim=(times_fit[0], times_fit[-1]),
    )
    pdf.savefig()
    # Log scale x.
    ax.set_xlim(times_fit[1], times[-1])
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot uncertainty of the moving average.
    fig, ax = plt.subplots(clear=True)
    ax.plot(
        times_t_movav[: fit_start - movav_t_shift],
        msd_t_movav_sd[: fit_start - movav_t_shift],
        color=color_movav,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t_movav[fit_stop - movav_t_shift :],
        msd_t_movav_sd[fit_stop - movav_t_shift :],
        color=color_movav,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t_movav[fit_start - movav_t_shift : fit_stop - movav_t_shift],
        msd_t_movav_sd[fit_start - movav_t_shift : fit_stop - movav_t_shift],
        color=color_fit,
        label="Fit Region",
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel="Diffusion Time / ns",
        ylabel=r"SE of Mov. Avg. / nm$^2$ ns$^{-1}$",
        xlim=(times_t_movav[0], times_t_movav[-1]),
    )
    ax.legend(loc="upper right")
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale y.
    ax.set_xscale("linear")
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale xy.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot covariance of moving average.
    fig, ax = plt.subplots(clear=True)
    ax.plot(
        times_t_movav_grad[: fit_start - movav_t_shift + 1],
        msd_t_movav_cov[: fit_start - movav_t_shift + 1],
        color=color_movav,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t_movav_grad[fit_stop - movav_t_shift + 1 :],
        msd_t_movav_cov[fit_stop - movav_t_shift + 1 :],
        color=color_movav,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t_movav_grad[
            fit_start - movav_t_shift + 1 : fit_stop - movav_t_shift + 1
        ],
        msd_t_movav_cov[
            fit_start - movav_t_shift + 1 : fit_stop - movav_t_shift + 1
        ],
        color=color_fit,
        label="Fit Region",
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel="Diffusion Time / ns",
        ylabel=r"Cov. of Mov. Avg. / nm$^4$ ns$^{-2}$",
        xlim=(times_t_movav_grad[0], times_t_movav_grad[-1]),
    )
    ax.legend(loc="upper right")
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale y.
    ax.set_xscale("linear")
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale xy.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot derivative of moving average of MSD/time vs time.
    fig, ax = plt.subplots(clear=True)
    ax.axhline(0, color="black", linestyle="dashed")
    ax.fill_between(
        times_t_movav_grad,
        y1=msd_t_movav_grad + msd_t_movav_grad_sd,
        y2=msd_t_movav_grad - msd_t_movav_grad_sd,
        color=color_movav,
        edgecolor=None,
        alpha=leap.plot.ALPHA / 2,
        rasterized=True,
    )
    ax.plot(
        times_t_movav_grad[: fit_start - movav_t_shift + 1],
        msd_t_movav_grad[: fit_start - movav_t_shift + 1],
        color=color_movav,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t_movav_grad[fit_stop - movav_t_shift + 1 :],
        msd_t_movav_grad[fit_stop - movav_t_shift + 1 :],
        color=color_movav,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t_movav_grad[
            fit_start - movav_t_shift + 1 : fit_stop - movav_t_shift + 1
        ],
        msd_t_movav_grad[
            fit_start - movav_t_shift + 1 : fit_stop - movav_t_shift + 1
        ],
        color=color_fit,
        label="Fit Region",
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel="Diffusion Time / ns",
        ylabel=r"d$/$d$t$ Mov. Avg. / nm$^2$ ns$^{-2}$",
        xlim=(times_t_movav_grad[0], times_t_movav_grad[-1]),
    )
    ax.axvline(
        times_t[fit_stop_tot],
        label="Fit Cutoff",
        color=color_fit_stop_tot,
        alpha=leap.plot.ALPHA,
    )
    ax.legend(loc="lower center")
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Linear scale, zoom to 0 +/- 2*tolerance.
    ax.plot(
        times_t_movav_grad,
        -msd_t_movav_grad_sd,
        label=r"$%.2f \sigma$ CI" % deriv_sd_factor,
        color=color_thresh,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t_movav_grad,
        msd_t_movav_grad_sd,
        color=color_thresh,
        alpha=leap.plot.ALPHA,
    )
    ax.set_xscale("linear")
    ax.set_ylim(-10 * msd_t_movav_grad_sd_min, 10 * msd_t_movav_grad_sd_min)
    ax.legend(loc="upper center")
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.legend(loc="upper left")
    pdf.savefig()
    plt.close()

    # Plot uncertainty of the derivative of the moving average.
    fig, ax = plt.subplots(clear=True)
    ax.plot(
        times_t_movav_grad[: fit_start - movav_t_shift + 1],
        msd_t_movav_grad_sd[: fit_start - movav_t_shift + 1],
        color=color_movav,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t_movav_grad[fit_stop - movav_t_shift + 1 :],
        msd_t_movav_grad_sd[fit_stop - movav_t_shift + 1 :],
        color=color_movav,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t_movav_grad[
            fit_start - movav_t_shift + 1 : fit_stop - movav_t_shift + 1
        ],
        msd_t_movav_grad_sd[
            fit_start - movav_t_shift + 1 : fit_stop - movav_t_shift + 1
        ],
        color=color_fit,
        label="Fit Region",
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel="Diffusion Time / ns",
        ylabel="SE of Deriv. of Mov. Avg.",
        xlim=(times_t_movav_grad[0], times_t_movav_grad[-1]),
    )
    ax.legend(loc="upper right")
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale y.
    ax.set_xscale("linear")
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale xy.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()


print("Created {}".format(outfile_pdf))
print("Done")
