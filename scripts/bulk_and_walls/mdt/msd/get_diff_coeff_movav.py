#!/usr/bin/env python3

"""
Identify the diffusive regime of the mean squared displacement (MSD) and
extract the self-diffusion coefficient by fitting a straight line to it.
"""


# Standard libraries
import argparse

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
        Number of dimensions :math:`d` in which the diffusion process
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

# Window length for calculating the moving average of MSD/t.
# Set to the number of frames between restarting points used for
# calculating the MSD.
movav_wlen = 501
if movav_wlen % 2 == 0:
    # Ensure that `movav_wlen` is odd.
    movav_wlen += 1
# Regard the derivative of the moving average of MSD/t as zero if it
# lies within +/- the minimum uncertainty of the derivative times
# `deriv_sd_factor`.
msd_t_movav_grad_sd_min_fac = 0.5
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
discard = (movav_wlen - 1) // 2
times_t_movav = times_t[discard : len(times_t) - discard]

# Calculate the moving average of MSD/t.
msd_t_movav = mdt.stats.movav(msd_t, wlen=movav_wlen)

# Calculate the derivative of the moving average.
# Use `np.diff` instead of `numpy.gradient`, because otherwise the
# propagation of uncertainty is unclear.
msd_t_movav_grad = np.diff(msd_t_movav, prepend=msd_t_movav[0])
msd_t_movav_grad /= time_diff

# Calculate the variance (squared uncertainty) of the moving average.
msd_t_movav_var = mdt.stats.movav(msd_t**2, wlen=movav_wlen)
msd_t_movav_var -= msd_t_movav**2
msd_t_movav_var /= movav_wlen - 1

# Calculate the uncertainty of the derivative of the moving average.
msd_t_movav_grad_sd = msd_t_movav_var[:-1] + msd_t_movav_var[1:]
msd_t_movav_grad_sd = np.sqrt(msd_t_movav_grad_sd, out=msd_t_movav_grad_sd)
msd_t_movav_grad_sd /= time_diff
msd_t_movav_grad_sd = np.insert(msd_t_movav_grad_sd, 0, msd_t_movav_grad_sd[0])
msd_t_movav_grad_sd_min = np.min(msd_t_movav_grad_sd)
msd_t_movav_grad_sd_min_ix = np.argmin(msd_t_movav_grad_sd)

# Convert the uncertainty of the moving average.
msd_t_movav_sd = np.sqrt(msd_t_movav_var, out=msd_t_movav_var)
del msd_t_movav_var

deriv_tol = msd_t_movav_grad_sd_min_fac * msd_t_movav_grad_sd_min
diffusive = (msd_t_movav_grad >= -deriv_tol) & (msd_t_movav_grad <= deriv_tol)

if not np.any(diffusive):
    fit_start, fit_stop = -1, -1
else:
    first_true = np.argmax(diffusive)
    fit_start, length, _value = mdt.nph.find_const_seq_long(
        diffusive[first_true:], tol=0.5
    )
    fit_start += first_true + discard
    fit_stop = fit_start + length + discard

fit_stop_tot = int(fit_stop_pct * len(msd_t))
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
else:
    diff_coeff, diff_coeff_sd = np.nan, np.nan
    times_fit, msd_fit = np.array([]), np.array([])


print("Creating output...")
header = ""
# TODO


print("Creating plots...")
label_orig = "Original"
label_movav = "Mov. Avg."
label_fit = "Fit"
color_orig = "tab:blue"
color_movav = "tab:orange"
color_fit = "black"
color_thresh = "tab:red"
color_msd_t_movav_grad_sd_min_ix = "tab:green"
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
        alpha=leap.plot.ALPHA,
    )
    ax.fill_between(
        times_t_movav,
        y1=msd_t_movav + msd_t_movav_sd,
        y2=msd_t_movav - msd_t_movav_sd,
        color=color_movav,
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

    # Plot uncertainty of the moving average.
    fig, ax = plt.subplots(clear=True)
    ax.plot(
        times_t_movav, msd_t_movav_sd, color=color_movav, alpha=leap.plot.ALPHA
    )
    ax.set(
        xlabel="Diffusion Time / ns",
        ylabel="SE of Mov. Avg.",
        xlim=(times_t[discard // 2], times_t[-discard // 2]),
    )
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale xy.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot derivative of moving average of MSD/time vs time.
    fig, ax = plt.subplots(clear=True)
    # ax.fill_between(
    #     times_t_movav,
    #     y1=msd_t_movav_grad + msd_t_movav_grad_sd,
    #     y2=msd_t_movav_grad - msd_t_movav_grad_sd,
    #     color=color_movav,
    #     alpha=leap.plot.ALPHA / 2,
    #     rasterized=True,
    # )
    ax.plot(
        times_t_movav[: fit_start - discard],
        msd_t_movav_grad[: fit_start - discard],
        color=color_movav,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t_movav[fit_stop - discard :],
        msd_t_movav_grad[fit_stop - discard :],
        color=color_movav,
        alpha=leap.plot.ALPHA,
    )
    ax.plot(
        times_t_movav[fit_start - discard : fit_stop - discard],
        msd_t_movav_grad[fit_start - discard : fit_stop - discard],
        color=color_fit,
        label="Fit Region",
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel="Diffusion Time / ns",
        ylabel="Derivative of Mov. Avg.",
        xlim=(times_t[discard // 2], times_t[-discard // 2]),
    )
    ax.axvline(
        times_t[fit_stop_tot],
        label="Fit Stop",
        color=color_fit_stop_tot,
        alpha=leap.plot.ALPHA,
    )
    # ax.axvline(
    #     times_t[msd_t_movav_grad_sd_min_ix],
    #     label=r"$\sigma_\mathrm{min}$",
    #     color=color_msd_t_movav_grad_sd_min_ix,
    #     alpha=leap.plot.ALPHA,
    # )
    ax.legend(loc="lower center")
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Linear scale, zoom to 0 +/- 2*tolerance.
    ax.axhline(
        -deriv_tol,
        label=(
            r"Tolerance ($%.1f \sigma_\mathrm{min}$)"
            % msd_t_movav_grad_sd_min_fac
        ),
        color=color_thresh,
        alpha=leap.plot.ALPHA,
    )
    ax.axhline(deriv_tol, color=color_thresh, alpha=leap.plot.ALPHA)
    ax.set_xscale("linear")
    ax.set_ylim(-3 * deriv_tol, 3 * deriv_tol)
    ax.legend(loc="upper left", **mdtplt.LEGEND_KWARGS_XSMALL)
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Plot uncertainty of the derivative of the moving average.
    fig, ax = plt.subplots(clear=True)
    ax.plot(
        times_t_movav,
        msd_t_movav_grad_sd,
        color=color_movav,
        alpha=leap.plot.ALPHA,
    )
    ax.axvline(
        times_t[msd_t_movav_grad_sd_min_ix],
        label=r"$\sigma_\mathrm{min}$",
        color=color_msd_t_movav_grad_sd_min_ix,
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel="Diffusion Time / ns",
        ylabel="SE of Deriv. of Mov. Avg.",
        xlim=(times_t[discard // 2], times_t[-discard // 2]),
    )
    ax.legend(loc="upper center")
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale xy.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()


print("Created {}".format(outfile_pdf))
print("Done")
