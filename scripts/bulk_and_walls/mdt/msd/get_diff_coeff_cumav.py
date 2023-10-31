#!/usr/bin/env python3

"""
.. warning::

    This script is work in progress!

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
fit_region_test_step = 50
fit_stop_pct = 0.9
fit_stop = int(fit_stop_pct * len(msd_t))
# fit_start_prev = fit_stop
# diff_coeff_sd_prev = np.inf
# for fit_start_test in range(fit_stop - 3, -1, -fit_region_test_step):
#     diff_coeff_sd = np.std(
#         msd_t[fit_stop - 1 : fit_start_test : -1], ddof=1
#     )
#     if diff_coeff_sd > diff_coeff_sd_prev:
#         break
#     fit_start_prev = fit_start_test
#     diff_coeff_sd_prev = diff_coeff_sd
# fit_start = fit_start_prev

# Reverse cumulative mean.
cum_av = mdt.stats.cumav(msd_t[::-1])
# Reverse uncertainty of the cumulative mean.
cum_sd = mdt.stats.cumav(msd_t[::-1] ** 2)
cum_sd -= cum_av**2
cum_sd = np.sqrt(cum_sd, out=cum_sd)
cum_sd /= np.sqrt(np.arange(1, len(msd_t) + 1))
# Discard mean and average over just one data point.
cum_av = cum_av[1:]
cum_sd = cum_sd[1:]

fit_start = np.argmin(cum_sd)
fit_stop = 0.9

if fit_start >= fit_stop:
    fit_start, fit_stop = -1, -1
    diff_coeff, diff_coeff_sd = np.nan, np.nan
    times_fit, msd_fit = np.array([]), np.array([])
else:
    diff_coeff = np.mean(msd_t[fit_start:fit_stop]) / (2 * n_dim)
    diff_coeff_sd = np.std(msd_t[fit_start:fit_stop], ddof=1)
    diff_coeff_sd /= 2 * n_dim
    times_fit = times_t[fit_start:fit_stop]
    msd_fit = einstein_msd(times_fit, diff_coeff, n_dim)


print("Creating output...")
header = ""
# TODO


print("Creating plots...")
label_fit = "Fit"
color_fit = "tab:orange"
color_thresh = "tab:red"
ls_fit = "dashed"

mdt.fh.backup(outfile_pdf)
with PdfPages(outfile_pdf) as pdf:
    # Plot MSD vs time.
    fig, ax = plt.subplots(clear=True)
    ax.plot(times, msd, alpha=leap.plot.ALPHA)
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
    ax.plot(times_t, msd_t, alpha=leap.plot.ALPHA)
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

    # Plot reverse cumulative mean.
    fig, ax = plt.subplots(clear=True)
    lines = ax.plot(times_t[::-1][1:], cum_av, alpha=leap.plot.ALPHA)
    ax.fill_between(
        times_t[::-1][1:],
        y1=cum_av + 100 * cum_sd,
        y2=cum_av - 100 * cum_sd,
        color=lines[0].get_color(),
        alpha=0.25,
        rasterized=True,
    )
    ax.axvline(
        times_t[fit_start],
        color=color_thresh,
        label="Fit Start",
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel=r"Diffusion Time $t$ / ns",
        ylabel=r"Cum. Avg. MSD$(t)/t$ / nm$^2$/ns",
        xlim=(times_t[-1], times_t[0]),
    )
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    # Log scale xy.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(clear=True)
    lines = ax.plot(
        times_t[::-1][1:],
        np.gradient(cum_av, time_diff),
        alpha=leap.plot.ALPHA,
    )
    ax.fill_between(
        times_t[::-1][1:],
        y1=np.gradient(cum_av, time_diff) + cum_sd,
        y2=np.gradient(cum_av, time_diff) - cum_sd,
        color=lines[0].get_color(),
        alpha=0.25,
        rasterized=True,
    )
    ax.axvline(
        times_t[fit_start],
        color=color_thresh,
        label="Fit Start",
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel=r"Diffusion Time $t$ / ns",
        ylabel=r"Deriv. Cum. Avg. MSD$(t)/t$",
        xlim=(times_t[-1], times_t[0]),
        ylim=(-np.max(cum_sd), np.max(cum_sd)),
    )
    pdf.savefig()
    # Log scale x.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()

    # Plot reverse cumulative standard deviation.
    fig, ax = plt.subplots(clear=True)
    ax.plot(times_t[::-1][1:], cum_sd, alpha=leap.plot.ALPHA)
    ax.axvline(
        times_t[fit_start],
        color=color_thresh,
        label="Fit Start",
        alpha=leap.plot.ALPHA,
    )
    ax.set(
        xlabel=r"Diffusion Time $t$ / ns",
        ylabel=r"Cum. Std. MSD$(t)/t$ / nm$^2$/ns",
        xlim=(times_t[-1], times_t[0]),
    )
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
