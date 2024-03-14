#!/usr/bin/env python3


"""
Plot the mean displacement, the mean squared displacement and the
displacement variance as function of the diffusion time for various bins
for a single simulation.
"""


# Standard libraries
import argparse
import os

# Third-party libraries
import matplotlib as mpl
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Circumvent "OverflowError: Exceeded cell block limit in Agg."
# https://github.com/matplotlib/matplotlib/issues/5907#issuecomment-179001811
mpl.rcParams["agg.path.chunksize"] = 10000


def fit_msd_slope1(xdata, ydata, start=0, stop=-1):
    """
    Fit the logarithmic `ydata` as function of the logarithmic `xdata`
    with a straight line with slope 1.

    The obtained fit parameters are converted from the parameters of a
    straight line to the parameters of the corresponding power law.
    """
    xdata = np.log(xdata[start:stop])
    ydata = np.log(ydata[start:stop])
    valid = np.isfinite(xdata) & np.isfinite(ydata)
    if np.count_nonzero(valid) < 2:
        raise ValueError("The fitting requires at least two valid data points")
    xdata, ydata = xdata[valid], ydata[valid]
    popt, pcov = curve_fit(
        f=lambda x, c: leap.misc.straight_line(x=x, m=1, c=c),
        xdata=xdata,
        ydata=ydata,
        p0=xdata[0],
    )
    perr = np.sqrt(np.diag(pcov))
    # Convert straight-line parameters to the corresponding power-law
    # parameters.
    popt = np.array([np.exp(popt[0])])
    # Propagation of uncertainty.
    # Std[exp(A)] = |exp(A)| * Std[A]
    perr = np.array([np.abs(popt[0]) * perr[0]])
    return popt, perr


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the mean displacement and the displacement variance as function"
        " of the diffusion time for various bins for a single simulation."
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
    default="z",
    choices=("x", "y", "z"),
    help="The MSD component to use for the analysis.  Default: %(default)s",
)
args = parser.parse_args()

analysis = "msd_layer"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
ana_path = os.path.join(analysis, analysis + analysis_suffix)
tool = "mdt"  # Analysis software.
outfile = (
    args.settings
    + "_"
    + args.system
    + "_"
    + args.cmp
    + "_displvar"
    + args.msd_component
    + "_layer.pdf"
)


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
times, bins, md_data, msd_data_true = leap.simulation.read_displvar_single(
    Sim, args.cmp, args.msd_component, displvar=False
)
# Calculate displacement variance.
msd_data = msd_data_true - md_data**2
# Discard fist lag time of zero, because plots use log scale.
times = times[1:]
md_data = md_data[1:]
msd_data = msd_data[1:]
msd_data_true = msd_data_true[1:]  # True MSD, not displacement variance
msd_0_min = np.nanmin(msd_data[0])
msd_0_min_true = np.nanmin(msd_data_true[0])


print("Fitting straight line...")
bin_nums = np.arange(1, len(bins))
bin_ix = len(bin_nums) // 2 - 1
fit_start, fit_stop = int(np.sqrt(len(times))), int(0.98 * len(times))
popt, perr = fit_msd_slope1(
    times, msd_data[:, bin_ix], start=fit_start, stop=fit_stop
)
popt_true, perr_true = fit_msd_slope1(
    times, msd_data_true[:, bin_ix], start=fit_start, stop=fit_stop
)


print("Creating plot(s)...")
xlabel = r"Diffusion Time $\Delta t$ / ns"
xlim = (times[0], times[-1])

if args.cmp in ("NBT", "OBT", "OE"):
    legend_title = (
        "$" + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp] + "$" + ", "
    )
else:
    legend_title = leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp] + ", "
if surfq is not None:
    legend_title += r"$\sigma_s = \pm %.2f$ $e$/nm$^2$" % surfq + "\n"
legend_title += (
    r"$n_{EO} = %d$, " % Sim.O_per_chain
    + r"$r = %.4f$" % Sim.Li_O_ratio
    + "\n"
    + "Bin Number"
)

invalid = np.all(np.isnan(msd_data), axis=0)
n_bins_valid = len(invalid) - np.count_nonzero(invalid)
if n_bins_valid == 0:
    raise ValueError("The displacements in all bins are NaN")
cmap = plt.get_cmap()
c_vals = np.arange(n_bins_valid)
c_norm = max(1, n_bins_valid - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot mean displacement vs time.
    ylabel = (
        r"$\langle \Delta %s (\Delta t) \rangle$ / nm" % args.msd_component
    )
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    ax.axhline(y=0, color="black")
    md_min, md_max = 0, 0
    for bix, bin_num in enumerate(bin_nums):
        if np.all(np.isnan(md_data[:, bix])):
            continue
        # Discard last data points, because they are quite noisy and
        # tend to distort the automatic y axis scaling.
        ax.plot(
            times[:fit_stop],
            md_data[:, bix][:fit_stop],
            label=r"$%d$" % bin_num,
            alpha=leap.plot.ALPHA,
            rasterized=True,
        )
        md_min = min(md_min, np.nanmin(md_data[:, bix][:fit_stop]))
        md_max = max(md_max, np.nanmax(md_data[:, bix][:fit_stop]))
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    if md_max >= abs(md_min):
        legend_loc = "upper left"
    else:
        legend_loc = "lower left"
    legend = ax.legend(
        title=legend_title,
        loc=legend_loc,
        ncol=1 + n_bins_valid // 4,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot true MSD vs time.
    ylabel = (
        r"$\langle \Delta %s^2 (\Delta t) \rangle$ / nm$^2$"
        % args.msd_component
    )
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for bix, bin_num in enumerate(bin_nums):
        if np.all(np.isnan(msd_data_true[:, bix])):
            continue
        ax.plot(
            times[:fit_stop],
            msd_data_true[:, bix][:fit_stop],
            label=r"$%d$" % bin_num,
            alpha=leap.plot.ALPHA,
            rasterized=True,
        )
    ax.plot(
        times[fit_start:fit_stop],
        leap.misc.power_law(times[fit_start:fit_stop], m=1, c=popt_true[0]),
        label=r"$\propto \Delta t$",
        color="black",
        linestyle="dashed",
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    ylim = ax.get_ylim()
    if ylim[0] < msd_0_min_true / 2:
        ax.set_ylim(msd_0_min_true / 2, ylim[1])
    legend = ax.legend(
        title=legend_title,
        loc="upper left",
        ncol=1 + n_bins_valid // 5,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot displacement variance vs time.
    ylabel = r"Var$[\Delta %s (\Delta t)]$ / nm$^2$" % args.msd_component
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for bix, bin_num in enumerate(bin_nums):
        if np.all(np.isnan(msd_data[:, bix])):
            continue
        ax.plot(
            times[:fit_stop],
            msd_data[:, bix][:fit_stop],
            label=r"$%d$" % bin_num,
            alpha=leap.plot.ALPHA,
            rasterized=True,
        )
    ax.plot(
        times[fit_start:fit_stop],
        leap.misc.power_law(times[fit_start:fit_stop], m=1, c=popt[0]),
        label=r"$\propto \Delta t$",
        color="black",
        linestyle="dashed",
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    ylim = ax.get_ylim()
    if ylim[0] < msd_0_min / 2:
        ax.set_ylim(msd_0_min / 2, ylim[1])
    legend = ax.legend(
        title=legend_title,
        loc="upper left",
        ncol=1 + n_bins_valid // 5,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
