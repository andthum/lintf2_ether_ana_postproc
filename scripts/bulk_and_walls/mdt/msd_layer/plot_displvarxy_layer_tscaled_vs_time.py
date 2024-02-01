#!/usr/bin/env python3


"""
Scale and plot the displacement variance of the lithium ions in the
first layer at the right (negative) electrode such that it overlaps with
the corresponding displacement variance in a bulk layer for a single
simulation.
"""


# Standard libraries
import argparse
import os

# Third-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# First-party libraries
import lintf2_ether_ana_postproc as leap


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


def fit_msd(xdata, ydata, start=0, stop=-1):
    """
    Fit the logarithmic `ydata` as function of the logarithmic `xdata`
    with a straight line.

    The obtained fit parameters are converted from the parameters of a
    straight line to the parameters of the corresponding power law.
    """
    xdata = np.log(xdata[start:stop])
    ydata = np.log(ydata[start:stop])
    valid = np.isfinite(xdata) & np.isfinite(ydata)
    slope_guess = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
    intercept_guess = ydata[0] - slope_guess * xdata[0]
    if np.count_nonzero(valid) < 2:
        raise ValueError("The fitting requires at least two valid data points")
    xdata, ydata = xdata[valid], ydata[valid]
    popt, pcov = curve_fit(
        f=leap.misc.straight_line,
        xdata=xdata,
        ydata=ydata,
        p0=(slope_guess, intercept_guess),
    )
    perr = np.sqrt(np.diag(pcov))
    # Convert straight-line parameters to the corresponding power-law
    # parameters.
    popt = np.array([popt[0], np.exp(popt[1])])
    # Propagation of uncertainty.
    # Std[exp(A)] = |exp(A)| * Std[A]
    perr = np.array([perr[0], np.abs(popt[1]) * perr[1]])
    return popt, perr


def cost_func(times, msd, msd_ref, t0):
    r"""
    Cost function to quantify the overlap between a MSD curve with a
    reference MSD curve.

    Parameters
    ----------
    times, msd, msd_ref : array_like
        Lag times, corresponding MSD values and corresponding reference
        MSD values.
    t0 : scalar
        Scaling factor to scale the lag times such that the MSD becomes
        equal to the reference MSD.

    Returns
    -------
    cost : scalar
        A value quantifying the difference of the time-scaled MSD and
        the reference MSD.

    Notes
    -----
    The used cost function is

    .. math::

        f(t_0) = \sum_i
        \left( \frac{MSD_i^{ref}}{t_i} - t_0 \frac{MSD_i}{t_i} \right)^2

    where the index :math:`i` labels the measured data points.

    See notes in my "sketch book" from 22.01.2021.
    """
    times = np.asarray(times)
    msd = np.asarray(msd)
    msd_ref = np.asarray(msd_ref)

    cost = msd_ref / times
    cost -= t0 * msd / times
    cost **= 2
    return cost


def opt_scaling(times, msd, msd_ref):
    r"""
    Calculate the optimal time scaling factor such the MSD has the
    greatest possible overlap with the reference MSD.

    Parameters
    ----------
    times, msd, msd_ref : array_like
        Lag times, corresponding MSD values and corresponding reference
        MSD values.

    Returns
    -------
    t0 : float
        The optimal scaling factor.

    Notes
    -----
    The optimal scaling factor is calculated by settings the derivative
    of the cost function (:func:`cost_func`) with respect to the scaling
    factor :math:`t_0` to zero and resolving for :math:`t_0`.

    .. math::

        -2 \sum_i
        \left(
            \frac{MSD_i MSD_i^{ref}}{t_i^2} -
            t_0 \frac{MSD_i^2}{t_i^2}
        \right)
        = 0

    .. math::

        t_0 = \frac{
            \frac{MSD_i MSD_i^{ref}}{t_i^2}
        }{
            \frac{MSD_i^2}{t_i^2}
        }

    The index :math:`i` labels the measured data points.

    See notes in my "sketch book" from 22.01.2021.
    """
    times = np.asarray(times)
    msd = np.asarray(msd)
    msd_ref = np.asarray(msd_ref)

    numerator = np.sum(msd * msd_ref / times**2)
    denominator = np.sum(msd**2 / times**2)
    return numerator / denominator


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
    "--msd-component",
    type=str,
    required=False,
    default="xy",
    choices=("xy", "z"),
    help="The MSD component to use for the analysis.  Default: %(default)s",
)
args = parser.parse_args()

cmp = "Li"
analysis = "msd_layer"  # Analysis name.
analysis_suffix = "_" + cmp  # Analysis name specification.
ana_path = os.path.join(analysis, analysis + analysis_suffix)
tool = "mdt"  # Analysis software.
outfile = (
    args.settings
    + "_"
    + args.system
    + "_"
    + cmp
    + "_displvar"
    + args.msd_component
    + "_layer_tscaled.pdf"
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
if args.msd_component == "xy":
    dimensions = ("x", "y")
elif args.msd_component == "z":
    dimensions = ("z",)
else:
    raise ValueError(
        "Unknown --msd-component: '{}'".format(args.msd_component)
    )
md = [None for dim in dimensions]
msd = [None for dim in dimensions]
for dim_ix, dim in enumerate(dimensions):
    (
        times_dim,
        bins_dim,
        md_data,
        msd_data,
    ) = leap.simulation.read_displvar_single(Sim, cmp, dim)
    stop = int(0.99 * len(times_dim))
    if dim_ix == 0:
        # Discard fist lag time of zero, because plots use log scale.
        # Discard last one percent of the data, because they are quite
        # noisy.
        times, bins = times_dim[1:stop], bins_dim
        md, msd = md_data[1:stop], msd_data[1:stop]
    else:
        if bins_dim.shape != bins.shape:
            raise ValueError(
                "The input files do not contain the same number of bins"
            )
        if not np.allclose(bins_dim, bins, atol=0):
            raise ValueError(
                "The bin edges are not the same in all input files"
            )
        times_dim = times_dim[1:stop]
        if times_dim.shape != times.shape:
            raise ValueError(
                "The input files do not contain the same number of lag times"
            )
        if not np.allclose(times_dim, times, atol=0):
            raise ValueError(
                "The lag times are not the same in all input files"
            )
        md += md_data[1:stop]
        msd += msd_data[1:stop]
del times_dim, bins_dim, md_data, msd_data

# Get displacements in the first layer at the right (negative) electrode
# and in the bulk.
valid_bins = np.isfinite(msd)
valid_bins &= msd > 0
valid_bins = np.any(valid_bins, axis=0)
if np.count_nonzero(valid_bins) == 0:
    raise ValueError(
        "The displacement variances in all bins are NaN, inf or less than zero"
    )
last_bin = np.flatnonzero(valid_bins)[-1]
bulk_bin = msd.shape[1] // 2 - 1
md_wall, md_bulk = md[:, last_bin], md[:, bulk_bin]
msd_wall, msd_bulk = msd[:, last_bin], msd[:, bulk_bin]
del md, msd


print("Calculating scaling factor...")
# Scale the lag times of the displacement variance in the first layer at
# the right (negative) electrode such that it has the greatest possible
# overlap with the displacement variance in the bulk layer.
# # Only use the first part of the MSD to determine the scaling factor,
# # because the scaling factor is heavily influenced by the noise in the
# # later part of the MSD.
# _, scale_stop = mdt.nph.find_nearest(times, 1e2, return_index=True)
scale_stop = len(times)
tscales = np.zeros(2, dtype=np.float64)
tscales[1] = opt_scaling(
    times=times[:scale_stop],
    msd=msd_wall[:scale_stop],
    msd_ref=msd_bulk[:scale_stop],
)
tscales[0] = opt_scaling(
    times=times[:scale_stop],
    msd=msd_bulk[:scale_stop],
    msd_ref=msd_bulk[:scale_stop],
)


print("Fitting straight line...")
fit_start, fit_stop = int(np.sqrt(len(times))), len(times)
popt, perr = fit_msd_slope1(times, msd_bulk, start=fit_start, stop=fit_stop)

if np.isclose(Sim.Li_O_ratio, 1 / 20, rtol=0) and Sim.O_per_chain >= 16:
    _, fit_start_region1 = mdt.nph.find_nearest(times, 2e-3, return_index=True)
    _, fit_stop_region1 = mdt.nph.find_nearest(times, 2e-2, return_index=True)
    popt_region1, perr_region1 = fit_msd(
        times, msd_bulk, start=fit_start_region1, stop=fit_stop_region1
    )
    _, fit_start_region2 = mdt.nph.find_nearest(times, 1e-1, return_index=True)
    _, fit_stop_region2 = mdt.nph.find_nearest(times, 1e1, return_index=True)
    popt_region2, perr_region2 = fit_msd(
        times, msd_bulk, start=fit_start_region2, stop=fit_stop_region2
    )


print("Creating plot(s)...")
xlabel = r"Scaled Diffusion Time $\Delta t/t^{*}$"
xlim = (times[0] / 4, times[-1])

if cmp in ("NBT", "OBT", "OE"):
    legend_title = "$" + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp] + "$" + ", "
else:
    legend_title = leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp] + ", "
if surfq is not None:
    legend_title += r"$\sigma_s = \pm %.2f$ $e$/nm$^2$" % surfq + "\n"
legend_title += (
    r"$n_{EO} = %d$, " % Sim.O_per_chain + r"$r = %.4f$" % Sim.Li_O_ratio
)


mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot mean displacement vs scaled time.
    ylabel = r"$\langle \Delta "
    if args.msd_component == "xy":
        ylabel += r"%s \rangle" % args.msd_component[0]
        ylabel += r"+ \langle \Delta %s" % args.msd_component[1]
    elif args.msd_component == "z":
        ylabel += r"%s" % args.msd_component
    else:
        raise ValueError(
            "Unknown --msd-component: '{}'".format(args.msd_component)
        )
    ylabel += r" \rangle$ / nm"
    fig, ax = plt.subplots(clear=True)
    ax.axhline(y=0, color="black")
    md_min, md_max = 0, 0
    for ix, (bin_ix, md) in enumerate(
        zip((bulk_bin, last_bin), (md_bulk, md_wall))
    ):
        ax.plot(
            times / tscales[ix],
            md,
            label=(
                r"Bin $%d$, " % (bin_ix + 1)
                + r"$t^{*} = %.2f$ ns" % tscales[ix]
            ),
            alpha=leap.plot.ALPHA,
        )
        md_min = min(md_min, np.nanmin(md))
        md_max = max(md_max, np.nanmax(md))
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    if md_max >= abs(md_min):
        legend_loc = "upper left"
    else:
        legend_loc = "lower left"
    legend = ax.legend(
        title=legend_title, loc=legend_loc, **mdtplt.LEGEND_KWARGS_XSMALL
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot displacement variance vs scaled time.
    ylabel = r"Var$[\Delta "
    if args.msd_component == "xy":
        ylabel += r"\mathbf{r}_{%s}" % args.msd_component
    elif args.msd_component == "z":
        ylabel += r"%s" % args.msd_component
    else:
        raise ValueError(
            "Unknown --msd-component: '{}'".format(args.msd_component)
        )
    ylabel += r"]$ / nm$^2$"
    fig, ax = plt.subplots(clear=True)
    for ix, (bin_ix, msd) in enumerate(
        zip((bulk_bin, last_bin), (msd_bulk, msd_wall))
    ):
        ax.plot(
            times / tscales[ix],
            msd,
            label=(
                r"Bin $%d$, " % (bin_ix + 1)
                + r"$t^{*} = %.2f$ ns" % tscales[ix]
            ),
            alpha=leap.plot.ALPHA,
        )
    ax.plot(
        times[fit_start:fit_stop],
        leap.misc.power_law(times[fit_start:fit_stop], m=1, c=popt[0]),
        label=r"$\propto \Delta t^{1.00}$",
        color="black",
        linestyle="dashed",
    )
    if np.isclose(Sim.Li_O_ratio, 1 / 20, rtol=0) and Sim.O_per_chain >= 16:
        fit = leap.misc.power_law(
            times[fit_start_region1:fit_stop_region1], *popt_region1
        )
        ax.plot(
            times[fit_start_region1:fit_stop_region1],
            fit * 1.5,
            label=r"$\propto \Delta t^{%.2f}$" % popt_region1[0],
            linestyle="dashdot",
        )
        fit = leap.misc.power_law(
            times[fit_start_region2:fit_stop_region2], *popt_region2
        )
        ax.plot(
            times[fit_start_region2:fit_stop_region2],
            fit * 1.5,
            label=r"$\propto \Delta t^{%.2f}$" % popt_region2[0],
            linestyle="dashdot",
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    legend = ax.legend(
        title=legend_title, loc="upper left", **mdtplt.LEGEND_KWARGS_XSMALL
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
