#!/usr/bin/env python3


"""
Plot the average end-to-end distance and the average radius of gyration
of the PEO chains as function of the PEO chain length.
"""


# Standard libraries
import argparse

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# First-party libraries
import lintf2_ether_ana_postproc as leap


def fit_stats(xdata, ydata, ydata_sd, start=0, stop=-1):
    """
    Fit the logarithmic `ydata` as function of the logarithmic `xdata`
    with a straight line.

    The obtained fit parameters are converted from the parameters of a
    straight line to the parameters of the corresponding power law.
    """
    # Propagation of uncertainty.  See
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
    # Std[ln(A)] = Std[A] / |A|
    sd = ydata_sd[start:stop] / np.abs(ydata[start:stop])
    popt, pcov = curve_fit(
        f=leap.misc.straight_line,
        xdata=np.log(xdata[start:stop]),
        ydata=np.log(ydata[start:stop]),
        p0=(1.2, np.log(ydata[0])),
        sigma=sd,
        absolute_sigma=True,
    )
    perr = np.sqrt(np.diag(pcov))
    # Convert straight-line parameters to the corresponding power-law
    # parameters.
    popt = np.array([popt[0], np.exp(popt[1])])
    # Propagation of uncertainty.
    # Std[exp(A)] = |exp(A)| * Std[A]
    perr = np.array([perr[0], np.abs(popt[1]) * perr[1]])
    return popt, perr


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the average end-to-end distance and the average radius of"
        " gyration of the PEO chains as function of the PEO chain length."
    ),
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "polystat"  # Analysis name.
tool = "gmx"  # Analysis software.
outfile = (  # Output file name.
    settings + "_lintf2_peoN_20-1_sc80_" + analysis + ".pdf"
)

cols = (  # Columns to read from the input file(s).
    1,  # End-to-end distance [nm].
    2,  # Radius of gyration [nm].
)


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="O_per_chain"
)


print("Reading data...")
file_suffix = analysis + ".xvg.gz"
infiles = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
n_infiles = len(infiles)

polymer_stats = np.full((2, n_infiles), np.nan, dtype=np.float64)
polymer_stats_sd = np.full_like(polymer_stats, np.nan)
for sim_ix, infile in enumerate(infiles):
    stats = np.loadtxt(infile, comments=["#", "@"], usecols=cols)
    stats **= 2
    polymer_stats[:, sim_ix] = np.nanmean(stats, axis=0)
    polymer_stats_sd[:, sim_ix] = np.nanstd(stats, ddof=1, axis=0)
    polymer_stats_sd[:, sim_ix] /= np.sqrt(Sims.res_nums["solvent"][sim_ix])
del stats

ratio = polymer_stats[0] / polymer_stats[1]
# Propagation of uncertainty:
# https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
# Std[A/B] = |A/B| * sqrt{(Std[A]/A)^2 + (Std[B]/B)^2 - 2 Cov[A,B]/(AB))
ratio_sd = np.abs(polymer_stats[0] / polymer_stats[1]) * (
    (polymer_stats_sd[0] / polymer_stats[0]) ** 2
    + (polymer_stats_sd[1] / polymer_stats[1]) ** 2
)


print("Fitting power law...")
# End-to-end distance and radius of gyration.
fit_stats_starts = (0, np.flatnonzero(Sims.O_per_chain == 6)[0])
fit_stats_stops = (
    np.flatnonzero(Sims.O_per_chain == 6)[0],
    len(Sims.O_per_chain),
)
popt_stats = np.full(
    (polymer_stats.shape[0], len(fit_stats_starts), 2),
    np.nan,
    dtype=np.float64,
)
perr_stats = np.full_like(popt_stats, np.nan)
for stats_ix, stats in enumerate(polymer_stats):
    for fit_ix, start in enumerate(fit_stats_starts):
        popt_stats[stats_ix, fit_ix], perr_stats[stats_ix, fit_ix] = fit_stats(
            xdata=Sims.O_per_chain,
            ydata=stats,
            ydata_sd=polymer_stats_sd[stats_ix],
            start=start,
            stop=fit_stats_stops[fit_ix],
        )


print("Creating plot(s)...")
xlabel = r"Ether Oxygens per Chain $n_{EO}$"
xlim = (1, 200)
legend_title = r"$r = %.2f$" % Sims.Li_O_ratios[0]
labels = ("End-to-End", "Gyration")
markers = ("^", "v")
linestyles = ("dotted", "dashed")
if len(labels) != polymer_stats.shape[0]:
    raise ValueError(
        "`len(labels)` ({}) != `polymer_stats.shape[0]`"
        " ({})".format(len(labels), polymer_stats.shape[0])
    )
if len(markers) != polymer_stats.shape[0]:
    raise ValueError(
        "`len(markers)` ({}) != `polymer_stats.shape[0]`"
        " ({})".format(len(markers), polymer_stats.shape[0])
    )
if len(linestyles) != len(fit_stats_starts):
    raise ValueError(
        "`len(linestyles)` ({}) != `len(fit_stats_starts)`"
        " ({})".format(len(linestyles), len(fit_stats_starts))
    )

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot end-to-end distance and radius of gyration.
    ylabel = r"$\langle R^2 \rangle$ / nm$^2$"
    ylim = (2e-2, 5e1)
    fig, ax = plt.subplots(clear=True)
    for stats_ix, stats in enumerate(polymer_stats):
        lines = ax.errorbar(
            Sims.O_per_chain,
            stats,
            yerr=polymer_stats_sd[stats_ix],
            label=labels[stats_ix],
            marker=markers[stats_ix],
            linestyle="none",
        )
        # Fits.
        for fit_ix, popt in enumerate(popt_stats[stats_ix]):
            xdata = Sims.O_per_chain[
                fit_stats_starts[fit_ix] : fit_stats_stops[fit_ix]
            ]
            fit = leap.misc.power_law(xdata, *popt)
            ax.plot(
                xdata,
                fit,
                color=lines[0].get_color(),
                linestyle=linestyles[fit_ix],
            )
            x_position = np.sqrt(xdata[0] * xdata[-1])
            y_position = leap.misc.power_law(x_position, *popt)
            if stats_ix == 0:
                y_position += 0.2 * y_position
            else:
                y_position -= 0.2 * y_position
            ax.text(
                x_position,
                y_position,
                r"$\propto n_{EO}^{%.2f}$" % popt[0],
                rotation=np.rad2deg(np.arctan(popt[0])) / 1.8,
                rotation_mode="anchor",
                transform_rotates_text=False,
                horizontalalignment="center",
                verticalalignment="bottom" if stats_ix == 0 else "top",
                fontsize="small",
            )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    legend = ax.legend(title=legend_title)
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    yd_min, yd_max = leap.plot.get_ydata_min_max(ax)
    if np.any(np.greater(yd_min, 0)):
        # Log scale y.
        ax.relim()
        ax.autoscale()
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlim=xlim, ylim=ylim)
        pdf.savefig()
    plt.close()

    # Plot end-to-end distance divided by radius of gyration.
    ylabel = r"$\langle R_e^2 \rangle / \langle R_g^2 \rangle$"
    ylim = (None, None)
    fig, ax = plt.subplots(clear=True)
    lines = ax.errorbar(Sims.O_per_chain, ratio, yerr=ratio_sd, marker="o")
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    legend = ax.legend(title=legend_title)
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    yd_min, yd_max = leap.plot.get_ydata_min_max(ax)
    if np.any(np.greater(yd_min, 0)):
        # Log scale y.
        ax.relim()
        ax.autoscale()
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlim=xlim, ylim=ylim)
        pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
