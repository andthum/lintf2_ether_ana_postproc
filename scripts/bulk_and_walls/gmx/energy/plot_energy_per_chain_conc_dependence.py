#!/usr/bin/env python3


"""
Plot the average total energy of the system divided by the total number
of PEO chains as function of the salt concentration.
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
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit

# First-party libraries
import lintf2_ether_ana_postproc as leap


def equalize_xticks(ax):
    """
    Equalize x-ticks so that plots can be better stacked together.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` for which to equalize the x
        ticks.
    """
    ax.xaxis.set_major_locator(MultipleLocator(0.1))


def fit_energy(xdata, ydata, ydata_sd, start=0, stop=-1):
    """
    Fit the linear `ydata` as function of the linear `xdata` with a
    straight line.
    """
    xdata = xdata[start:stop]
    ydata = ydata[start:stop]
    sd = ydata_sd[start:stop]
    slope_guess = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
    intercept_guess = ydata[0] - slope_guess * xdata[0]
    popt, pcov = curve_fit(
        f=leap.misc.straight_line,
        xdata=xdata,
        ydata=ydata,
        p0=(slope_guess, intercept_guess),
        sigma=sd,
        absolute_sigma=True,
    )
    perr = np.sqrt(np.diag(pcov))
    fit = leap.misc.straight_line(xdata, *popt)
    fit_quality = leap.misc.fit_goodness(ydata, fit)
    return popt, perr, fit_quality


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the average total energy of the system divided by the total"
        " number of PEO chains as function of the salt concentration."
    ),
)
parser.add_argument(
    "--sol",
    type=str,
    required=True,
    choices=("g1", "g4", "peo63"),
    help="Solvent name.",
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
outfile = (  # Output file name.
    settings + "_lintf2_" + args.sol + "_r_sc80_energy_per_chain.pdf"
)

observables = ("Potential", "Kinetic En.", "Total Energy")
begin = 50000  # First frame to read from the .edr file.
end = -1  # Last frame to read from the .edr file (exclusive).
every = 1  # Read ever n-th frame from the .edr file.


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_" + args.sol + "_[0-9]*-[0-9]*_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="Li_O_ratio"
)


print("Reading data...")
energies = np.full((len(observables), Sims.n_sims), np.nan, dtype=np.float64)
energies_sd = np.full_like(energies, np.nan)
for sim_ix, Sim in enumerate(Sims.sims):
    infile = Sim.settings + "_out_" + Sim.system + ".edr.gz"
    infile = os.path.join(Sim.path, infile)
    data, units = leap.io_handler.read_edr(
        infile, observables=observables, begin=begin, end=end, every=every
    )
    n_chains = Sims.res_nums["solvent"][sim_ix]
    for obs_ix, obs in enumerate(observables):
        energies[obs_ix, sim_ix] = np.mean(data[obs]) / n_chains
        energies_sd[obs_ix, sim_ix] = np.std(data[obs]) / n_chains
del data, units


print("Fitting straight line...")
if args.sol in ("g1", "peo63"):
    fit_energy_starts = (0,)
    fit_energy_stops = (
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 6))[0] + 1,
    )
elif args.sol == "g4":
    fit_energy_starts = (
        0,
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 5))[0],
    )
    fit_energy_stops = (
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 6))[0] + 1,
        len(Sims.Li_O_ratios),
    )
else:
    raise ValueError("Unknown --sol ({})".format(args.sol))
popt_energy = np.full(
    (len(observables), len(fit_energy_starts), 2), np.nan, dtype=np.float64
)
perr_energy = np.full_like(popt_energy, np.nan)
fit_quality = np.full_like(popt_energy, np.nan)
for obs_ix, energy in enumerate(energies):
    for fit_ix, start in enumerate(fit_energy_starts):
        (
            popt_energy[obs_ix, fit_ix],
            perr_energy[obs_ix, fit_ix],
            fit_quality[obs_ix, fit_ix],
        ) = fit_energy(
            xdata=Sims.Li_O_ratios,
            ydata=energy,
            ydata_sd=energies_sd[obs_ix],
            start=start,
            stop=fit_energy_stops[fit_ix],
        )


print("Creating plot(s)...")
xlabel = r"Li-to-EO Ratio $r$"
xlim_lin = (0, 0.4 + 0.0125)
xlim_log = (1e-2, 5e-1)
legend_title = r"$n_{EO} = %d$" % Sims.O_per_chain[0]
labels = (r"$E_{pot}$", r"$E_{kin}$", r"$E_{tot}$")
markers = ("o", "s", "^")
linestyles = ("dashed", "dotted")
if len(labels) != len(observables):
    raise ValueError(
        "`len(labels)` ({}) != `len(observables)`"
        " ({})".format(len(labels), len(observables))
    )
if len(markers) != len(observables):
    raise ValueError(
        "`len(markers)` ({}) != `len(observables)`"
        " ({})".format(len(markers), len(observables))
    )
if len(linestyles) < len(fit_energy_starts):
    raise ValueError(
        "`len(linestyles)` ({}) < `len(fit_energy_starts)`"
        " ({})".format(len(linestyles), len(fit_energy_starts))
    )

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot energies.
    fig, ax = plt.subplots(clear=True)
    for obs_ix, label in enumerate(labels):
        lines = ax.errorbar(
            Sims.Li_O_ratios,
            energies[obs_ix],
            yerr=energies_sd[obs_ix],
            label=label,
            marker=markers[obs_ix],
            linestyle="none",
        )
        # Fits.
        for fit_ix, popt in enumerate(popt_energy[obs_ix]):
            xdata = Sims.Li_O_ratios[
                fit_energy_starts[fit_ix] : fit_energy_stops[fit_ix]
            ]
            fit = leap.misc.straight_line(xdata, *popt)
            ax.plot(
                xdata,
                fit,
                color=lines[0].get_color(),
                linestyle=linestyles[fit_ix],
            )
            ax.text(
                xdata[-1],
                fit[-1],
                r"$%.0f r %+.0f$" % tuple(popt),
                rotation=np.rad2deg(np.arctan(popt[0])),
                rotation_mode="anchor",
                transform_rotates_text=True,
                horizontalalignment="right",
                verticalalignment="bottom",
                fontsize="x-small",
            )
    ax.set(xlabel=xlabel, ylabel=r"$E / N_{Chain}$ / kJ", xlim=xlim_lin)
    equalize_xticks(ax)
    legend = ax.legend(title=legend_title)
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    # Log scale x.
    ax.relim()
    ax.autoscale()
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_xlim(xlim_log)
    pdf.savefig()
    yd_min, yd_max = leap.plot.get_ydata_min_max(ax)
    if np.any(np.greater(yd_min, 0)):
        # Log scale y.
        ax.relim()
        ax.autoscale()
        ax.set_xscale("linear")
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set_xlim(xlim_lin)
        equalize_xticks(ax)
        pdf.savefig()
        # Log scale xy.
        ax.relim()
        ax.autoscale()
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set_xlim(xlim_log)
        pdf.savefig()
        plt.close()

    # Print fit parameters.
    fig, ax = plt.subplots(clear=True)
    ax.axis("off")
    fontsize = "xx-small"
    x_offset = 0.07
    y_offset = 0.03
    x_pos = x_offset
    y_pos = 1 - x_offset
    fig.text(
        x_pos, y_pos, "Fit Parameters, " + legend_title, fontsize=fontsize
    )
    for obs_ix, label in enumerate(labels):
        y_pos -= 2 * y_offset
        fig.text(x_pos, y_pos, label, fontsize=fontsize)
        for fit_ix, popt in enumerate(popt_energy[obs_ix]):
            text = r"Fit Region $%d$" % fit_ix
            fig.text(x_pos + x_offset, y_pos, text, fontsize=fontsize)
            for param_ix, param in enumerate(popt):
                if param_ix == 0:
                    text = "Slope:"
                elif param_ix == 1:
                    text = "Intercept:"
                else:
                    raise ValueError(
                        "Unknown parameter index: {}".format(param_ix)
                    )
                fig.text(x_pos + 4 * x_offset, y_pos, text, fontsize=fontsize)
                text = r"$%+16.9e \pm %16.9e$" % (
                    param,
                    perr_energy[obs_ix, fit_ix, param_ix],
                )
                fig.text(x_pos + 6 * x_offset, y_pos, text, fontsize=fontsize)
                y_pos -= y_offset
            for qual_ix, quality in enumerate(fit_quality[obs_ix, fit_ix]):
                if qual_ix == 0:
                    text = r"$R^2$"
                elif param_ix == 1:
                    text = "RMSE"
                else:
                    raise ValueError(
                        "Unknown quality index: {}".format(qual_ix)
                    )
                fig.text(x_pos + 4 * x_offset, y_pos, text, fontsize=fontsize)
                text = r"$%+16.9e$" % quality
                fig.text(x_pos + 6 * x_offset, y_pos, text, fontsize=fontsize)
                y_pos -= y_offset
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
