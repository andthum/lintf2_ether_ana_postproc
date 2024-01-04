#!/usr/bin/env python3


"""
Plot the average total volume of the system divided by the total number
of PEO chains as function of the PEO chain length.
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
from scipy.optimize import curve_fit

# First-party libraries
import lintf2_ether_ana_postproc as leap


def fit_volume(xdata, ydata, ydata_sd, start=0, stop=-1):
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
        "Plot the average total volume of the system divided by the total"
        " number of PEO chains as function of the PEO chain length."
    ),
)
args = parser.parse_args()

# Temperatures [K].
temps = (303, 423)
# Simulation settings.
settings_lst = ["eq_npt%d_pr_nh" % temp for temp in temps]
# Output filename.
outfile = "eq_npt_pr_nh_lintf2_peoN_20-1_sc80_volume_per_chain.pdf"


observables = ("Volume",)
begin = 50000  # First frame to read from the .edr file.
end = -1  # Last frame to read from the .edr file (exclusive).
every = 1  # Read ever n-th frame from the .edr file.


print("Creating Simulation instance(s)...")
Sims_lst = []
for settings in settings_lst:
    sys_pat = "lintf2_[gp]*[0-9]*_20-1_sc80"
    if settings == "eq_npt303_pr_nh":
        set_pat = "[0-9][0-9]_"
    elif settings == "eq_npt423_pr_nh":
        set_pat = "[0-9][0-9]_[3-9]_"
    set_pat += settings + "_" + sys_pat
    Sims = leap.simulation.get_sims(
        sys_pat, set_pat, path_key="bulk", sort_key="O_per_chain"
    )
    Sims_lst.append(Sims)


print("Reading data...")
vol_lst, vol_sd_lst = [], []
for Sims in Sims_lst:
    vol = np.full(Sims.n_sims, np.nan, dtype=np.float64)
    vol_sd = np.full_like(vol, np.nan)
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
del vol, vol_sd


print("Fitting straight line...")
fit_energy_starts = (0,)
fit_energy_stops = (len(Sims.O_per_chain),)
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
        ) = fit_volume(
            xdata=Sims.O_per_chain,
            ydata=energy,
            ydata_sd=energies_sd[obs_ix],
            start=start,
            stop=fit_energy_stops[fit_ix],
        )


print("Creating plot(s)...")
xlabel = r"Ether Oxygens per Chain $n_{EO}$"
xlim_lin = (0, 135)
xlim_log = (1, 200)
legend_title = r"$r = %.2f$" % Sims.Li_O_ratios[0]
labels = (r"$E_{pot}$", r"$E_{kin}$", r"$E_{tot}$")
markers = ("o", "s", "^")
linestyles = ("dashed",)
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
if len(linestyles) != len(fit_energy_starts):
    raise ValueError(
        "`len(linestyles)` ({}) != `len(fit_energy_starts)`"
        " ({})".format(len(linestyles), len(fit_energy_starts))
    )

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot energies.
    fig, ax = plt.subplots(clear=True)
    for obs_ix, label in enumerate(labels):
        lines = ax.errorbar(
            Sims.O_per_chain,
            energies[obs_ix],
            yerr=energies_sd[obs_ix],
            label=label,
            marker=markers[obs_ix],
            linestyle="none",
        )
        # Fits.
        for fit_ix, popt in enumerate(popt_energy[obs_ix]):
            xdata = Sims.O_per_chain[
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
                r"$%.0f n_{EO} %+.0f$" % tuple(popt),
                rotation=np.rad2deg(np.arctan(popt[0])),
                rotation_mode="anchor",
                transform_rotates_text=True,
                horizontalalignment="right",
                verticalalignment="bottom" if obs_ix == 0 else "top",
                fontsize="x-small",
            )
    ax.set(xlabel=xlabel, ylabel=r"$E / N_{Chain}$ / kJ", xlim=xlim_lin)
    legend = ax.legend(title=legend_title)
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    # Log scale x.
    ax.relim()
    ax.autoscale()
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_xlim(xlim_log)
    pdf.savefig()
    # Log scale y.
    yd_min, yd_max = leap.plot.get_ydata_min_max(ax)
    if np.any(np.greater(yd_min, 0)):
        ax.relim()
        ax.autoscale()
        ax.set_xscale("linear")
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set_xlim(xlim_lin)
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
    fontsize = "x-small"
    x_offset = 0.10
    y_offset = 0.03
    x_pos = 0.07
    y_pos = 0.97
    fig.text(
        x_pos, y_pos, "Fit Parameters, " + legend_title, fontsize=fontsize
    )
    for obs_ix, label in enumerate(labels):
        y_pos -= 2 * y_offset
        fig.text(x_pos, y_pos, label, fontsize=fontsize)
        for fit_ix, popt in enumerate(popt_energy[obs_ix]):
            if fit_ix > 0:
                y_pos -= y_offset
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
                y_pos -= y_offset
                fig.text(x_pos + x_offset, y_pos, text, fontsize=fontsize)
                text = r"$%+16.9e \pm %16.9e$" % (
                    param,
                    perr_energy[obs_ix, fit_ix, param_ix],
                )
                fig.text(x_pos + 3 * x_offset, y_pos, text, fontsize=fontsize)
            for qual_ix, quality in enumerate(fit_quality[obs_ix, fit_ix]):
                if qual_ix == 0:
                    text = r"$R^2$:"
                elif param_ix == 1:
                    text = "RMSE:"
                else:
                    raise ValueError(
                        "Unknown quality index: {}".format(qual_ix)
                    )
                y_pos -= y_offset
                fig.text(x_pos + x_offset, y_pos, text, fontsize=fontsize)
                text = r"$%+16.9e$" % quality
                fig.text(x_pos + 3 * x_offset, y_pos, text, fontsize=fontsize)
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
