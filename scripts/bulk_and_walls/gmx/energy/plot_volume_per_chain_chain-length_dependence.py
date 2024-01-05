#!/usr/bin/env python3


"""
Plot the time-averaged total volume of the system divided by the total
number of PEO chains as function of the PEO chain length.
"""


# Standard libraries
import argparse

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

# First-party libraries
import lintf2_ether_ana_postproc as leap


def fit_line(xdata, ydata, ydata_sd, start=0, stop=-1):
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
        "Plot the time-averaged total volume of the system divided by the"
        " total number of PEO chains as function of the PEO chain length."
    ),
)
args = parser.parse_args()

# Temperatures [K].
temps = (303, 423)
# Simulation settings.
settings_lst = ["eq_npt%d_pr_nh" % temp for temp in temps]
# System name.
system = "lintf2_peoN_20-1_sc80"
# Output filename.
outfile = "eq_npT_pr_nh_" + system + "_volume_per_chain.pdf"

# Columns to read from the input files.
cols = (
    0,  # Number of ether oxygens per PEO chain.
    1,  # Number of PEO chains.
    13,  # Volume [nm^3].
    14,  # Standard deviation of the volume [nm^3].
)


print("Reading data...")
infiles = [
    settings + "_" + system + "_energy.txt.gz" for settings in settings_lst
]
n_infiles = len(infiles)
xdata = [None for infile in infiles]
ydata = [None for infile in infiles]
ydata_sd = [None for infiles in infiles]
for set_ix, infile in enumerate(infiles):
    xdata[set_ix], n_chains, ydata[set_ix], ydata_sd[set_ix] = np.loadtxt(
        infile, usecols=cols, unpack=True
    )
    # Divide volume by number of PEO chain.
    ydata[set_ix] /= n_chains
    ydata_sd[set_ix] /= n_chains
    if set_ix == 0:
        ydata_min = np.min(ydata[set_ix])
        ydata_max = np.max(ydata[set_ix])
    else:
        ydata_min = min(np.min(ydata[set_ix]), ydata_min)
        ydata_max = max(np.max(ydata[set_ix]), ydata_max)
ydata_max_diff = ydata_max - ydata_min


print("Fitting straight line...")
fit_starts = (0,)
fit_stops = (len(xdata[0]),)
popts = np.full((n_infiles, len(fit_starts), 2), np.nan, dtype=np.float64)
perrs = np.full_like(popts, np.nan)
fit_quality = np.full_like(popts, np.nan)
for set_ix in range(n_infiles):
    for fit_ix, start in enumerate(fit_starts):
        (
            popts[set_ix, fit_ix],
            perrs[set_ix, fit_ix],
            fit_quality[set_ix, fit_ix],
        ) = fit_line(
            xdata=xdata[set_ix],
            ydata=ydata[set_ix],
            ydata_sd=ydata_sd[set_ix],
            start=start,
            stop=fit_stops[fit_ix],
        )


print("Creating plot(s)...")
xlabel = r"Ether Oxygens per Chain $n_{EO}$"
xlim_lin = (0, 135)
xlim_log = (1, 200)
legend_title = r"$r = 0.05$" + "\n" + "T / K"
markers = ("o", "s")
if len(markers) != n_infiles:
    raise ValueError(
        "`len(markers)` ({}) != `n_infiles`"
        " ({})".format(len(markers), n_infiles)
    )
linestyles = ("dashed",)
if len(linestyles) != len(fit_starts):
    raise ValueError(
        "`len(linestyles)` ({}) != `len(fit_starts)`"
        " ({})".format(len(linestyles), len(fit_starts))
    )

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    fig, ax = plt.subplots(clear=True)
    for set_ix, temp in enumerate(temps):
        lines = ax.errorbar(
            xdata[set_ix],
            ydata[set_ix],
            yerr=ydata_sd[set_ix],
            label=r"$%d$" % temp,
            marker=markers[set_ix],
            linestyle="none",
        )
        # Fits.
        for fit_ix, popt in enumerate(popts[set_ix]):
            x = xdata[set_ix][fit_starts[fit_ix] : fit_stops[fit_ix]]
            fit = leap.misc.straight_line(x, *popt)
            ax.plot(
                x,
                fit,
                color=lines[0].get_color(),
                linestyle=linestyles[fit_ix],
            )
            x_pos = x[-1] - 0.03 * xlim_lin[-1]
            y_pos = leap.misc.straight_line(x_pos, *popt)
            y_offset = 0.02 * ydata_max_diff
            ax.text(
                x_pos,
                y_pos - y_offset if set_ix == 0 else y_pos + y_offset,
                r"$%.3f n_{EO} %+.3f$" % tuple(popt),
                rotation=np.rad2deg(np.arctan(popt[0])),
                rotation_mode="anchor",
                transform_rotates_text=True,
                horizontalalignment="right",
                verticalalignment="top" if set_ix == 0 else "bottom",
                linespacing=5,
                fontsize="x-small",
            )
    ax.set(xlabel=xlabel, ylabel=r"$V / N_{Chain}$ / nm$^3$", xlim=xlim_lin)
    legend = ax.legend(title=legend_title, loc="lower right")
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    # Cutout.
    ax.relim()
    ax.autoscale()
    ax.set(xlim=(1.5, 8.5), ylim=(0.15, 0.85))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(MaxNLocator(integer=True))
    legend = ax.legend(title=legend_title, loc="best")
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    # Log scale x.
    ax.relim()
    ax.autoscale()
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_xlim(xlim_log)
    pdf.savefig()
    # Log scale y.
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
    fontsize = "xx-small"
    x_offset = 0.10
    y_offset = 0.03
    x_pos = 0.07
    y_pos = 0.97
    text = "Fit Parameters, " + legend_title.split("\n")[0]
    fig.text(x_pos, y_pos, text, fontsize=fontsize)
    for set_ix, temp in enumerate(temps):
        y_pos -= 2 * y_offset
        text = "$%d$ K" % temp
        fig.text(x_pos, y_pos, text, fontsize=fontsize)
        for fit_ix, popt in enumerate(popts[set_ix]):
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
                    perrs[set_ix, fit_ix, param_ix],
                )
                fig.text(x_pos + 3 * x_offset, y_pos, text, fontsize=fontsize)
            for qual_ix, quality in enumerate(fit_quality[set_ix, fit_ix]):
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
