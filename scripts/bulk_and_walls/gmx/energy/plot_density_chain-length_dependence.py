#!/usr/bin/env python3


"""
Plot the time-averaged mass density of the system as function of the PEO
chain length.
"""


# Standard libraries
import argparse

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
        "Plot the time-averaged density of the system as function of the PEO"
        " chain length."
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
outfile = "eq_npT_pr_nh_" + system + "_density.pdf"

# Columns to read from the input files.
cols = (
    0,  # Number of ether oxygens per PEO chain.
    15,  # Density [kg/nm^3].
    16,  # Standard deviation of the density [kg/nm^3].
)


# Densities from literature.

# Becht et al., Kolloid-Zeitschrift und Zeitschrift für Polymere, 1967,
# 216, 1, 150-158.
# Values taken from Figure 4.
# Neat, OH-terminated PEO.
n_eo = [2.00, 3.00, 5.24, 9.67, 13.53, 45.99, 145.87, 216.24, 369.46]
n_eo = np.asarray(n_eo)
Becht_1967_353K = [
    1.06673e00,
    1.06812e00,
    1.07015e00,
    1.07136e00,
    1.07227e00,
    1.07998e00,
    1.07363e00,
    1.07327e00,
    1.07292e00,
]
Becht_1967_353K = np.asarray(Becht_1967_353K)
Becht_1967_353K_xdata = np.asarray([n_eo, 1 / n_eo])
Becht_1967_353K_ydata = np.asarray([Becht_1967_353K, 1 / Becht_1967_353K])

# Zhang et al., J. Phys. Chem. B, 2014, 118, 19, 5144–5153.
# Values taken from Table S1 in SI.
# r = Li/EO ~ 1/20 = 0.05
# Actual r values:  0.0625,  0.0417,  0.0625,  0.0500
# 1/r = EO/Li:     16.00  , 18.00  , 16.00  , 20.00
n_eo = np.asarray([2, 3, 4, 5])
Zhang_2014_303K = np.asarray([1.0510, 1.1049, 1.1610, 1.1550])
Zhang_2014_303K_xdata = np.asarray([n_eo, 1 / n_eo])
Zhang_2014_303K_ydata = np.asarray([Zhang_2014_303K, 1 / Zhang_2014_303K])

# Villaluenga et al., J. Electrochem. Soc., 2018, 165, 11, A2766-A2773.
# Same as Pesko et al., J. Electrochem. Soc., 2017, 164, 11,
# E3569-E3575.
# Values taken from Table 1.
# Actual r values:  0.06
# 1/r = EO/Li:     16.68
n_eo = np.asarray([112.45])
Villaluenga_2018_363K = np.asarray([1.230])
Villaluenga_2018_363K_xdata = np.asarray([n_eo, 1 / n_eo])
Villaluenga_2018_363K_ydata = np.asarray(
    [Villaluenga_2018_363K, 1 / Villaluenga_2018_363K]
)

# Densities from experiments.
experiments_xdata = np.concatenate(
    [Zhang_2014_303K_xdata, Villaluenga_2018_363K_xdata], axis=-1
)
experiments_ydata = np.concatenate(
    [Zhang_2014_303K_ydata, Villaluenga_2018_363K_ydata], axis=-1
)

# Chattoraj et al., J. Phys. Chem. B, 2015, 119, 6786−6791.
# APPLE&P force field.
# Value taken from text (Section "MOLECULAR DYNAMICS SIMULATIONS").
n_eo = np.asarray([54])
Chattoraj_2015_423K = np.asarray([1.127])
Chattoraj_2015_423K_xdata = np.asarray([n_eo, 1 / n_eo])
Chattoraj_2015_423K_ydata = np.asarray(
    [Chattoraj_2015_423K, 1 / Chattoraj_2015_423K]
)

# Diddens et al., J. Electrochem. Soc., 2017, 164, 1, E3225-E3231.
# APPLE&P force field.
# Value taken from simulations of Diddo Diddens.
# See also Supplementary of Thum et al., J. Phys. Chem. C, 2021, 125,
# 25392−25403.
n_eo = np.asarray([64])
Diddens_2017_423K = np.asarray([1.15096])
Diddens_2017_423K_xdata = np.asarray([n_eo, 1 / n_eo])
Diddens_2017_423K_ydata = np.asarray(
    [Diddens_2017_423K, 1 / Diddens_2017_423K]
)

# Densities from MD simulations with the APPLE&P force field.
apple_p_xdata = np.concatenate(
    [Chattoraj_2015_423K_xdata, Diddens_2017_423K_xdata], axis=-1
)
apple_p_ydata = np.concatenate(
    [Chattoraj_2015_423K_ydata, Diddens_2017_423K_ydata], axis=-1
)


print("Reading data...")
infiles = [
    settings + "_" + system + "_energy.txt.gz" for settings in settings_lst
]
n_infiles = len(infiles)
xdata = [[None for infile in infiles] for i in range(2)]
ydata = [[None for infile in infiles] for i in range(2)]
ydata_sd = [[None for infile in infiles] for i in range(2)]
for set_ix, infile in enumerate(infiles):
    xdata[0][set_ix], ydata[0][set_ix], ydata_sd[0][set_ix] = np.loadtxt(
        infile, usecols=cols, unpack=True
    )
    ydata[0][set_ix] /= 1e3  # kg/m^3 -> g/cm^3.
    ydata_sd[0][set_ix] /= 1e3  # Std[c*A] = c * Std[A].

    # Calculate the specific volume (= inverse density).
    xdata[1][set_ix] = 1 / xdata[0][set_ix]
    ydata[1][set_ix] = 1 / ydata[0][set_ix]
    # Propagation of uncertainty.
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
    # Std[1/A] = 1/|A| * Std[A]/|A| = Std[A]/A^2
    ydata_sd[1][set_ix] = ydata_sd[0][set_ix] / ydata[0][set_ix] ** 2
    # Sort data in ascending x order.
    xdata[1][set_ix] = xdata[1][set_ix][::-1]
    ydata[1][set_ix] = ydata[1][set_ix][::-1]
    ydata_sd[1][set_ix] = ydata_sd[1][set_ix][::-1]

    if set_ix == 0:
        ydata_min = np.min(ydata[1][set_ix])
        ydata_max = np.max(ydata[1][set_ix])
    else:
        ydata_min = min(np.min(ydata[1][set_ix]), ydata_min)
        ydata_max = max(np.max(ydata[1][set_ix]), ydata_max)
ydata_max_diff = ydata_max - ydata_min


print("Fitting straight line...")
# Fit straight line to specific volume as function of inverse chain
# length.
fit_starts = (0,)
fit_stops = (len(xdata[1][0]),)
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
            xdata=xdata[1][set_ix],
            ydata=ydata[1][set_ix],
            ydata_sd=ydata_sd[1][set_ix],
            start=start,
            stop=fit_stops[fit_ix],
        )


print("Creating plot(s)...")
xlabels = (
    r"Ether Oxygens per Chain $n_{EO}$",
    r"Inverse Chain Length $1 / n_{EO}$",
)
ylabels = (r"Density / g cm$^{-3}$", r"Specific Volume / cm$^3$ g$^{-1}$")
linestyles_data = (None, "none")
xlims_lin = [(0, 135), (0.00, 0.52)]
xlims_log = [(1, 200), (6e-3, 6e-1)]
legend_title = r"$r = 0.05$"
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
    for data_ix, xlabel in enumerate(xlabels):
        fig, ax = plt.subplots(clear=True)
        for set_ix, temp in enumerate(temps):
            lines = ax.errorbar(
                xdata[data_ix][set_ix],
                ydata[data_ix][set_ix],
                yerr=ydata_sd[data_ix][set_ix],
                label=r"$%d$ K" % temp,
                marker=markers[set_ix],
                linestyle=linestyles_data[data_ix],
            )
            if data_ix == 1:  # Specific volume.
                # Fits.
                for fit_ix, popt in enumerate(popts[set_ix]):
                    x = xdata[data_ix][set_ix][
                        fit_starts[fit_ix] : fit_stops[fit_ix]
                    ]
                    fit = leap.misc.straight_line(x, *popt)
                    ax.plot(
                        x,
                        fit,
                        color=lines[0].get_color(),
                        linestyle=linestyles[fit_ix],
                    )
                    x_pos = x[-1] - 0.03 * xlims_lin[data_ix][-1]
                    y_pos = leap.misc.straight_line(x_pos, *popt)
                    y_offset = 0.02 * ydata_max_diff
                    ax.text(
                        x_pos,
                        y_pos + y_offset if set_ix == 0 else y_pos - y_offset,
                        r"$%.3f n_{EO} %+.3f$" % tuple(popt),
                        rotation=np.rad2deg(np.arctan(popt[0])),
                        rotation_mode="anchor",
                        transform_rotates_text=True,
                        horizontalalignment="right",
                        verticalalignment="bottom" if set_ix == 0 else "top",
                        linespacing=5,
                        fontsize="x-small",
                    )
        ax.errorbar(
            experiments_xdata[data_ix],
            experiments_ydata[data_ix],
            yerr=None,
            label=r"Exp. ($r \approx 0.05$)",
            marker="p",
            color="tab:cyan",
            linestyle="none",
        )
        ax.errorbar(
            apple_p_xdata[data_ix],
            apple_p_ydata[data_ix],
            yerr=None,
            label="APPLE&P",
            marker="D",
            color="tab:red",
            linestyle="none",
        )
        ax.set(xlabel=xlabel, ylabel=ylabels[data_ix], xlim=xlims_lin[data_ix])
        legend = ax.legend(
            title=legend_title,
            loc="lower right",
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        if data_ix == 1:  # Specific volume.
            # Cutout.
            ax.relim()
            ax.autoscale()
            ax.set(xlim=(0.004, 0.036), ylim=(0.790, 0.895))
            ax.xaxis.set_major_locator(MultipleLocator(0.01))
            legend = ax.legend(
                title=legend_title, loc="best", **mdtplt.LEGEND_KWARGS_XSMALL
            )
            legend.get_title().set_multialignment("center")
            pdf.savefig()
        # Log scale x.
        ax.relim()
        ax.autoscale()
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set_xlim(xlims_log[data_ix])
        pdf.savefig()
        # Log scale y.
        ax.relim()
        ax.autoscale()
        ax.set_xscale("linear")
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set_xlim(xlims_lin[data_ix])
        pdf.savefig()
        # Log scale xy.
        ax.relim()
        ax.autoscale()
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set_xlim(xlims_log[data_ix])
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
    text = "Fit Parameters, " + legend_title
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
