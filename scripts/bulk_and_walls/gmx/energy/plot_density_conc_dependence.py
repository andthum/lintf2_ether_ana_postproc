#!/usr/bin/env python3


"""
Plot the time-averaged mass density of the system as function of the
salt concentration.
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


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the time-averaged density of the system as function of the salt"
        " concentration."
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

# Temperatures [K].
temps = (303, 423)
# Simulation settings.
settings_lst = ["eq_npt%d_pr_nh" % temp for temp in temps]
# System name.
system = "lintf2_" + args.sol + "_r_sc80"
# Output filename.
outfile = "eq_npT_pr_nh_" + system + "_density.pdf"

# Columns to read from the input files.
cols = (
    0,  # Lithium-to-ether-oxygen ratio.
    15,  # Density [kg/nm^3].
    16,  # Standard deviation of the density [kg/nm^3].
)


# Densities from literature.

# Zhang et al., J. Phys. Chem. B, 2014, 118, 19, 5144–5153.
# Values taken from Table S1 in SI.
# T = 303K.
# n_EO = 2 (G1):
O_Li_ratios = [60.00, 32.00, 16.00, 8.00, 6.00, 5.00, 4.00, 3.60, 3.00]
O_Li_ratios = np.asarray(O_Li_ratios)
Zhang_2014_G1 = [
    0.9165,
    0.9630,
    1.0510,
    1.1873,
    1.2561,
    1.2998,
    1.3523,
    1.3773,
    1.4193,
]
Zhang_2014_G1 = np.asarray(Zhang_2014_G1)
Zhang_2014_G1_xdata = np.asarray([1 / O_Li_ratios, O_Li_ratios])
Zhang_2014_G1_ydata = np.asarray([Zhang_2014_G1, 1 / Zhang_2014_G1])
# n_EO = 5 (G4):
O_Li_ratios = [150.00, 40.00, 20.00, 10.00, 7.50, 5.00, 4.00, 3.75]
O_Li_ratios = np.asarray(O_Li_ratios)
Zhang_2014_G4 = [
    1.0260,
    # 1.4080, 1.0408?, 1/r = 75.
    1.0780,
    1.1550,
    1.2640,
    1.3171,
    1.4000,
    1.4471,
    1.4534,
]
Zhang_2014_G4 = np.asarray(Zhang_2014_G4)
Zhang_2014_G4_xdata = np.asarray([1 / O_Li_ratios, O_Li_ratios])
Zhang_2014_G4_ydata = np.asarray([Zhang_2014_G4, 1 / Zhang_2014_G4])

# Yoshida et al., J. Phys. Chem. C, 2011, 115, 37, 18384-18394.
# Values taken from Table 1.
# T = 303K.
# n_EO = 5 (G4):
O_Li_ratios = np.asarray([2000.00, 450.00, 150.00, 40.00, 20.00, 10.00, 5.00])
Yoshida_2011_G4 = np.asarray([1.00, 1.01, 1.03, 1.08, 1.16, 1.26, 1.40])
Yoshida_2011_G4_xdata = np.asarray([1 / O_Li_ratios, O_Li_ratios])
Yoshida_2011_G4_ydata = np.asarray([Yoshida_2011_G4, 1 / Yoshida_2011_G4])

# Villaluenga et al., J. Electrochem. Soc., 2018, 165, 11, A2766-A2773.
# Same as Pesko et al., J. Electrochem. Soc., 2017, 164, 11,
# E3569-E3575.
# In Pesko et al., the relative error of the measured densities is
# estimated to be 2%.
# Values taken from Table 1.
# T = 363 K.
# n_EO = 112.45 (molecular weight = 5000 g/mol).
Li_O_ratios = [
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.14,
    0.16,
    0.18,
    0.21,
    0.24,
    0.27,
    0.30,
]
Li_O_ratios = np.asarray(Li_O_ratios)
Villaluenga_2018 = [
    1.160,
    1.180,
    1.210,
    1.230,
    1.330,
    1.365,
    1.380,
    1.430,
    1.450,
    1.470,
    1.516,
    1.580,
    1.572,
    1.640,
]
Villaluenga_2018 = np.asarray(Villaluenga_2018)
Villaluenga_2018_xdata = np.asarray([Li_O_ratios, 1 / Li_O_ratios])
Villaluenga_2018_ydata = np.asarray([Villaluenga_2018, 1 / Villaluenga_2018])


# Densities from MD simulations with the APPLE&P force field.
# Chattoraj et al., J. Phys. Chem. B, 2015, 119, 6786−6791.
# Diddens et al., J. Electrochem. Soc., 2017, 164, 1, E3225-E3231.
Li_O_ratios = np.asarray([1 / 20, 1 / 20])
apple_p = np.asarray([1.127, 1.15096])
apple_p_xdata = np.asarray([Li_O_ratios, 1 / Li_O_ratios])
apple_p_ydata = np.asarray([apple_p, 1 / apple_p])


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


print("Creating plot(s)...")
xlabels = (r"Li-to-EO Ratio $r$", r"EO-to-Li Ratio $1/r$")
ylabels = (r"Density / g cm$^{-3}$", r"Specific Volume / cm$^3$ g$^{-1}$")
xlims_lin = [(0, 0.4 + 0.0125), (0, 85)]
xlims_log = [(1e-2, 5e-1), (2e0, 9e1)]
if args.sol == "g1":
    legend_title = r"$n_{EO} = 2$"
elif args.sol == "g4":
    legend_title = r"$n_{EO} = 5$"
elif args.sol == "peo63":
    legend_title = r"$n_{EO} = 64$"
else:
    raise ValueError("Unknown --sol: {}".format(args.sol))
markers = ("o", "s")
if len(markers) != n_infiles:
    raise ValueError(
        "`len(markers)` ({}) != `n_infiles`"
        " ({})".format(len(markers), n_infiles)
    )

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for data_ix, xlabel in enumerate(xlabels):
        fig, ax = plt.subplots(clear=True)
        for set_ix, temp in enumerate(temps):
            ax.errorbar(
                xdata[data_ix][set_ix],
                ydata[data_ix][set_ix],
                yerr=ydata_sd[data_ix][set_ix],
                label=r"$%d$ K" % temp,
                marker=markers[set_ix],
            )
        if args.sol == "g1":
            ax.errorbar(
                Zhang_2014_G1_xdata[data_ix],
                Zhang_2014_G1_ydata[data_ix],
                yerr=None,
                label=r"Zhang $303$ K",
                marker="p",
                color="tab:cyan",
            )
        elif args.sol == "g4":
            ax.errorbar(
                Zhang_2014_G4_xdata[data_ix],
                Zhang_2014_G4_ydata[data_ix],
                yerr=None,
                label=r"Zhang $303$ K",
                marker="p",
                color="tab:cyan",
            )
        elif args.sol == "peo63":
            ax.errorbar(
                Villaluenga_2018_xdata[data_ix],
                Villaluenga_2018_ydata[data_ix],
                yerr=None,  # Villaluenga_2018_ydata[data_ix] * 0.02,
                label=r"Villaluenga $363$ K",
                marker="^",
            )
            ax.errorbar(
                apple_p_xdata[data_ix],
                apple_p_ydata[data_ix],
                yerr=None,
                label=r"APPLE&P $423$ K",
                marker="D",
                color="tab:red",
                linestyle="none",
            )
        else:
            raise ValueError("Unknown --sol: {}".format(args.sol))
        ax.set(xlabel=xlabel, ylabel=ylabels[data_ix], xlim=xlims_lin[data_ix])
        if data_ix == 0:  # Density.
            equalize_xticks(ax)
        legend = ax.legend(title=legend_title, **mdtplt.LEGEND_KWARGS_XSMALL)
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
        if data_ix == 0:  # Density.
            equalize_xticks(ax)
        pdf.savefig()
        # Log scale xy.
        ax.relim()
        ax.autoscale()
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set_xlim(xlims_log[data_ix])
        pdf.savefig()
        plt.close()

print("Created {}".format(outfile))
print("Done")
