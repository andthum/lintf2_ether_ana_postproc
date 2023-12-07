#!/usr/bin/env python3


"""
Plot the diffusion coefficients of Li, TFSI and PEO obtained from
experiments as function of the salt concentration.
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


def fit_diff_coeff(n_eo, diff_coeffs, start=0, stop=-1):
    """
    Fit the logarithmic diffusion coefficient as function of the linear
    salt concentration with a straight line.

    The obtained fit parameters are converted from the parameters of a
    straight line to the parameters of the corresponding exponential
    law.
    """
    popt, pcov = curve_fit(
        f=leap.misc.straight_line,
        xdata=n_eo[start:stop],
        ydata=np.log(diff_coeffs[start:stop]),
        p0=(-2, np.log(diff_coeffs[0])),
    )
    perr = np.sqrt(np.diag(pcov))
    # Convert straight-line parameters to the corresponding
    # exponential-law parameters.
    popt = np.array([popt[0], np.exp(popt[1])])
    # Propagation of uncertainty.
    # Std[exp(A)] = |exp(A)| * Std[A]
    perr = np.array([perr[0], np.abs(popt[1]) * perr[1]])
    return popt, perr


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the center-of-mass diffusion coefficients of Li, TFSI and PEO as"
        " function of the salt concentration."
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
analysis = "msd"  # Analysis name.
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    settings
    + "_lintf2_"
    + args.sol
    + "_r_sc80_"
    + analysis
    + "_diff_coeff_tot_exp.pdf"
)


# Diffusion coefficients from literature.

# Zhang et al., J. Phys. Chem. B, 2014, 118, 19, 5144–5153.
# Self-diffusion coefficients from PFG-NMR.
# Values taken from Table S1 in SI.
# n_EO = 2 (G1):
Li_O_ratios = 1 / np.array([32.00, 16.00, 8.00, 6.00, 5.00, 4.00, 3.60])
Zhang_2014_G1 = [  # D in nm^2/ns at 303 K.
    [  # D(G1).
        1.92200e00,
        1.13800e00,
        3.70600e-01,
        1.31200e-01,
        7.21000e-02,
        4.08000e-02,
        2.84000e-02,
    ],
    [  # D(TFSI).
        8.10700e-01,
        5.11500e-01,
        1.90400e-01,
        7.58000e-02,
        4.58000e-02,
        2.81000e-02,
        1.96000e-02,
    ],
    [  # D(Li).
        8.97300e-01,
        5.13100e-01,
        1.86300e-01,
        7.98000e-02,
        4.94000e-02,
        3.25000e-02,
        2.34000e-02,
    ],
]
Zhang_2014_G1 = np.asarray(Zhang_2014_G1)
Zhang_2014_G1 = np.row_stack([Li_O_ratios, Zhang_2014_G1])
del Li_O_ratios
# n_EO = 5 (G4):
Li_O_ratios = 1 / np.array([40.00, 20.00, 10.00, 5.00, 4.00, 3.75])
Zhang_2014_G4 = [  # D in nm^2/ns at 303 K.
    [  # D(G4).
        2.06000e-01,
        1.38000e-01,
        4.21000e-02,
        1.26000e-02,
        4.10000e-03,
        1.70000e-03,
    ],
    [  # D(TFSI).
        1.49000e-01,
        1.07000e-01,
        3.98000e-02,
        1.22000e-02,
        3.80000e-03,
        1.50000e-03,
    ],
    [  # D(Li).
        1.22000e-01,
        9.18000e-02,
        3.32000e-02,
        1.26000e-02,
        3.80000e-03,
        1.80000e-03,
    ],
]
Zhang_2014_G4 = np.asarray(Zhang_2014_G4)
Zhang_2014_G4 = np.row_stack([Li_O_ratios, Zhang_2014_G4])
del Li_O_ratios

# Yoshida et al., J. Phys. Chem. C, 2011, 115, 37, 18384-18394.
# Self-diffusion coefficients from PFG-NMR.
# Values taken from Figure 3 and Table 1.
# n_EO = 5 (G4):
Li_O_ratios = 1 / np.array([150, 40, 20, 10, 5])
Yoshida_2011_G4 = [  # D in nm^2/ns at 303 K.
    [  # D(G4).
        3.08824e-01,
        2.09857e-01,
        1.38315e-01,
        4.19714e-02,
        1.24006e-02,
    ],
    [  # D(TFSI).
        1.98172e-01,
        1.45469e-01,
        1.06836e-01,
        4.03021e-02,
        1.24006e-02,
    ],
    [  # D(Li).
        1.74324e-01,
        1.20668e-01,
        9.22893e-02,
        3.31479e-02,
        1.24006e-02,
    ],
]
Yoshida_2011_G4 = np.asarray(Yoshida_2011_G4)
Yoshida_2011_G4 = np.row_stack([Li_O_ratios, Yoshida_2011_G4])
del Li_O_ratios

# Orädd et al., Solid State Ionics, 2002, 152-153, 131-136.
# Self-diffusion coefficients from PFG-NMR.
# Values taken from Figure 4.
# n_EO = 113498.57 (molecular weight = 5e6 g/mol).
Li_O_ratios = 1 / np.array([50, 30, 20, 16, 12, 8, 5])
Oraedd_2002 = [  # D in nm^2/ns at 358 K.
    [  # D(TFSI).
        3.55743e-02,
        2.98148e-02,
        2.43424e-02,
        2.23579e-02,
        1.63331e-02,
        9.42827e-03,
        4.30054e-03,
    ],
    [  # D(Li).
        7.21032e-03,
        6.04296e-03,
        4.59126e-03,
        4.24464e-03,
        2.75639e-03,
        1.29908e-03,
        8.05842e-04,
    ],
]
Oraedd_2002 = np.asarray(Oraedd_2002)
Oraedd_2002 = np.row_stack([Li_O_ratios, Oraedd_2002])
del Li_O_ratios

# Timachova et al., Macromolecules, 2015, 48, 7882-7888.
# Self-diffusion coefficients from PFG-NMR.
# Values taken from Figure 1.
# n_EO = 91 (molecular weight = 4000 g/mol).
Timachova_2015 = [  # D in nm^2/ns at 363 K.
    [  # r = Li/EO.
        0.0100,
        0.0200,
        0.0400,
        0.0600,
        0.0800,
    ],
    [  # D(TFSI).
        6.90424e-02,
        6.41182e-02,
        5.26498e-02,
        4.70472e-02,
        4.10536e-02,
    ],
    [  # D(Li).
        3.00480e-02,
        2.60735e-02,
        1.57783e-02,
        1.49802e-02,
        1.42938e-02,
    ],
]
Timachova_2015 = np.asarray(Timachova_2015)

# Pesko et al., J. Electrochem. Soc., 2017, 164, 11, E3569-E3575.
# Self-diffusion coefficients from PFG-NMR.
# Values taken from Figure 1.
# n_EO = 112.45 (molecular weight = 5000 g/mol).
Pesko_2017 = [  # D in nm^2/ns at 363 K.
    [  # r = Li/EO.
        0.0100,
        0.0200,
        0.0400,
        0.0600,
        0.0800,
        0.1000,
        0.1200,
        0.1400,
    ],
    [  # D(TFSI).
        6.56089e-02,
        6.16827e-02,
        5.10406e-02,
        3.47159e-02,
        3.08413e-02,
        2.15424e-02,
        1.73579e-02,
        1.41550e-02,
    ],
    [  # D(Li).
        2.13875e-02,
        1.72546e-02,
        1.45683e-02,
        9.86716e-03,
        7.59410e-03,
        4.54613e-03,
        3.77122e-03,
        2.63469e-03,
    ],
]
Pesko_2017 = np.asarray(Pesko_2017)


print("Fitting exponential law...")
# PEO
fit_peo_start_g1, fit_peo_stop_g1 = 0, 3
popt_peo_g1, perr_peo_g1 = fit_diff_coeff(
    n_eo=Zhang_2014_G1[0],
    diff_coeffs=Zhang_2014_G1[1],
    start=fit_peo_start_g1,
    stop=fit_peo_stop_g1,
)
fit_peo_start_g4, fit_peo_stop_g4 = 0, 4
popt_peo_g4, perr_peo_g4 = fit_diff_coeff(
    n_eo=Zhang_2014_G4[0],
    diff_coeffs=Zhang_2014_G4[1],
    start=fit_peo_start_g4,
    stop=fit_peo_stop_g4,
)

# TFSI
fit_tfsi_start_g1, fit_tfsi_stop_g1 = 0, 3
popt_tfsi_g1, perr_tfsi_g1 = fit_diff_coeff(
    n_eo=Zhang_2014_G1[0],
    diff_coeffs=Zhang_2014_G1[2],
    start=fit_tfsi_start_g1,
    stop=fit_tfsi_stop_g1,
)
fit_tfsi_start_g4, fit_tfsi_stop_g4 = 0, 4
popt_tfsi_g4, perr_tfsi_g4 = fit_diff_coeff(
    n_eo=Zhang_2014_G4[0],
    diff_coeffs=Zhang_2014_G4[2],
    start=fit_tfsi_start_g4,
    stop=fit_tfsi_stop_g4,
)
fit_tfsi_start_peo, fit_tfsi_stop_peo = 0, len(Pesko_2017[1])
popt_tfsi_peo, perr_tfsi_peo = fit_diff_coeff(
    n_eo=Pesko_2017[0],
    diff_coeffs=Pesko_2017[1],
    start=fit_tfsi_start_peo,
    stop=fit_tfsi_stop_peo,
)

# Li
fit_li_start_g1, fit_li_stop_g1 = 0, 3
popt_li_g1, perr_li_g1 = fit_diff_coeff(
    n_eo=Zhang_2014_G1[0],
    diff_coeffs=Zhang_2014_G1[3],
    start=fit_li_start_g1,
    stop=fit_li_stop_g1,
)
fit_li_start_g4, fit_li_stop_g4 = 0, 4
popt_li_g4, perr_li_g4 = fit_diff_coeff(
    n_eo=Zhang_2014_G4[0],
    diff_coeffs=Zhang_2014_G4[3],
    start=fit_li_start_g4,
    stop=fit_li_stop_g4,
)
fit_li_start_peo, fit_li_stop_peo = 0, len(Pesko_2017[2])
popt_li_peo, perr_li_peo = fit_diff_coeff(
    n_eo=Pesko_2017[0],
    diff_coeffs=Pesko_2017[2],
    start=fit_li_start_peo,
    stop=fit_li_stop_peo,
)


print("Creating plot(s)...")
xlabel = r"Li-to-EO Ratio $r$"
ylabel = r"Diff. Coeff. / nm$^2$ ns$^{-1}$"
xlim = (0, 0.4 + 0.0125)
labels = ("PEO", "TFSI", "Li")
colors = ("tab:blue", "tab:orange", "tab:green")
markers = ("o", "s", "^")
fillstyles = ("left", "right", "bottom", "top", "full")
linestyles = ("dashed", "dashdot", "dotted")

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Diffusion coefficients vs salt concentration.
    fig, ax = plt.subplots(clear=True)
    if args.sol == "g1":
        ylim = (1e-2, 4e0)
        # for cmp_ix, label in enumerate(labels):
        #     ax.plot(
        #         [],
        #         [],
        #         label=label,
        #         color=colors[cmp_ix],
        #         marker=markers[cmp_ix],
        #         alpha=leap.plot.ALPHA,
        #     )
        for cmp_ix, marker in enumerate(markers):
            ax.plot(
                Zhang_2014_G1[0],  # r = Li/EO.
                Zhang_2014_G1[cmp_ix + 1],  # D(PEO,TFSI,Li).
                # label="Zhang" if cmp_ix == 2 else None,
                label=labels[cmp_ix],
                color=colors[cmp_ix],
                marker=marker,
                alpha=leap.plot.ALPHA,
            )
        # PEO fit.
        xdata = Zhang_2014_G1[0][fit_peo_start_g1:fit_peo_stop_g1]
        fit = leap.misc.exp_law(xdata, *popt_peo_g1)
        fit *= 1.5  # Create an offset to the real data.
        ax.plot(xdata, fit, color=colors[0], linestyle="dashed")
        ax.text(
            xdata[0],
            fit[0],
            r"$D_{PEO} \propto \exp(%.2f r)$" % popt_peo_g1[0],
            rotation=-40,  # np.rad2deg(np.arctan(popt_peo_g1[0])) / 1.8,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize="small",
        )
        # TFSI fit.
        xdata = Zhang_2014_G1[0][fit_tfsi_start_g1:fit_tfsi_stop_g1]
        fit = leap.misc.exp_law(xdata, *popt_tfsi_g1)
        fit /= 1.5  # Create an offset to the real data.
        ax.plot(xdata, fit, color=colors[1], linestyle="dashed")
        ax.text(
            xdata[0],
            fit[0],
            r"$D_{TFSI} \propto \exp(%.2f r)$" % popt_tfsi_g1[0],
            rotation=-38,  # np.rad2deg(np.arctan(popt_tfsi_g1[0])) / 1.8,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment="top",
            fontsize="small",
        )
        # Li fit.
        xdata = Zhang_2014_G1[0][fit_li_start_g1:fit_li_stop_g1]
        fit = leap.misc.exp_law(xdata, *popt_li_g1)
        fit /= 3  # Create an offset to the real data.
        ax.plot(xdata, fit, color=colors[2], linestyle="dashed")
        ax.text(
            xdata[0],
            fit[0],
            r"$D_{Li} \propto \exp(%.2f r)$" % popt_li_g1[0],
            rotation=-40,  # np.rad2deg(np.arctan(popt_li_g1[0])) / 1.8,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment="top",
            fontsize="small",
        )
        legend = ax.legend(
            title=r"$n_{EO} = 2$" + "\nZhang $303$ K", loc="upper right"
        )
    elif args.sol == "g4":
        ylim = (7e-4, 4e-1)
        brightnesses = np.linspace(0, 2, 2 + 2)[1:-1]
        # for cmp_ix, label in enumerate(labels):
        #     ax.plot(
        #         [],
        #         [],
        #         label=label,
        #         color=colors[cmp_ix],
        #         marker=markers[cmp_ix],
        #         alpha=leap.plot.ALPHA,
        #     )
        for cmp_ix, marker in enumerate(markers):
            ax.plot(
                Zhang_2014_G4[0],  # r = Li/EO.
                Zhang_2014_G4[cmp_ix + 1],  # D(PEO,TFSI,Li).
                label=labels[cmp_ix],
                # label="Zhang" if cmp_ix == 2 else None,
                # linestyle="dashed",
                # fillstyle="left",
                # color=leap.plot.change_brightness(
                #     colors[cmp_ix], brightnesses[0]
                # ),
                color=colors[cmp_ix],
                marker=marker,
                alpha=leap.plot.ALPHA,
            )
        # for cmp_ix, marker in enumerate(markers):
        #     ax.plot(
        #         Yoshida_2011_G4[0],  # r = Li/EO.
        #         Yoshida_2011_G4[cmp_ix + 1],  # D(PEO,TFSI,Li).
        #         label="Yoshida" if cmp_ix == 2 else None,
        #         linestyle="dotted",
        #         fillstyle="right",
        #         color=leap.plot.change_brightness(
        #             colors[cmp_ix], brightnesses[1]
        #         ),
        #         marker=marker,
        #         alpha=leap.plot.ALPHA,
        #     )
        # legend = ax.legend(
        #     title=r"$n_{EO} = 5$" + "\n$303$ K", loc="upper right"
        # )
        # PEO fit.
        xdata = Zhang_2014_G4[0][fit_peo_start_g4:fit_peo_stop_g4]
        fit = leap.misc.exp_law(xdata, *popt_peo_g4)
        fit *= 1.5  # Create an offset to the real data.
        ax.plot(xdata, fit, color=colors[0], linestyle="dashed")
        ax.text(
            xdata[0],
            fit[0],
            r"$D_{PEO} \propto \exp(%.2f r)$" % popt_peo_g4[0],
            rotation=-36,  # np.rad2deg(np.arctan(popt_peo_g4[0])) / 1.8,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize="small",
        )
        # TFSI fit.
        xdata = Zhang_2014_G4[0][fit_tfsi_start_g4:fit_tfsi_stop_g4]
        fit = leap.misc.exp_law(xdata, *popt_tfsi_g4)
        fit /= 2  # Create an offset to the real data.
        ax.plot(xdata, fit, color=colors[1], linestyle="dashed")
        ax.text(
            xdata[0],
            fit[0],
            r"$D_{TFSI} \propto \exp(%.2f r)$" % popt_tfsi_g4[0],
            rotation=-34,  # np.rad2deg(np.arctan(popt_tfsi_g4[0])) / 1.8,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment="top",
            fontsize="small",
        )
        # Li fit.
        xdata = Zhang_2014_G4[0][fit_li_start_g4:fit_li_stop_g4]
        fit = leap.misc.exp_law(xdata, *popt_li_g4)
        fit /= 4  # Create an offset to the real data.
        ax.plot(xdata, fit, color=colors[2], linestyle="dashed")
        ax.text(
            xdata[0],
            fit[0],
            r"$D_{Li} \propto \exp(%.2f r)$" % popt_li_g4[0],
            rotation=-32,  # np.rad2deg(np.arctan(popt_li_g4[0])) / 1.8,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment="top",
            fontsize="small",
        )
        legend = ax.legend(
            title=r"$n_{EO} = 5$" + "\nZhang $303$ K", loc="upper right"
        )
    elif args.sol == "peo63":
        ylim = (3e-4, 2e-1)
        brightnesses = np.linspace(0, 2, 3 + 2)[1:-1]
        for cmp_ix, label in enumerate(labels[1:], start=1):
            ax.plot(
                [],
                [],
                label=label,
                color=colors[cmp_ix],
                marker=markers[cmp_ix],
                alpha=leap.plot.ALPHA,
            )
        for cmp_ix, marker in enumerate(markers[1:], start=1):
            ax.plot(
                Timachova_2015[0],  # r = Li/EO.
                Timachova_2015[cmp_ix],  # D(TFSI,Li).
                label=(
                    "Timachova $363$ K\n" + r"($n_{EO} \approx 91$)"
                    if cmp_ix == 2
                    else None
                ),
                linestyle="solid",
                fillstyle="none",
                color=leap.plot.change_brightness(
                    colors[cmp_ix], brightnesses[0]
                ),
                marker=marker,
                alpha=leap.plot.ALPHA,
            )
        for cmp_ix, marker in enumerate(markers[1:], start=1):
            ax.plot(
                Pesko_2017[0],  # r = Li/EO.
                Pesko_2017[cmp_ix],  # D(TFSI,Li).
                label=(
                    "Pesko $363$ K\n" + r"($n_{EO} \approx 112$)"
                    if cmp_ix == 2
                    else None
                ),
                linestyle="dashed",
                fillstyle="left",
                color=leap.plot.change_brightness(
                    colors[cmp_ix], brightnesses[1]
                ),
                marker=marker,
                alpha=leap.plot.ALPHA,
            )
        for cmp_ix, marker in enumerate(markers[1:], start=1):
            ax.plot(
                Oraedd_2002[0],  # r = Li/EO.
                Oraedd_2002[cmp_ix],  # D(TFSI,Li).
                label=(
                    "Orädd $358$ K\n" + r"($n_{EO} \approx 113500$)"
                    if cmp_ix == 2
                    else None
                ),
                linestyle="dotted",
                fillstyle="right",
                color=leap.plot.change_brightness(
                    colors[cmp_ix], brightnesses[2]
                ),
                marker=marker,
                alpha=leap.plot.ALPHA,
            )
        # TFSI fit.
        xdata = Pesko_2017[0][fit_tfsi_start_peo:fit_tfsi_stop_peo]
        fit = leap.misc.exp_law(xdata, *popt_tfsi_peo)
        fit *= 2  # Create an offset to the real data.
        ax.plot(xdata, fit, color=colors[1], linestyle="dashed")
        ax.text(
            xdata[0],
            fit[0],
            r"$D_{TFSI} \propto \exp(%.2f r)$" % popt_tfsi_peo[0],
            rotation=-29,  # np.rad2deg(np.arctan(popt_tfsi_peo[0])) / 1.8,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize="small",
        )
        # Li fit.
        xdata = Pesko_2017[0][fit_li_start_peo:fit_li_stop_peo]
        fit = leap.misc.exp_law(xdata, *popt_li_peo)
        fit /= 4  # Create an offset to the real data.
        ax.plot(xdata, fit, color=colors[2], linestyle="dashed")
        ax.text(
            xdata[0],
            fit[0],
            r"$D_{Li} \propto \exp(%.2f r)$" % popt_li_peo[0],
            rotation=-36,  # np.rad2deg(np.arctan(popt_li_peo[0])) / 1.8,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment="top",
            fontsize="small",
        )
        legend = ax.legend(
            loc="upper right",
            title=r"$n_{EO} > 64$",
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
    else:
        raise ValueError("Unknown --sol ({})".format(args.sol))
    legend.get_title().set_multialignment("center")
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
