#!/usr/bin/env python3


"""
Plot the diffusion coefficients of Li, TFSI and PEO obtained from
experiments as function of the PEO chain length.
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
    Fit the logarithmic diffusion coefficient as function of the
    logarithmic chain length with a straight line.

    The obtained fit parameters are converted from the parameters of a
    straight line to the parameters of the corresponding power law.
    """
    popt, pcov = curve_fit(
        f=leap.misc.straight_line,
        xdata=np.log(n_eo[start:stop]),
        ydata=np.log(diff_coeffs[start:stop]),
        p0=(-2, np.log(diff_coeffs[0])),
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
        "Plot the center-of-mass diffusion coefficients of Li, TFSI and PEO as"
        " function of the PEO chain length."
    ),
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "msd"  # Analysis name.
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    settings + "_lintf2_peoN_20-1_sc80_" + analysis + "_diff_coeff_tot_exp.pdf"
)


# Diffusion coefficients from literature.

# Shi and Vincent, Solid State Ionics, 1993, 60, 1, 11-17, Figure 6.
# (Or Vincent, Electrochimica Acta, 1995, 40, 13, 2035-2040, Figure 4.)
# Self-diffusion coefficients from PFG-NMR.
# Values taken from Figure 6.
# Anion = CF3SO3 (Triflate), not TFSI.
Shi_1993_343K = [  # D in nm^2/ns at 343 K.
    [  # n_EO calculated from the molecular weight given in the text.
        9.03413e00,
        2.26540e01,
        4.53537e01,
        7.59983e01,
        1.36153e02,
        2.26951e02,  # n_EO > 200.
        9.07988e04,  # n_EO >> 200.
    ],
    [  # D(Li).
        1.09583e-01,
        1.81301e-02,
        1.18834e-02,
        7.66857e-03,
        7.02093e-03,
        8.98093e-03,  # n_EO > 200.
        5.00057e-03,  # n_EO >> 200.
    ],
]
Shi_1993_343K = np.asarray(Shi_1993_343K)
Shi_1993_363K = [  # D in nm^2/ns at 363 K.
    Shi_1993_343K[0],  # n_EO.
    [  # D(Li).
        1.88515e-01,
        3.04457e-02,
        2.18169e-02,
        1.27715e-02,
        1.09929e-02,
        1.30165e-02,  # n_EO > 200.
        9.48820e-03,  # n_EO >> 200.
    ],
]
Shi_1993_363K = np.asarray(Shi_1993_363K)

# Hayamizu et al., J. Chem. Phys., 2002, 117, 5929.
# Self-diffusion coefficients from PFG-NMR.
# Values taken from Figures 8 and 9 and Table 1.
Hayamizu_2002_303K = [  # D in nm^2/ns at 303 K (Figure 8).
    [6.0, 11.6],  # n_EO from Table 1.
    [6.97241e-02, 1.62601e-02],  # D(PEO).
    [7.08797e-02, 2.44353e-02],  # D(TFSI).
    [5.60011e-02, 1.44467e-02],  # D(Li).
]
Hayamizu_2002_303K = np.asarray(Hayamizu_2002_303K)
Hayamizu_2002_333K = [  # D in nm^2/ns at 333 K (Figure 8).
    [  # n_EO from Table 1 and Figure 9.
        4.0,
        5.0,
        6.0,
        11.6,
        23.9,
        56.3,
        # 179.0,  # Cross-linked PEO.
        # 224.7,  # Cross-linked PEO.
    ],
    [  # D(PEO).
        3.56010e-01,
        2.32830e-01,
        1.64683e-01,
        4.85299e-02,
        9.16646e-03,
        1.91032e-03,
        # 5.07389e-04,  # Cross-linked PEO.
        # 9.02664e-04,  # Cross-linked PEO.
    ],
    [  # D(TFSI).
        2.92574e-01,
        2.13849e-01,
        1.63611e-01,
        6.86411e-02,
        2.69746e-02,
        1.61154e-02,
        # 1.19537e-02,  # Cross-linked PEO.
        # 1.31057e-02,  # Cross-linked PEO.
    ],
    [  # D(Li).
        2.22289e-01,
        1.63543e-01,
        1.17199e-01,
        4.25789e-02,
        1.03120e-02,
        3.89734e-03,
        # 2.47083e-03,  # Cross-linked PEO.
        # 3.80653e-03,  # Cross-linked PEO.
    ],
]
Hayamizu_2002_333K = np.asarray(Hayamizu_2002_333K)
Hayamizu_2002_343K = [  # D in nm^2/ns at 343 K (Figure 8).
    [6.0, 11.6, 56.3],  # n_EO from Table 1.
    [2.03542e-01, 6.46224e-02, 2.80010e-03],  # D(PEO).
    [2.06350e-01, 9.12374e-02, 2.32371e-02],  # D(TFSI).
    [1.37189e-01, 6.33618e-02, 5.79263e-03],  # D(Li).
]
Hayamizu_2002_343K = np.asarray(Hayamizu_2002_343K)
Hayamizu_2002_363K = [  # D in nm^2/ns at 363 K (Figure 8).
    [11.6, 56.3],  # n_EO from Table 1.
    [1.05838e-01, 5.58900e-03],  # D(PEO).
    [1.50907e-01, 4.31457e-02],  # D(TFSI).
    [1.08657e-01, 1.12581e-02],  # D(Li).
]
Hayamizu_2002_363K = np.asarray(Hayamizu_2002_363K)

# Zhang et al., J. Phys. Chem. B, 2014, 118, 19, 5144–5153.
# Self-diffusion coefficients from PFG-NMR.
# Values taken from Table S1 in SI.
# r = Li/EO ~ 1/20 = 0.05
Zhang_2014_303K = [
    # Actual r values:  0.0625,  0.0417,  0.0625,  0.0500
    # 1/r = EO/Li:     16.00  , 24.00  , 16.00  , 20.00
    [2, 3, 4, 5],  # n_EO.
    [1.13800e00, 5.84100e-01, 1.73000e-01, 1.38000e-01],  # D(PEO).
    [5.11500e-01, 3.14200e-01, 1.22000e-01, 1.07000e-01],  # D(TFSI).
    [5.13100e-01, 2.79200e-01, 1.05000e-01, 9.18000e-02],  # D(Li).
]

# Number of different temperatures.
n_temps = 4


print("Fitting power law for Hayamizu_2002_333K...")
# PEO
fit_peo_start, fit_peo_stop = 0, len(Hayamizu_2002_333K[1])
popt_peo, perr_peo = fit_diff_coeff(
    n_eo=Hayamizu_2002_333K[0],
    diff_coeffs=Hayamizu_2002_333K[1],
    start=fit_peo_start,
    stop=fit_peo_stop,
)

# # TFSI
# fit_tfsi_start, fit_tfsi_stop = 0, -1
# popt_tfsi, perr_tfsi = fit_diff_coeff(
#     n_eo=Hayamizu_2002_333K[0],
#     diff_coeffs=Hayamizu_2002_333K[2],
#     start=fit_tfsi_start,
#     stop=fit_tfsi_stop,
# )

# Li
fit_li_start, fit_li_stop = 0, -1
popt_li, perr_li = fit_diff_coeff(
    n_eo=Hayamizu_2002_333K[0],
    diff_coeffs=Hayamizu_2002_333K[3],
    start=fit_li_start,
    stop=fit_li_stop,
)


print("Creating plot(s)...")
xlabel = r"Ether Oxygens per Chain $n_{EO}$"
ylabel = r"Diff. Coeff. / nm$^2$ ns$^{-1}$"
xlim = (1, 200)
xlim_Shi = (1e0, 2e5)  # noqa: N816
ylim = (1e-3, 2e1)
labels = ("PEO", "TFSI", "Li")
colors = ("tab:blue", "tab:orange", "tab:green")
markers = ("o", "s", "^")
fillstyles = ("left", "right", "bottom", "top", "full")
linestyles = ("dashed", "dashdot", "dotted")

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Diffusion coefficients vs chain length (303 K).
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            [],
            [],
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
            alpha=leap.plot.ALPHA,
        )
    for cmp_ix, marker in enumerate(markers):
        ax.plot(
            Zhang_2014_303K[0],  # n_EO.
            Zhang_2014_303K[cmp_ix + 1],  # D(PEO,TFSI,Li).
            label=r"Zhang ($r \approx 0.05$)" if cmp_ix == 2 else None,
            linestyle=linestyles[0],
            fillstyle=fillstyles[0],
            color=colors[cmp_ix],
            marker=marker,
            alpha=leap.plot.ALPHA,
        )
    for cmp_ix, marker in enumerate(markers):
        ax.plot(
            Hayamizu_2002_303K[0],  # n_EO.
            Hayamizu_2002_303K[cmp_ix + 1],  # D(PEO,TFSI,Li).
            label="Hayamizu" if cmp_ix == 2 else None,
            linestyle=linestyles[1],
            fillstyle=fillstyles[1],
            color=colors[cmp_ix],
            marker=marker,
            alpha=leap.plot.ALPHA,
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    ax.legend(
        loc="upper right", title="$303$ K", **mdtplt.LEGEND_KWARGS_XSMALL
    )
    pdf.savefig()
    plt.close()

    # Diffusion coefficients vs chain length (333 K).
    fig, ax = plt.subplots(clear=True)
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
            Hayamizu_2002_333K[0],  # n_EO.
            Hayamizu_2002_333K[cmp_ix + 1],  # D(PEO,TFSI,Li).
            # label="Hayamizu" if cmp_ix == 2 else None,
            label=labels[cmp_ix],
            linestyle=linestyles[1],
            fillstyle=fillstyles[1],
            color=colors[cmp_ix],
            marker=marker,
            alpha=leap.plot.ALPHA,
        )
    # PEO fit.
    xdata = Hayamizu_2002_333K[0][fit_peo_start:fit_peo_stop]
    fit = leap.misc.power_law(xdata, *popt_peo)
    fit /= 2.5  # Create an offset to the real data.
    ax.plot(xdata, fit, color=colors[0], linestyle="dashed")
    ax.text(
        xdata[0],
        fit[0],
        r"$D_{PEO} \propto n_{EO}^{%.2f}$" % popt_peo[0],
        rotation=-38,  # np.rad2deg(np.arctan(popt_peo[0])) / 1.8,
        rotation_mode="anchor",
        transform_rotates_text=False,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize="small",
    )
    # # TFSI fit.
    # xdata = Hayamizu_2002_333K[0][fit_tfsi_start:fit_tfsi_stop]
    # fit = leap.misc.power_law(xdata, *popt_tfsi)
    # fit *= 2  # Create an offset to the real data.
    # ax.plot(xdata, fit, color=colors[1], linestyle="dashed")
    # ax.text(
    #     xdata[0],
    #     fit[0],
    #     r"$D_{TFSI} \propto n_{EO}^{%.2f}$" % popt_tfsi[0],
    #     rotation=-25,  # np.rad2deg(np.arctan(popt_tfsi[0])) / 1.8,
    #     rotation_mode="anchor",
    #     transform_rotates_text=False,
    #     horizontalalignment="left",
    #     verticalalignment="bottom",
    #     fontsize="small",
    # )
    # Li fit.
    xdata = Hayamizu_2002_333K[0][fit_li_start:fit_li_stop]
    fit = leap.misc.power_law(xdata, *popt_li)
    fit /= 7  # Create an offset to the real data.
    ax.plot(xdata, fit, color=colors[2], linestyle="dashed")
    ax.text(
        xdata[0],
        fit[0],
        r"$D_{Li} \propto n_{EO}^{%.2f}$" % popt_li[0],
        rotation=-35,  # np.rad2deg(np.arctan(popt_li[0])) / 1.8,
        rotation_mode="anchor",
        transform_rotates_text=False,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize="small",
    )
    # Format axes.
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    legend = ax.legend(
        loc="upper right",
        title="Hayamizu $333$ K",
        ncol=1,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Diffusion coefficients vs chain length (343 K).
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            [],
            [],
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
            alpha=leap.plot.ALPHA,
        )
    for cmp_ix, marker in enumerate(markers):
        ax.plot(
            Hayamizu_2002_343K[0],  # n_EO.
            Hayamizu_2002_343K[cmp_ix + 1],  # D(PEO,TFSI,Li).
            label="Hayamizu" if cmp_ix == 2 else None,
            linestyle=linestyles[1],
            fillstyle=fillstyles[1],
            color=colors[cmp_ix],
            marker=marker,
            alpha=leap.plot.ALPHA,
        )
    ax.plot(
        Shi_1993_343K[0],  # n_EO.
        Shi_1993_343K[1],  # D(Li).
        label="Shi",
        linestyle=linestyles[2],
        fillstyle=fillstyles[2],
        color=colors[2],
        marker=markers[2],
        alpha=leap.plot.ALPHA,
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    ax.legend(
        loc="upper right", title="$343$ K", **mdtplt.LEGEND_KWARGS_XSMALL
    )
    pdf.savefig()
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim_Shi, ylim=ylim)
    pdf.savefig()
    plt.close()

    # Diffusion coefficients vs chain length (363 K).
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            [],
            [],
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
            alpha=leap.plot.ALPHA,
        )
    for cmp_ix, marker in enumerate(markers):
        ax.plot(
            Hayamizu_2002_363K[0],  # n_EO.
            Hayamizu_2002_363K[cmp_ix + 1],  # D(PEO,TFSI,Li).
            label="Hayamizu" if cmp_ix == 2 else None,
            linestyle=linestyles[1],
            fillstyle=fillstyles[1],
            color=colors[cmp_ix],
            marker=marker,
            alpha=leap.plot.ALPHA,
        )
    ax.plot(
        Shi_1993_363K[0],  # n_EO.
        Shi_1993_363K[1],  # D(Li).
        label="Shi",
        color=colors[2],
        linestyle=linestyles[2],
        fillstyle=fillstyles[2],
        marker=markers[2],
        alpha=leap.plot.ALPHA,
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    ax.legend(
        loc="upper right", title="$363$ K", **mdtplt.LEGEND_KWARGS_XSMALL
    )
    pdf.savefig()
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim_Shi, ylim=ylim)
    pdf.savefig()
    plt.close()

    # Diffusion coefficients vs chain length (all temperatures).
    brightnesses = np.linspace(0, 2, 3 + 2)[1:-1]
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            [],
            [],
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
            alpha=leap.plot.ALPHA,
        )
    for cmp_ix, marker in enumerate(markers):
        ax.plot(
            Zhang_2014_303K[0],  # n_EO.
            Zhang_2014_303K[cmp_ix + 1],  # D(PEO,TFSI,Li).
            label=r"Zhang $303$ K ($r \approx 0.05$)" if cmp_ix == 2 else None,
            linestyle="solid",
            # fillstyle=fillstyles[0],
            color=leap.plot.change_brightness(colors[cmp_ix], brightnesses[0]),
            marker=marker,
            alpha=leap.plot.ALPHA,
        )
    # for cmp_ix, marker in enumerate(markers):
    #     ax.plot(
    #         Hayamizu_2002_303K[0],  # n_EO.
    #         Hayamizu_2002_303K[cmp_ix + 1],  # D(PEO,TFSI,Li).
    #         label="Hayamizu $303$ K" if cmp_ix == 2 else None,
    #         linestyle=linestyles[1],
    #         fillstyle=fillstyles[1],
    #         color=leap.plot.change_brightness(
    #             colors[cmp_ix], brightnesses[0]
    #         ),
    #         marker=marker,
    #         alpha=leap.plot.ALPHA,
    #     )
    for cmp_ix, marker in enumerate(markers):
        ax.plot(
            Hayamizu_2002_333K[0],  # n_EO.
            Hayamizu_2002_333K[cmp_ix + 1],  # D(PEO,TFSI,Li).
            label="Hayamizu $333$ K" if cmp_ix == 2 else None,
            linestyle="dashed",
            # fillstyle=fillstyles[1],
            color=leap.plot.change_brightness(colors[cmp_ix], brightnesses[1]),
            marker=marker,
            alpha=leap.plot.ALPHA,
        )
    # PEO fit.
    xdata = Hayamizu_2002_333K[0][fit_peo_start:fit_peo_stop]
    fit = leap.misc.power_law(xdata, *popt_peo)
    fit /= 4.5  # Create an offset to the real data.
    ax.plot(
        xdata,
        fit,
        color=leap.plot.change_brightness(colors[0], brightnesses[1]),
        linestyle="dashed",
        alpha=leap.plot.ALPHA,
    )
    ax.text(
        xdata[0],
        fit[0],
        r"$D_{PEO} \propto n_{EO}^{%.2f}$" % popt_peo[0],
        rotation=-38,  # np.rad2deg(np.arctan(popt_peo[0])) / 1.8,
        rotation_mode="anchor",
        transform_rotates_text=False,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize="small",
    )
    # # TFSI fit.
    # xdata = Hayamizu_2002_333K[0][fit_tfsi_start:fit_tfsi_stop]
    # fit = leap.misc.power_law(xdata, *popt_tfsi)
    # fit *= 3  # Create an offset to the real data.
    # ax.plot(
    #     xdata,
    #     fit,
    #     color=leap.plot.change_brightness(colors[1], brightnesses[1]),
    #     linestyle="dashed",
    #     alpha=leap.plot.ALPHA,
    # )
    # ax.text(
    #     xdata[0],
    #     fit[0],
    #     r"$D_{TFSI} \propto n_{EO}^{%.2f}$" % popt_tfsi[0],
    #     rotation=-25,  # np.rad2deg(np.arctan(popt_tfsi[0])) / 1.8,
    #     rotation_mode="anchor",
    #     transform_rotates_text=False,
    #     horizontalalignment="left",
    #     verticalalignment="bottom",
    #     fontsize="small",
    # )
    # Li fit.
    xdata = Hayamizu_2002_333K[0][fit_li_start:fit_li_stop]
    fit = leap.misc.power_law(xdata, *popt_li)
    fit /= 10  # Create an offset to the real data.
    ax.plot(
        xdata,
        fit,
        color=leap.plot.change_brightness(colors[2], brightnesses[1]),
        linestyle="dashed",
        alpha=leap.plot.ALPHA,
    )
    ax.text(
        xdata[0],
        fit[0],
        r"$D_{Li} \propto n_{EO}^{%.2f}$" % popt_li[0],
        rotation=-35,  # np.rad2deg(np.arctan(popt_li[0])) / 1.8,
        rotation_mode="anchor",
        transform_rotates_text=False,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize="small",
    )
    # for cmp_ix, marker in enumerate(markers):
    #     ax.plot(
    #         Hayamizu_2002_343K[0],  # n_EO.
    #         Hayamizu_2002_343K[cmp_ix + 1],  # D(PEO,TFSI,Li).
    #         label="Hayamizu $343$ K" if cmp_ix == 2 else None,
    #         linestyle=linestyles[1],
    #         fillstyle=fillstyles[1],
    #         color=leap.plot.change_brightness(
    #             colors[cmp_ix], brightnesses[2]
    #         ),
    #         marker=marker,
    #         alpha=leap.plot.ALPHA,
    #     )
    # ax.plot(
    #     Shi_1993_343K[0],  # n_EO.
    #     Shi_1993_343K[1],  # D(Li).
    #     label="Shi $343$ K",
    #     linestyle=linestyles[2],
    #     fillstyle=fillstyles[2],
    #     color=leap.plot.change_brightness(colors[2], brightnesses[2]),
    #     marker=markers[2],
    #     alpha=leap.plot.ALPHA,
    # )
    # for cmp_ix, marker in enumerate(markers):
    #     ax.plot(
    #         Hayamizu_2002_363K[0],  # n_EO.
    #         Hayamizu_2002_363K[cmp_ix + 1],  # D(PEO,TFSI,Li).
    #         label="Hayamizu $363$ K" if cmp_ix == 2 else None,
    #         linestyle=linestyles[1],
    #         fillstyle=fillstyles[1],
    #         color=leap.plot.change_brightness(
    #             colors[cmp_ix], brightnesses[-1]
    #         ),
    #         marker=marker,
    #         alpha=leap.plot.ALPHA,
    #     )
    ax.plot(
        Shi_1993_363K[0],  # n_EO.
        Shi_1993_363K[1],  # D(Li).
        label="Shi $363$ K",
        linestyle=linestyles[2],
        # fillstyle=fillstyles[2],
        color=leap.plot.change_brightness(colors[2], brightnesses[-1]),
        marker=markers[2],
        alpha=leap.plot.ALPHA,
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    ax.legend(loc="upper right", **mdtplt.LEGEND_KWARGS_XSMALL)
    pdf.savefig()
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim_Shi, ylim=ylim)
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
