#!/usr/bin/env python3


"""
Plot the center-of-mass diffusion coefficients of Li, TFSI and PEO as
function of the PEO chain length.
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
        2.26951e02,
        9.07988e04,
    ],
    [  # D(Li).
        1.09583e-01,
        1.81301e-02,
        1.18834e-02,
        7.66857e-03,
        7.02093e-03,
        8.98093e-03,
        5.00057e-03,
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
        1.30165e-02,
        9.48820e-03,
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
        179.0,  # Cross-linked PEO.
        224.7,  # Cross-linked PEO.
    ],
    [  # D(PEO).
        3.56010e-01,
        2.32830e-01,
        1.64683e-01,
        4.85299e-02,
        9.16646e-03,
        1.91032e-03,
        5.07389e-04,  # Cross-linked PEO.
        9.02664e-04,  # Cross-linked PEO.
    ],
    [  # D(TFSI).
        2.92574e-01,
        2.13849e-01,
        1.63611e-01,
        6.86411e-02,
        2.69746e-02,
        1.61154e-02,
        1.19537e-02,  # Cross-linked PEO.
        1.31057e-02,  # Cross-linked PEO.
    ],
    [  # D(Li).
        2.22289e-01,
        1.63543e-01,
        1.17199e-01,
        4.25789e-02,
        1.03120e-02,
        3.89734e-03,
        2.47083e-03,  # Cross-linked PEO.
        3.80653e-03,  # Cross-linked PEO.
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
    [5.13100e-01, 3.14200e-01, 1.22000e-01, 1.07000e-01],  # D(TFSI).
    [5.11500e-01, 2.79200e-01, 1.05000e-01, 9.18000e-02],  # D(Li).
]


def fit_diff_coeff(diff_coeffs, diff_coeffs_sd, Sims, start=0, stop=-1):
    """
    Fit the logarithmic diffusion coefficient as function of the
    logarithmic chain length with a straight line.

    The obtained fit parameters are converted from the parameters of a
    straight line to the parameters of the corresponding power law.
    """
    # Propagation of uncertainty.  See
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
    # Std[ln(A)] = Std[A] / |A|
    sd = diff_coeffs_sd[start:stop] / np.abs(diff_coeffs[start:stop])
    popt, pcov = curve_fit(
        f=leap.misc.straight_line,
        xdata=np.log(Sims.O_per_chain[start:stop]),
        ydata=np.log(diff_coeffs[start:stop]),
        p0=(-2, np.log(diff_coeffs[0])),
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


def einstein_msd(times, diff_coeff, n_dim=3):
    r"""
    Calculate the Einstein MSD

    .. math::

        \langle r^2 \rangle = 2d \cdot D t

    Parameters
    ----------
    times : array_like
        Times :math:`t` at which to evaluate the Einstein MSD.
    diff_coeff : float
        The diffusion coefficient :math:`D`.
    n_dim : int, optional
        Number of dimensions :math:`d` in which the diffusive process
        takes place.

    Returns
    -------
    msd : numpy.ndarray
        Mean squared displacement.
    """
    times = np.asarray(times)
    return 2 * n_dim * diff_coeff * times


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
    settings + "_lintf2_peoN_20-1_sc80_" + analysis + "_diff_coeff_tot.pdf"
)


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="O_per_chain"
)


print("Reading data...")
compounds = ("ether", "NTf2", "Li")
diff_coeffs = [[] for cmp in compounds]
diff_coeffs_sd = [[] for cmp in compounds]
fit_starts = [[] for cmp in compounds]
fit_stops = [[] for cmp in compounds]
r2s = [[] for cmp in compounds]
rmses = [[] for cmp in compounds]
for cmp_ix, cmp in enumerate(compounds):
    file_suffix = analysis + "_" + cmp + "_tot_diff_coeff.txt.gz"
    infiles = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
    for infile in infiles:
        d_coeff, d_coeff_sd, fit_start, fit_stop, r2, rmse = np.loadtxt(
            infile, dtype=np.float32
        )
        diff_coeffs[cmp_ix].append(d_coeff)
        diff_coeffs_sd[cmp_ix].append(d_coeff_sd)
        fit_starts[cmp_ix].append(fit_start)
        fit_stops[cmp_ix].append(fit_stop)
        r2s[cmp_ix].append(r2)
        rmses[cmp_ix].append(rmse)


print("Fitting power law...")
# PEO
cmp_ix = compounds.index("ether")
fit_peo_starts = (0, np.flatnonzero(Sims.O_per_chain == 6)[0])
fit_peo_stops = (
    np.flatnonzero(Sims.O_per_chain == 6)[0],
    np.flatnonzero(Sims.O_per_chain == 54)[0],
)
popt_peo = np.full((len(fit_peo_starts), 2), np.nan, dtype=np.float64)
perr_peo = np.full_like(popt_peo, np.nan)
for fit_ix, start in enumerate(fit_peo_starts):
    popt_peo[fit_ix], perr_peo[fit_ix] = fit_diff_coeff(
        diff_coeffs=diff_coeffs[cmp_ix],
        diff_coeffs_sd=diff_coeffs_sd[cmp_ix],
        Sims=Sims,
        start=start,
        stop=fit_peo_stops[fit_ix],
    )

# TFSI
cmp_ix = compounds.index("NTf2")
fit_tfsi_starts = (
    0,
    # np.flatnonzero(Sims.O_per_chain == 6)[0],
)
fit_tfsi_stops = (
    np.flatnonzero(Sims.O_per_chain == 6)[0],
    # np.flatnonzero(Sims.O_per_chain == 54)[0],
)
popt_tfsi = np.full((len(fit_tfsi_starts), 2), np.nan, dtype=np.float64)
perr_tfsi = np.full_like(popt_tfsi, np.nan)
for fit_ix, start in enumerate(fit_tfsi_starts):
    popt_tfsi[fit_ix], perr_tfsi[fit_ix] = fit_diff_coeff(
        diff_coeffs=diff_coeffs[cmp_ix],
        diff_coeffs_sd=diff_coeffs_sd[cmp_ix],
        Sims=Sims,
        start=start,
        stop=fit_tfsi_stops[fit_ix],
    )

# Li
cmp_ix = compounds.index("Li")
fit_li_starts = (0, np.flatnonzero(Sims.O_per_chain == 6)[0])
fit_li_stops = (
    np.flatnonzero(Sims.O_per_chain == 6)[0],
    np.flatnonzero(Sims.O_per_chain == 54)[0],
)
popt_li = np.full((len(fit_li_starts), 2), np.nan, dtype=np.float64)
perr_li = np.full_like(popt_li, np.nan)
for fit_ix, start in enumerate(fit_li_starts):
    popt_li[fit_ix], perr_li[fit_ix] = fit_diff_coeff(
        diff_coeffs=diff_coeffs[cmp_ix],
        diff_coeffs_sd=diff_coeffs_sd[cmp_ix],
        Sims=Sims,
        start=start,
        stop=fit_li_stops[fit_ix],
    )


print("Creating plot(s)...")
xlabel = r"Ether Oxygens per Chain $n_{EO}$"
xlim = (1, 200)
labels = ("PEO", "TFSI", "Li")
colors = ("tab:blue", "tab:orange", "tab:green")
markers = ("o", "s", "^")

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Diffusion coefficients vs chain length.
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.errorbar(
            Sims.O_per_chain,
            diff_coeffs[cmp_ix],
            yerr=None,  # diff_coeffs_sd[cmp_ix], (SD < symbols).
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
            alpha=leap.plot.ALPHA,
        )
    # PEO fit.
    for fit_ix, popt in enumerate(popt_peo):
        xdata = Sims.O_per_chain[
            fit_peo_starts[fit_ix] : fit_peo_stops[fit_ix]
        ]
        fit = leap.misc.power_law(xdata, *popt)
        if fit_ix == 0:
            fit *= 2  # Create an offset to the real data.
            rotation = -36
            verticalalignment = "bottom"
        else:
            fit /= 2  # Create an offset to the real data.
            rotation = -37
            verticalalignment = "top"
        ax.plot(
            xdata, fit, color=colors[labels.index("PEO")], linestyle="dashed"
        )
        ax.text(
            xdata[0] if fit_ix != 1 else xdata[0] + 0.1,
            fit[0],
            # r"$D_{PEO} \propto n_{EO}^{%.2f \pm %.2f}$"
            # % (popt[0], perr_peo[fit_ix][0]),
            r"$D_{PEO} \propto n_{EO}^{%.2f}$" % popt[0],
            rotation=rotation,  # np.rad2deg(np.arctan(popt[0])) / 1.8,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment=verticalalignment,
            fontsize="small",
        )
    # TFSI Fit.
    for fit_ix, popt in enumerate(popt_tfsi):
        xdata = Sims.O_per_chain[
            fit_tfsi_starts[fit_ix] : fit_tfsi_stops[fit_ix]
        ]
        fit = leap.misc.power_law(xdata, *popt)
        if fit_ix == 0:
            fit /= 2  # Create an offset to the real data.
            rotation = -35
            verticalalignment = "top"
        else:
            fit *= 2  # Create an offset to the real data.
            rotation = -18
            verticalalignment = "bottom"
        ax.plot(
            xdata, fit, color=colors[labels.index("TFSI")], linestyle="dashed"
        )
        ax.text(
            xdata[0] if fit_ix != 0 else xdata[0] - 0.1,
            fit[0],
            # r"$D_{TFSI} \propto n_{EO}^{%.2f \pm %.2f}$"
            # % (popt[0], perr_tfsi[fit_ix][0]),
            r"$D_{TFSI} \propto n_{EO}^{%.2f}$" % popt[0],
            rotation=rotation,  # np.rad2deg(np.arctan(popt[0])) / 2,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment=verticalalignment,
            fontsize="small",
        )
    # Li Fit.
    for fit_ix, popt in enumerate(popt_li):
        xdata = Sims.O_per_chain[fit_li_starts[fit_ix] : fit_li_stops[fit_ix]]
        fit = leap.misc.power_law(xdata, *popt)
        fit /= 8  # Create an offset to the real data.
        rotation = -35
        ax.plot(
            xdata, fit, color=colors[labels.index("Li")], linestyle="dashed"
        )
        ax.text(
            xdata[0],
            fit[0],
            # r"$D_{Li} \propto n_{EO}^{%.2f \pm %.2f}$"
            # % (popt[0], perr_li[fit_ix][0]),
            r"$D_{Li} \propto n_{EO}^{%.2f}$" % popt[0],
            rotation=rotation,  # np.rad2deg(np.arctan(popt[0])) / 2,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment="top",
            fontsize="small",
        )
    del xdata, fit
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=r"Diff. Coeff. / nm$^2$ ns$^{-1}$", xlim=xlim)
    ax.legend(loc="upper right")
    pdf.savefig()
    plt.close()

    # Fit start vs chain length.
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            Sims.O_per_chain,
            fit_starts[cmp_ix],
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel="Fit Start / ns", xlim=xlim)
    ax.legend()
    pdf.savefig()
    # Log scale y.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Fit stop vs chain length.
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            Sims.O_per_chain,
            fit_stops[cmp_ix],
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel="Fit Stop / ns", xlim=xlim)
    ax.legend()
    pdf.savefig()
    # Log scale y.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Fit region (fit stop minus fit start) vs chain length.
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            Sims.O_per_chain,
            np.subtract(fit_stops[cmp_ix], fit_starts[cmp_ix]),
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel="Fit Stop - Start / ns", xlim=xlim)
    ax.legend()
    pdf.savefig()
    # Log scale y.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Coefficient of determination (R^2) vs chain length.
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            Sims.O_per_chain,
            r2s[cmp_ix],
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=r"Coeff. of Determ. $R^2$", xlim=xlim)
    ax.legend()
    pdf.savefig()
    # Log scale y.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()
    del r2s

    # RMSE vs chain length.
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            Sims.O_per_chain,
            rmses[cmp_ix],
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=r"RMSE / nm$^2$", xlim=xlim)
    ax.legend()
    pdf.savefig()
    # Log scale y.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()
    del rmses

    # Plot MSDs together with Einstein fit.
    legend_title = r"$r = %.2f$" % Sims.Li_O_ratios[0] + "\n" + r"$n_{EO}$"
    ls_fit = "dashed"
    cmap = plt.get_cmap()
    c_vals = np.arange(Sims.n_sims)
    c_norm = max(1, Sims.n_sims - 1)
    c_vals_normed = c_vals / c_norm
    colors = cmap(c_vals_normed)

    for cmp_ix, cmp in enumerate(compounds):
        fig_msd, ax_msd = plt.subplots(clear=True)
        ax_msd.set_prop_cycle(color=colors)
        fig_msd_t, ax_msd_t = plt.subplots(clear=True)
        ax_msd_t.set_prop_cycle(color=colors)
        fig_res, ax_res = plt.subplots(clear=True)
        ax_res.set_prop_cycle(color=colors)
        fig_res_t, ax_res_t = plt.subplots(clear=True)
        ax_res_t.set_prop_cycle(color=colors)
        fig_res_msd, ax_res_msd = plt.subplots(clear=True)
        ax_res_msd.set_prop_cycle(color=colors)

        for sim_ix, Sim in enumerate(Sims.sims):
            # Read MSD from file.
            file_suffix = analysis + "_" + cmp + ".txt.gz"
            infile = leap.simulation.get_ana_file(
                Sim, analysis, tool, file_suffix
            )
            times, msd = np.loadtxt(
                infile, usecols=(0, 1), unpack=True, dtype=np.float32
            )
            times *= 1e-3  # ps -> ns
            msd *= 1e-2  # A^2 -> nm^2

            # Calculate MSD from the diffusion coefficient.
            _, fit_start_ix = mdt.nph.find_nearest(
                times, fit_starts[cmp_ix][sim_ix], return_index=True
            )
            _, fit_stop_ix = mdt.nph.find_nearest(
                times, fit_stops[cmp_ix][sim_ix], return_index=True
            )
            times_fit = times[fit_start_ix:fit_stop_ix]
            msd_fit = einstein_msd(times_fit, diff_coeffs[cmp_ix][sim_ix])
            msd_fit_sd = einstein_msd(
                times_fit, diff_coeffs_sd[cmp_ix][sim_ix]
            )
            msd_fit_res = msd[fit_start_ix:fit_stop_ix] - msd_fit

            # Plot MSD vs time.
            lines = ax_msd.plot(
                times,
                msd,
                label=r"$%d$" % Sim.O_per_chain,
                linewidth=1,
                alpha=leap.plot.ALPHA,
            )
            ax_msd.fill_between(
                times_fit,
                y1=msd_fit + msd_fit_sd,
                y2=msd_fit - msd_fit_sd,
                color=lines[0].get_color(),
                edgecolor=None,
                alpha=leap.plot.ALPHA / 2,
                rasterized=True,
            )
            ax_msd.plot(
                times_fit,
                msd_fit,
                # label="Fit" if sim_ix == len(Sims.sims) - 1 else None,
                linestyle=ls_fit,
                color=lines[0].get_color(),
                alpha=leap.plot.ALPHA,
            )

            # Plot MSD/t vs time.
            lines = ax_msd_t.plot(
                times[1:],
                msd[1:] / times[1:],
                label=r"$%d$" % Sim.O_per_chain,
                linewidth=1,
                alpha=leap.plot.ALPHA,
            )
            ax_msd_t.fill_between(
                times_fit,
                y1=(msd_fit + msd_fit_sd) / times_fit,
                y2=(msd_fit - msd_fit_sd) / times_fit,
                color=lines[0].get_color(),
                edgecolor=None,
                alpha=leap.plot.ALPHA / 2,
                rasterized=True,
            )
            ax_msd_t.plot(
                times_fit,
                msd_fit / times_fit,
                # label="Fit" if sim_ix == len(Sims.sims) - 1 else None,
                linestyle=ls_fit,
                color=lines[0].get_color(),
                alpha=leap.plot.ALPHA,
            )

            # Plot fit residuals vs time.
            ax_res.plot(
                times_fit,
                msd_fit_res,
                label=r"$%d$" % Sim.O_per_chain,
                alpha=leap.plot.ALPHA,
            )
            # Plot fit residuals / t vs time.
            ax_res_t.plot(
                times_fit,
                msd_fit_res / times_fit,
                label=r"$%d$" % Sim.O_per_chain,
                alpha=leap.plot.ALPHA,
            )
            # Plot fit residuals / MSD vs time.
            ax_res_msd.plot(
                times_fit,
                msd_fit_res / msd[fit_start_ix:fit_stop_ix],
                label=r"$%d$" % Sim.O_per_chain,
                alpha=leap.plot.ALPHA,
            )

        # MSD vs time.
        ax_msd.set(
            xlabel="Diffusion Time / ns",
            ylabel=labels[cmp_ix] + r" MSD / nm$^2$",
            xlim=(times[0], times[-1]),
            ylim=(0, None),
        )
        legend = ax_msd.legend(
            title=legend_title,
            loc="upper left",
            ncol=3,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig_msd)
        # Log scale x.
        ax_msd.set_xlim(times[1], times[-1])
        ax_msd.set_xscale("log", base=10, subs=np.arange(2, 10))
        pdf.savefig(fig_msd)
        # Log scale y.
        ax_msd.relim()
        ax_msd.autoscale()
        ax_msd.set_xscale("linear")
        ax_msd.set_xlim(times[0], times[-1])
        ax_msd.set_yscale("log", base=10, subs=np.arange(2, 10))
        legend = ax_msd.legend(
            title=legend_title,
            loc="lower right",
            ncol=3,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig_msd)
        # Log scale xy.
        ax_msd.set_xlim(times[1], times[-1])
        ax_msd.set_xscale("log", base=10, subs=np.arange(2, 10))
        legend = ax_msd.legend(
            title=legend_title,
            loc="upper left",
            ncol=3,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig_msd)
        plt.close(fig_msd)

        # MSD/t vs time.
        ax_msd_t.set(
            xlabel=r"Diffusion Time $t$ / ns",
            ylabel=labels[cmp_ix] + r" MSD$(t)/t$ / nm$^2$ ns$^{-1}$",
            xlim=(times[1], times[-1]),
        )
        legend = ax_msd_t.legend(
            title=legend_title,
            loc="upper left",
            ncol=3,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig_msd_t)
        # Log scale x.
        ax_msd_t.set_xscale("log", base=10, subs=np.arange(2, 10))
        pdf.savefig(fig_msd_t)
        # Log scale y.
        ax_msd_t.set_xscale("linear")
        ax_msd_t.set_yscale("log", base=10, subs=np.arange(2, 10))
        pdf.savefig(fig_msd_t)
        # Log scale xy.
        ax_msd_t.set_xscale("log", base=10, subs=np.arange(2, 10))
        legend = ax_msd_t.legend(
            title=legend_title,
            loc="lower left",
            ncol=3,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig_msd_t)
        plt.close(fig_msd_t)

        # Fit residuals vs time.
        ax_res.set(
            xlabel="Diffusion Time / ns",
            ylabel=labels[cmp_ix] + r" Fit Res. / nm$^2$",
            xlim=(times[0], times[-1]),
        )
        legend = ax_res.legend(
            title=legend_title,
            loc="upper left",
            ncol=3,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig_res)
        # Log scale x.
        ax_res.set_xlim(times[1], times[-1])
        ax_res.set_xscale("log", base=10, subs=np.arange(2, 10))
        pdf.savefig(fig_res)
        plt.close(fig_res)

        # Fit residuals / t vs time.
        ax_res_t.set(
            xlabel=r"Diffusion Time $t$ / ns",
            ylabel=labels[cmp_ix] + r" Fit Res. / $t$ / nm$^2$ ns$^{-1}$",
            xlim=(times[0], times[-1]),
        )
        legend = ax_res_t.legend(
            title=legend_title,
            loc="upper left",
            ncol=3,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig_res_t)
        # Log scale x.
        ax_res_t.set_xlim(times[1], times[-1])
        ax_res_t.set_xscale("log", base=10, subs=np.arange(2, 10))
        pdf.savefig(fig_res_t)
        plt.close(fig_res_t)

        # Fit residuals / MSD vs time.
        ax_res_msd.set(
            xlabel="Diffusion Time / ns",
            ylabel=labels[cmp_ix] + r" Fit Res. / MSD",
            xlim=(times[0], times[-1]),
        )
        legend = ax_res_msd.legend(
            title=legend_title,
            loc="upper left",
            ncol=3,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig(fig_res_msd)
        # Log scale x.
        ax_res_msd.set_xlim(times[1], times[-1])
        ax_res_msd.set_xscale("log", base=10, subs=np.arange(2, 10))
        pdf.savefig(fig_res_msd)
        plt.close(fig_res_msd)


print("Created {}".format(outfile))
print("Done")
