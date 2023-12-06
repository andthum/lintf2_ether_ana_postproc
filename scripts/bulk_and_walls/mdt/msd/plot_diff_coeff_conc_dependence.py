#!/usr/bin/env python3


"""
Plot the center-of-mass diffusion coefficients of Li, TFSI and PEO as
function of the salt concentration.
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

# Zhang et al., J. Phys. Chem. B, 2014, 118, 19, 5144–5153.
# Self-diffusion coefficients from PFG-NMR.
# Values taken from Table S1 in SI.
# n_EO = 2 (G1):
Li_O_ratios_G1 = 1 / np.array([32.00, 16.00, 8.00, 6.00, 5.00, 4.00, 3.60])
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
        8.97300e-01,
        5.13100e-01,
        1.86300e-01,
        7.98000e-02,
        4.94000e-02,
        3.25000e-02,
        2.34000e-02,
    ],
    [  # D(Li).
        8.10700e-01,
        5.11500e-01,
        1.90400e-01,
        7.58000e-02,
        4.58000e-02,
        2.81000e-02,
        1.96000e-02,
    ],
]
Zhang_2014_G1 = np.asarray(Zhang_2014_G1)
Zhang_2014_G1 = np.row_stack([Li_O_ratios_G1, Zhang_2014_G1])
del Li_O_ratios_G1
# n_EO = 5 (G4):
Li_O_ratios_G4 = 1 / np.array([40.00, 20.00, 10.00, 5.00, 4.00, 3.75])
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
Zhang_2014_G4 = np.row_stack([Li_O_ratios_G4, Zhang_2014_G4])
del Li_O_ratios_G4

# Yoshida et al., J. Phys. Chem. C, 2011, 115, 37, 18384-18394.
# Self-diffusion coefficients from PFG-NMR.
# Values taken from Figure 3 and Table 1.
# n_EO = 5 (G4):
Li_O_ratios_G4 = 1 / np.array([150, 40, 20, 10, 5])
Yoshida_2011_G4 = [  # D in nm^2/ns at 303 K.
    [
        3.08824e-01,
        2.09857e-01,
        1.38315e-01,
        4.19714e-02,
        1.24006e-02,
    ],  # D(G4).
    [
        1.98172e-01,
        1.45469e-01,
        1.06836e-01,
        4.03021e-02,
        1.24006e-02,
    ],  # D(TFSI).
    [
        1.74324e-01,
        1.20668e-01,
        9.22893e-02,
        3.31479e-02,
        1.24006e-02,
    ],  # D(Li).
]
Yoshida_2011_G4 = np.asarray(Yoshida_2011_G4)
Yoshida_2011_G4 = np.row_stack([Li_O_ratios_G4, Yoshida_2011_G4])
del Li_O_ratios_G4

# Orädd et al., Solid State Ionics, 2002, 152-153, 131-136.
# Self-diffusion coefficients from PFG-NMR.
# Values taken from Figure 4.
# n_EO = 113498.57 (molecular weight = 5e6 g/mol).
Li_O_ratios_G4 = 1 / np.array([50, 30, 20, 16, 12, 8, 5])
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
Oraedd_2002 = np.row_stack([Li_O_ratios_G4, Oraedd_2002])
del Li_O_ratios_G4

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


def fit_diff_coeff(diff_coeffs, diff_coeffs_sd, Sims, start=0, stop=-1):
    """
    Fit the logarithmic diffusion coefficient as function of the linear
    salt concentration with a straight line.

    The obtained fit parameters are converted from the parameters of a
    straight line to the parameters of the corresponding exponential
    law.
    """
    # Propagation of uncertainty.  See
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
    # Std[ln(A)] = Std[A] / |A|
    sd = diff_coeffs_sd[start:stop] / np.abs(diff_coeffs[start:stop])
    popt, pcov = curve_fit(
        f=leap.misc.straight_line,
        xdata=Sims.Li_O_ratios[start:stop],
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
    + "_diff_coeff_tot.pdf"
)


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_" + args.sol + "_[0-9]*-[0-9]*_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="Li_O_ratio"
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
if args.sol == "g4":
    fit_peo_starts = (
        0,
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 5))[0],
    )
    fit_peo_stops = (
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 6))[0] + 1,
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 2 / 5))[0] + 1,
    )
else:
    fit_peo_starts = (np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 40))[0],)
    fit_peo_stops = (
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 6))[0] + 1,
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
if args.sol == "g4":
    fit_tfsi_starts = (
        0,
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 5))[0],
    )
    fit_tfsi_stops = (
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 6))[0] + 1,
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 2 / 5))[0] + 1,
    )
else:
    fit_tfsi_starts = (
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 40))[0],
    )
    fit_tfsi_stops = (
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 6))[0] + 1,
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
if args.sol == "g4":
    fit_li_starts = (0, np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 5))[0])
    fit_li_stops = (
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 6))[0] + 1,
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 2 / 5))[0] + 1,
    )
else:
    fit_li_starts = (np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 40))[0],)
    fit_li_stops = (
        np.flatnonzero(np.isclose(Sims.Li_O_ratios, 1 / 6))[0] + 1,
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
xlabel = r"Li-to-EO Ratio $r$"
xlim = (0, 0.4 + 0.0125)
labels = ("PEO", "TFSI", "Li")
colors = ("tab:blue", "tab:orange", "tab:green")
markers = ("o", "s", "^")
legend_title = r"$n_{EO} = %d$" % Sims.O_per_chain[0]

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Diffusion coefficients vs salt concentration.
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.errorbar(
            Sims.Li_O_ratios,
            diff_coeffs[cmp_ix],
            yerr=None,  # diff_coeffs_sd[cmp_ix], (SD < symbols).
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
            alpha=leap.plot.ALPHA,
        )
    # PEO fit.
    for fit_ix, popt in enumerate(popt_peo):
        xdata = Sims.Li_O_ratios[
            fit_peo_starts[fit_ix] : fit_peo_stops[fit_ix]
        ]
        fit = leap.misc.exp_law(xdata, *popt)
        # Create an offset to the real data.
        if args.sol == "g1":
            fit *= 1.5
            rotation = -45
            verticalalignment = "bottom"
        elif args.sol == "g4":
            fit *= 1.5
            rotation = np.rad2deg(np.arctan(popt[0])) / 3.4
            verticalalignment = "bottom"
        elif args.sol == "peo63":
            fit /= 1.5
            rotation = -34
            verticalalignment = "top"
        ax.plot(
            xdata, fit, color=colors[labels.index("PEO")], linestyle="dashed"
        )
        ax.text(
            xdata[0],
            fit[0],
            # r"$D_{PEO} \propto \exp[(%.2f \pm %.2f) r]$"
            # % (popt[0], perr_peo[fit_ix][0]),
            r"$D_{PEO} \propto \exp(%.2f r)$" % popt[0],
            rotation=rotation,  # np.rad2deg(np.arctan(popt[0])) / 1.8,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment=verticalalignment,
            fontsize="small",
        )
    # TFSI Fit.
    for fit_ix, popt in enumerate(popt_tfsi):
        xdata = Sims.Li_O_ratios[
            fit_tfsi_starts[fit_ix] : fit_tfsi_stops[fit_ix]
        ]
        fit = leap.misc.exp_law(xdata, *popt)
        if args.sol == "g1":
            fit /= 1.5
            rotation = -40
            verticalalignment = "top"
        elif args.sol == "g4":
            fit /= 1.5
            rotation = np.rad2deg(np.arctan(popt[0])) / 3.15
            verticalalignment = "top"
        elif args.sol == "peo63":
            fit *= 1.5
            rotation = -22
            verticalalignment = "bottom"
        ax.plot(
            xdata, fit, color=colors[labels.index("TFSI")], linestyle="dashed"
        )
        ax.text(
            xdata[0],
            fit[0],
            # r"$D_{TFSI} \propto \exp[(%.2f \pm %.2f) r]$"
            # % (popt[0], perr_tfsi[fit_ix][0]),
            r"$D_{TFSI} \propto \exp(%.2f r)$" % popt[0],
            rotation=rotation,  # np.rad2deg(np.arctan(popt[0])) / 2,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment=verticalalignment,
            fontsize="small",
        )
    # Li Fit.
    for fit_ix, popt in enumerate(popt_li):
        xdata = Sims.Li_O_ratios[fit_li_starts[fit_ix] : fit_li_stops[fit_ix]]
        fit = leap.misc.exp_law(xdata, *popt)
        if args.sol == "g1":
            fit /= 2.5  # Create an offset to the real data.
            rotation = -39
            verticalalignment = "top"
        elif args.sol == "g4":
            fit /= 2.8  # Create an offset to the real data.
            rotation = np.rad2deg(np.arctan(popt[0])) / 3.15
            verticalalignment = "top"
        elif args.sol == "peo63":
            fit *= 1.5
            rotation = -31
            verticalalignment = "bottom"
        ax.plot(
            xdata, fit, color=colors[labels.index("Li")], linestyle="dashed"
        )
        ax.text(
            xdata[0],
            fit[0],
            # r"$D_{Li} \propto \exp[(%.2f \pm %.2f) r}$"
            # % (popt[0], perr_li[fit_ix][0]),
            r"$D_{Li} \propto \exp(%.2f r)$" % popt[0],
            rotation=rotation,  # np.rad2deg(np.arctan(popt[0])) / 2,
            rotation_mode="anchor",
            transform_rotates_text=False,
            horizontalalignment="left",
            verticalalignment=verticalalignment,
            fontsize="small",
        )
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=r"Diff. Coeff. / nm$^2$ ns$^{-1}$", xlim=xlim)
    ax.legend(title=legend_title, loc="upper right")
    pdf.savefig()
    plt.close()

    # Fit start vs salt concentration.
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            Sims.Li_O_ratios,
            fit_starts[cmp_ix],
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
        )
    ax.set(xlabel=xlabel, ylabel="Fit Start / ns", xlim=xlim)
    ax.legend(title=legend_title)
    pdf.savefig()
    # Log scale y.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Fit stop vs salt concentration.
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            Sims.Li_O_ratios,
            fit_stops[cmp_ix],
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
        )
    ax.set(xlabel=xlabel, ylabel="Fit Stop / ns", xlim=xlim)
    ax.legend(title=legend_title)
    pdf.savefig()
    # Log scale y.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Fit stop - Fit start vs salt concentration.
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            Sims.Li_O_ratios,
            np.subtract(fit_stops[cmp_ix], fit_starts[cmp_ix]),
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
        )
    ax.set(xlabel=xlabel, ylabel="Fit Stop - Start / ns", xlim=xlim)
    ax.legend(title=legend_title)
    pdf.savefig()
    # Log scale y.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    # Coefficient of determination (R^2) vs salt concentration.
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            Sims.Li_O_ratios,
            r2s[cmp_ix],
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
        )
    ax.set(xlabel=xlabel, ylabel=r"Coeff. of Determ. $R^2$", xlim=xlim)
    ax.legend(title=legend_title)
    pdf.savefig()
    # Log scale y.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()
    del r2s

    # RMSE vs salt concentration.
    fig, ax = plt.subplots(clear=True)
    for cmp_ix, label in enumerate(labels):
        ax.plot(
            Sims.Li_O_ratios,
            rmses[cmp_ix],
            label=label,
            color=colors[cmp_ix],
            marker=markers[cmp_ix],
        )
    ax.set(xlabel=xlabel, ylabel=r"RMSE / nm$^2$", xlim=xlim)
    ax.legend(title=legend_title)
    pdf.savefig()
    # Log scale y.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()
    del rmses

    # Plot MSDs together with Einstein fit.
    legend_title = r"$n_{EO} = %d$" % Sims.O_per_chain[0] + "\n" + r"$r$"
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
                label=r"$%.4f$" % Sim.Li_O_ratio,
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
                label=r"$%.4f$" % Sim.Li_O_ratio,
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
                label=r"$%.4f$" % Sim.Li_O_ratio,
                alpha=leap.plot.ALPHA,
            )
            # Plot fit residuals / t vs time.
            ax_res_t.plot(
                times_fit,
                msd_fit_res / times_fit,
                label=r"$%.4f$" % Sim.Li_O_ratio,
                alpha=leap.plot.ALPHA,
            )
            # Plot fit residuals / MSD vs time.
            ax_res_msd.plot(
                times_fit,
                msd_fit_res / msd[fit_start_ix:fit_stop_ix],
                label=r"$%.4f$" % Sim.Li_O_ratio,
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
            ncol=1 + Sims.n_sims // (4 + 1),
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
            ncol=1 + Sims.n_sims // (4 + 1),
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
            ncol=1 + Sims.n_sims // (4 + 1),
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
            ncol=1 + Sims.n_sims // (4 + 1),
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
            ncol=1 + Sims.n_sims // (4 + 1),
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
            ncol=1 + Sims.n_sims // (4 + 1),
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
            ncol=1 + Sims.n_sims // (4 + 1),
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
            ncol=1 + Sims.n_sims // (4 + 1),
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
