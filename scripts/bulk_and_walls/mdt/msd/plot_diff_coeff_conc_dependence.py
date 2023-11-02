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

# First-party libraries
import lintf2_ether_ana_postproc as leap


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


print("Creating plot(s)...")
xlabel = r"Li-to-EO Ratio $r$"
xlim = (0, 0.4 + 0.0125)
labels = ("PEO", "TFSI", "Li")
colors = ("tab:blue", "tab:orange", "tab:green")
markers = ("o", "s", "^")

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
    ax.set(xlabel=xlabel, ylabel=r"Diff. Coeff. / nm$^2$ ns$^{-1}$", xlim=xlim)
    ax.legend(loc="upper right")
    pdf.savefig()
    # Log scale y.
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
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
    ax.legend()
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
    ax.legend()
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
    ax.legend()
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
    ax.legend()
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
    ax.legend()
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
