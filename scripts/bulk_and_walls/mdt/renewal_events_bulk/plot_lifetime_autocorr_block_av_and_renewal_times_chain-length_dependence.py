#!/usr/bin/env python3


"""
Plot the block averaged coordination correlation times and the renewal
times as function of the PEO chain length in one plot.
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
from matplotlib.ticker import MaxNLocator

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the block averaged coordination correlation times and the"
        " renewal times as function of the PEO chain length in one plot."
    )
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
prefix = settings + "_lintf2_peoN_20-1_sc80_"
infile_renew_peo = prefix + "renewal_times_Li-ether_continuous.txt.gz"
# infile_renew_tfsi = prefix + "renewal_times_Li-NTf2_continuous.txt.gz"
outfile = prefix + "lifetime_autocorr_block_average_and_renewal_times.pdf"

cmps = ("Li-OE", "Li-OBT", "Li-ether", "Li-NTf2")
analysis = "lifetime_autocorr"  # Analysis name.
analysis_dir = analysis + "_block_average"
tool = "mdt"  # Analysis software.


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="O_per_chain"
)


print("Reading data...")
# Renewal times.
rnw_peo_data = np.loadtxt(infile_renew_peo, usecols=(0, 29))
# rnw_tfsi_data = np.loadtxt(infile_renew_tfsi, usecols=(0, 29))

# Correlation times.
acf_data = np.full((3, len(cmps), Sims.n_sims), np.nan, dtype=np.float32)
for cmp_ix, cmp in enumerate(cmps):
    analysis_suffix = "_" + cmp  # Analysis name specification.
    ana_path = os.path.join(analysis_dir, analysis + analysis_suffix)
    file_suffix = analysis_dir + analysis_suffix + ".txt.gz"
    infiles = leap.simulation.get_ana_files(Sims, ana_path, tool, file_suffix)
    for sim_ix, infile in enumerate(infiles):
        lifetimes = np.loadtxt(infile, usecols=(3,))
        n_lifetime_measurements = np.count_nonzero(np.isfinite(lifetimes))
        if n_lifetime_measurements == 1:
            lifetime_av = lifetimes[np.isfinite(lifetimes)][0]
            lifetime_se = np.nan
        elif n_lifetime_measurements > 1:
            lifetime_av = np.nanmean(lifetimes)
            lifetime_se = np.nanstd(lifetimes, ddof=1) / np.sqrt(
                n_lifetime_measurements
            )
        else:
            lifetime_av, lifetime_se = np.nan, np.nan
        # Mean
        acf_data[0][cmp_ix][sim_ix] = lifetime_av
        # Standard error
        acf_data[1][cmp_ix][sim_ix] = lifetime_se
        # Number of n_lifetime_measurements
        acf_data[2][cmp_ix][sim_ix] = n_lifetime_measurements
    del lifetimes, n_lifetime_measurements, lifetime_av, lifetime_se

print("Creating plot(s)...")
xlabel = r"Ether Oxygens per Chain $n_{EO}$"
xlim = (1, 200)
legend_title = r"$r = 0.05$"
labels = (
    r"$Li-O_{PEO}$ Correlation",
    r"$Li-O_{TFSI}$ Correlation",
    r"$Li-PEO$ Correlation",
    r"$Li-TFSI$ Correlation",
    r"$Li-PEO$ Renewal",
    # r"$Li-TFSI$ Renewal",
)
colors = ("tab:blue", "tab:cyan", "tab:red", "tab:orange", "tab:purple")
markers = ("^", "v", "s", "D", "o")
linestyles = ("solid",) * 4 + ("dashed",)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot renewal and correlation times.
    fig, ax = plt.subplots(clear=True)
    for i, label in enumerate(labels):
        if label.startswith(r"$Li-PEO$ Renewal"):
            xdata, ydata = rnw_peo_data[:, 0], rnw_peo_data[:, 1]
            yerr = None
        # elif label.startswith(r"$Li-TFSI$ Renewal"):
        #     xdata, ydata = rnw_tfsi_data[:, 0], rnw_tfsi_data[:, 1]
        #     yerr = None
        else:
            xdata = Sims.O_per_chain
            ydata, yerr = acf_data[0, i], acf_data[1, i]
        # valid = np.isfinite(ydata)
        # xdata, ydata, yerr = xdata[valid], ydata[valid], yerr[valid]
        ax.errorbar(
            xdata,
            ydata,
            yerr=yerr,
            label=label,
            color=colors[i],
            marker=markers[i],
            linestyle=linestyles[i],
            alpha=leap.plot.ALPHA,
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel="Time Scale / ns", xlim=xlim)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    legend = ax.legend(title=legend_title, **mdtplt.LEGEND_KWARGS_XSMALL)
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    yd_min, yd_max = leap.plot.get_ydata_min_max(ax)
    if np.any(np.greater(yd_min, 0)):
        # Log scale y.
        ax.relim()
        ax.autoscale()
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set_xlim(xlim)
        pdf.savefig()
    plt.close()

    # Plot number of measurements.
    fig, ax = plt.subplots(clear=True)
    for i, label in enumerate(labels):
        if label.startswith(r"$Li-PEO$ Renewal") or label.startswith(
            r"$Li-TFSI$ Renewal"
        ):
            continue
        ax.plot(
            Sims.O_per_chain,
            acf_data[2, i],
            label=label,
            color=colors[i],
            marker=markers[i],
            linestyle=linestyles[i],
            alpha=leap.plot.ALPHA,
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_tick_params(which="minor", bottom=False, top=False)
    ax.set(
        xlabel=xlabel,
        ylabel="No. of Measurements",
        xlim=xlim,
        ylim=(-0.5, np.max(acf_data[2]) + 0.5),
    )
    legend = ax.legend(title=legend_title, **mdtplt.LEGEND_KWARGS_XSMALL)
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot standard errors.
    fig, ax = plt.subplots(clear=True)
    for i, label in enumerate(labels):
        if label.startswith(r"$Li-PEO$ Renewal") or label.startswith(
            r"$Li-TFSI$ Renewal"
        ):
            continue
        else:
            xdata = Sims.O_per_chain
            ydata = acf_data[1, i]
        # valid = np.isfinite(ydata)
        # xdata, ydata, = xdata[valid], ydata[valid]
        ax.plot(
            xdata,
            ydata,
            label=label,
            color=colors[i],
            marker=markers[i],
            linestyle=linestyles[i],
            alpha=leap.plot.ALPHA,
        )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel="Std. Err. of Corr. Times / ns", xlim=xlim)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    legend = ax.legend(title=legend_title, **mdtplt.LEGEND_KWARGS_XSMALL)
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    yd_min, yd_max = leap.plot.get_ydata_min_max(ax)
    if np.any(np.greater(yd_min, 0)):
        # Log scale y.
        ax.relim()
        ax.autoscale()
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set_xlim(xlim)
        pdf.savefig()
    plt.close()

    # Compare block averaged correlation times to correlation times
    # calculated using the full trajectory.
    fig, ax = plt.subplots(clear=True)
    # Block averaged correlation times.
    for i, label in enumerate(labels):
        if label.startswith(r"$Li-PEO$ Renewal") or label.startswith(
            r"$Li-TFSI$ Renewal"
        ):
            continue
        else:
            xdata = Sims.O_per_chain
            ydata, yerr = acf_data[0, i], acf_data[1, i]
        # valid = np.isfinite(ydata)
        # xdata, ydata, yerr = xdata[valid], ydata[valid], yerr[valid]
        ax.errorbar(
            xdata,
            ydata,
            yerr=yerr,
            label=None,
            color=leap.plot.change_brightness(colors[i], 0.8),
            marker=markers[i],
            linestyle="solid",
            alpha=leap.plot.ALPHA,
        )
    # Correlation times calculated from full trajectory.
    path_to_acf = "../../lifetime_autocorr/chain-length_dependence/"
    infile_acf = (
        path_to_acf
        + prefix
        + "lifetime_autocorr_combined_Li-OE_Li-OBT_Li-ether_Li-NTf2.txt.gz"
    )
    acf_data = np.loadtxt(infile_acf)
    for i, label in enumerate(labels):
        if label.startswith(r"$Li-O_{PEO}$ Correlation"):
            xdata, ydata = acf_data[:, 0], acf_data[:, 1]
        elif label.startswith(r"$Li-O_{TFSI}$ Correlation"):
            xdata, ydata = acf_data[:, 0], acf_data[:, 2]
        elif label.startswith(r"$Li-PEO$ Correlation"):
            xdata, ydata = acf_data[:, 0], acf_data[:, 3]
        elif label.startswith(r"$Li-TFSI$ Correlation"):
            xdata, ydata = acf_data[:, 0], acf_data[:, 4]
        elif label.startswith(r"$Li-PEO$ Renewal"):
            xdata, ydata = rnw_peo_data[:, 0], rnw_peo_data[:, 1]
        # elif label.startswith(r"$Li-TFSI$ Renewal"):
        #     xdata, ydata = rnw_tfsi_data[:, 0], rnw_tfsi_data[:, 1]
        else:
            raise ValueError("Unknown 'label': '{}'".format(label))
        valid = np.isfinite(ydata)
        xdata, ydata = xdata[valid], ydata[valid]
        ax.plot(
            xdata,
            ydata,
            label=label,
            color=leap.plot.change_brightness(colors[i], 1.4),
            marker=markers[i],
            linestyle="dashed",
            alpha=leap.plot.ALPHA,
        )
    ax.plot([], [], color="black", linestyle="solid", label="Block average")
    ax.plot([], [], color="gray", linestyle="dashed", label="Full trajectory")
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel="Time Scale / ns", xlim=xlim)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    legend = ax.legend(title=legend_title, **mdtplt.LEGEND_KWARGS_XSMALL)
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    yd_min, yd_max = leap.plot.get_ydata_min_max(ax)
    if np.any(np.greater(yd_min, 0)):
        # Log scale y.
        ax.relim()
        ax.autoscale()
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set_xlim(xlim)
        pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
