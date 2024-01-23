#!/usr/bin/env python3


"""
Plot free-energy profiles along the first- or second-nearest neighbor
axes of a hexagonal lattice for different chain lengths.

Extract the positions of the extrema of the free-energy profile and
calculate the free-energy barriers.
"""


# Standard libraries
import argparse
import glob
import os
import re

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy import constants
from scipy.signal import find_peaks, savgol_filter

# First-party libraries
import lintf2_ether_ana_postproc as leap


def equalize_yticks(ax):
    """
    Equalize y-ticks so that plots can be better stacked together.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` for which to equalize the y
        ticks.
    """
    ylim = np.asarray(ax.get_ylim())
    ylim_diff = ylim[-1] - ylim[0]
    yticks = np.asarray(ax.get_yticks())
    yticks_valid = (yticks >= ylim[0]) & (yticks <= ylim[-1])
    yticks = yticks[yticks_valid]
    if ylim_diff >= 10 and ylim_diff < 20:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if np.all(yticks >= 0) and np.all(yticks < 10) and ylim_diff > 2:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def get_slab(fname, prefix):
    """Get the position of the analyzed slab from the file name."""
    if not os.path.isfile(fname):
        raise FileNotFoundError("No such file: '{}'".format(fname))
    fname = os.path.basename(fname)  # Remove path to the file.
    fname = os.path.splitext(fname)[0]  # Remove (first) file extension.
    if not fname.startswith(prefix):
        raise ValueError(
            "The file name '{}' does not start with '{}'".format(fname, prefix)
        )
    slab = fname[len(prefix) :]  # Remove `prefix`.
    slab = re.sub("[^0-9|.|-]", "", slab)  # Remove non-numeric characters.
    slab = slab.strip(".")  # Remove leading and trailing periods.
    slab = slab.split("-")  # Split at hyphens.
    if len(slab) != 2:
        raise ValueError("Invalid slab: {}".format(slab))
    slab = [float(slab) for slab in slab]
    slab_start, slab_stop = min(slab), max(slab)
    return slab_start, slab_stop


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot free-energy profiles along the first- or second-nearest neighbor"
        " axes of a hexagonal lattice for different chain lengths."
    )
)
parser.add_argument(
    "--surfq",
    type=str,
    required=False,
    choices=("q1",),  # "q0", "q0.25", "q0.5", "q0.75"),
    default="q1",
    # Other layers than the 1st and 2nd Li-layer at negative electrodes
    # with a surface charge of q = -1 e/nm^2 (might) require clustering
    # of the slabs.
    help="Surface charge in e/nm^2.",
)
parser.add_argument(
    "--cmp",
    type=str,
    required=False,
    choices=("Li",),  # "NBT", "OBT", "OE"),
    default="Li",
    # Other layers than the 1st and 2nd Li-layer at negative electrodes
    # with a surface charge of q = -1 e/nm^2 (might) require clustering
    # of the slabs.
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--axis",
    type=int,
    required=False,
    default=1,
    choices=(1, 2),
    help=(
        "Whether to use the first- or second-nearest neighbor axis.  Default:"
        " %(default)s"
    ),
)
args = parser.parse_args()

temp = 423
settings = "pr_nvt%d_nh" % temp  # Simulation settings.
analysis = "axial_hex_dist"  # Analysis name.
analysis_axis = "_%dnn" % args.axis
analysis_suffix = "_" + args.cmp
analysis_tot = analysis + analysis_axis + analysis_suffix
ana_path = os.path.join(analysis, analysis + analysis_axis, analysis_tot)
tool = "mdt"  # Analysis software.
outfile_base = (
    settings
    + "_lintf2_peoN_20-1_gra_"
    + args.surfq
    + "_sc80_"
    + analysis_tot
    + "_free_energy"
)
outfile_txt = outfile_base + ".txt.gz"
outfile_pdf = outfile_base + ".pdf"

# C-C bond length in graphene in nm.
r0 = 0.142
# Ideal peak distance in nm.
if args.axis == 1:
    ideal_distance_nm = np.sqrt(3) * r0  # Lattice constant.
elif args.axis == 2:
    ideal_distance_nm = 3 * r0
else:
    raise ValueError("Unknown --axis: '{}'".format(args.axis))

# Parameters for peak finding with `scipy.signal.find_peaks`.
# Vertical properties are given in data units, horizontal properties
# are given in number of sample points if not specified otherwise.
distance_nm = 0.8 * ideal_distance_nm  # Minimum distance between peaks.
prominence = 0.01  # Required minimum peak prominence in kT.
min_width = 2  # Required minimum width at `rel_height` in sample points
max_width_nm = 1.2 * ideal_distance_nm  # Maximum width at `rel_height`.
rel_height = 0.2  # Relative height at which the peak width is measured.

# Parameters for data smoothing with a Savitzky-Golay filter.
# The advantage of a Savitzky-Golay filter compared to a moving average
# is that it preserves relative extrema while the moving average
# flattens them.  Note that a Savitzky-Golay filter with a polynomial
# order of zero is identical to a moving average.
polyorder = 3  # Order or the polynomial used to fit the samples.
wlen = 15  # Length of the filter window.

# beta = k * T
beta = constants.k * temp
beta = 1  # Because the free energy is already given in units of kT.


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_gra_" + args.surfq + "_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, args.surfq, sort_key="O_per_chain"
)


print("Reading data...")
# Get input files that contain the average compound density in each bin.
analysis_bin_pop = "discrete-z"
file_suffix_bin_pop = analysis_bin_pop + "_Li_bin_population.txt.gz"
infiles_bin_pop = leap.simulation.get_ana_files(
    Sims, analysis_bin_pop, tool, file_suffix_bin_pop
)

# Get the input files that contain the density along the hexagonal axes.
file_extension = ".txt.gz"
file_suffix_pattern = analysis_tot + "_[0-9]*-[0-9]*.txt.gz"
infiles = [None for Sim in Sims.sims]
slab_widths = np.full(Sims.n_sims, np.nan, dtype=np.float64)
slab_dens = np.full_like(slab_widths, np.nan)
for sim_ix, path in enumerate(Sims.paths_ana):
    fname_pattern = Sims.fnames_ana_base[sim_ix] + file_suffix_pattern
    fpath_pattern = os.path.join(path, tool, ana_path, fname_pattern)
    files = glob.glob(fpath_pattern)
    if len(files) == 0:
        raise ValueError(
            "Could not find any file matching the pattern"
            " '{}'".format(fpath_pattern)
        )

    # Get the file that contains the data for the slab/bin that is
    # closest to the negative electrode.
    file_prefix = Sims.fnames_ana_base[sim_ix] + analysis_tot
    slab_starts = np.full(len(files), np.nan, dtype=np.float64)
    slab_stops = np.full_like(slab_starts, np.nan)
    for f_ix, file in enumerate(files):
        slab_starts[f_ix], slab_stops[f_ix] = get_slab(file, file_prefix)
    ix_max = np.argmax(slab_starts)
    infiles[sim_ix] = files[ix_max]
    slab_widths[sim_ix] = slab_stops[ix_max] - slab_starts[ix_max]

    # Get the average compound density in the slab/bin.
    tol = 0.02
    bin_starts, bin_stops, bin_dens = np.loadtxt(
        infiles_bin_pop[sim_ix], usecols=(0, 1, 6), unpack=True
    )
    bin_starts, bin_stops = np.round(bin_starts, 2), np.round(bin_stops, 2)
    bin_ix_start = np.flatnonzero(
        np.isclose(bin_starts, slab_starts[ix_max], rtol=0, atol=tol)
    )
    bin_ix_stop = np.flatnonzero(
        np.isclose(bin_stops, slab_stops[ix_max], rtol=0, atol=tol)
    )
    if len(bin_ix_start) != 1:
        raise ValueError(
            "`len(bin_ix_start)` ({}) != 1".format(len(bin_ix_start))
        )
    if len(bin_ix_stop) != 1:
        raise ValueError(
            "`len(bin_ix_stop)` ({}) != 1".format(len(bin_ix_stop))
        )
    bin_ix_start, bin_ix_stop = bin_ix_start[0], bin_ix_stop[0]
    if bin_ix_stop != bin_ix_start:
        raise ValueError(
            "`bin_ix_stop` ({}) != `bin_ix_start`"
            " ({})".format(bin_ix_stop, bin_ix_start)
        )
    slab_dens[sim_ix] = bin_dens[bin_ix_start]  # 1/Angstrom^3.
del files, slab_starts, slab_stops
slab_widths /= 10  # Angstrom -> nm.

# Read data and calculate free-energy barriers.
peak_finding_factors = (-1, 1)  # -1 for finding minima, 1 for maxima.
minima_ix = peak_finding_factors.index(-1)
maxima_ix = peak_finding_factors.index(1)

xdata = [None for Sim in Sims.sims]  # Position on the hexagonal axis.
ydata = [None for Sim in Sims.sims]  # Corresponding free energy.
ydata_smoothed = [None for Sim in Sims.sims]
peak_ix = [[None for Sim in Sims.sims] for fac in peak_finding_factors]
n_peaks = np.zeros(Sims.n_sims, dtype=np.uint32)
barriers = np.full(Sims.n_sims, np.nan, dtype=np.float64)
barriers_sd = np.full_like(barriers, np.nan)
for sim_ix, infile in enumerate(infiles):
    data = np.loadtxt(infile, usecols=(0, 1, 2))
    xdat = data[:, 0] / 10  # Angstrom -> nm.
    ydat = np.nanmean(data[:, 1:], axis=1)  # 1/Angstrom^3.
    ydat /= slab_dens[sim_ix]
    del data

    # Calculate free-energy.
    ydat = -np.log(ydat)
    xdata[sim_ix], ydata[sim_ix] = xdat, ydat

    # `scipy.signal.find_peaks` cannot handle NaN values.
    if not np.all(np.isfinite(xdat)) or not np.all(np.isfinite(ydat)):
        raise ValueError(
            "Simulation: {}\n".format(Sims.paths[sim_ix])
            + "Encountered non-finite values in the free-energy profile"
        )

    # Smooth data with Savitzky-Golay filter.
    ydat = savgol_filter(
        ydat, window_length=wlen, polyorder=polyorder, mode="wrap"
    )
    ydata_smoothed[sim_ix] = ydat

    # Minimum distance between two peaks and maximum peak width for peak
    # finding with `scipy.signal.find_peaks`.
    x_sample_spacing = np.mean(np.diff(xdat))
    n_samples_per_nm = round(1 / x_sample_spacing)
    distance = int(distance_nm * n_samples_per_nm)
    max_width = int(max_width_nm * n_samples_per_nm)
    width = (min_width, max_width)  # Discard noise and broad peaks.

    # Find free-energy extrema.
    extrema = [None for fac in peak_finding_factors]
    for fac_ix, fac in enumerate(peak_finding_factors):
        peaks, _properties = find_peaks(
            fac * ydat,
            distance=distance,
            prominence=prominence,
            width=width,
            rel_height=rel_height,
        )
        extrema[fac_ix] = ydat[peaks]
        peak_ix[fac_ix][sim_ix] = peaks
    del _properties
    # Because the free-energy profile is periodic with an extrema at the
    # boundaries, there might be found one more minimum than maxima (or
    # the other way round).  Therefore, discard the excessive extremum.
    n_extrema = min(len(extrema_fac) for extrema_fac in extrema)
    n_peaks[sim_ix] = n_extrema
    extrema = [extrema_fac[:n_extrema] for extrema_fac in extrema]
    for fac_ix in range(len(peak_finding_factors)):
        peak_ix[fac_ix][sim_ix] = peak_ix[fac_ix][sim_ix][:n_extrema]

    # Calculate the average free-energy barrier.
    barriers[sim_ix] = np.mean(extrema[maxima_ix] - extrema[minima_ix])
    barriers_sd[sim_ix] = np.std(extrema[maxima_ix] - extrema[minima_ix])
    barriers_sd[sim_ix] /= np.sqrt(n_extrema)

    # Shift free-energy profiles such that the minima are zero.
    ydata[sim_ix] -= np.mean(extrema[maxima_ix])
    ydata_smoothed[sim_ix] -= np.mean(extrema[maxima_ix])
del extrema


print("Creating output file(s)...")
data = np.column_stack([Sims.O_per_chain, barriers, barriers_sd])
header = "Free-energy barriers along hexagonal axes.\n" + "\n"
if args.axis == 1:
    header += "Axis:            {}st nearest-neighbor axis\n".format(args.axis)
elif args.axis == 2:
    header += "Axis:            {}nd nearest-neighbor axis\n".format(args.axis)
else:
    raise ValueError("Unknown --axis: '{}'".format(args.axis))
header += (
    "Compound:        {}\n".format(args.cmp)
    + "Surface charge: -{:.2f} e/nm^2\n".format(Sims.surfqs[0])
    + "Li-to-EO ratio:  {:.4f}\n".format(Sims.Li_O_ratios[0])
    + "\n\n"
    + "Peak finding procedure:\n"
    + "\n"
    + "1) The free-energy profile F(r) is calculated from the density\n"
    + "profile p(r) according to\n"
    + "  F(r)/kT = -ln[p(r)/p0]\n"
    + "where k is the Boltzmann constant and T is the temperature.  p0 is\n"
    + "chosen such that the free-energy maxima are zero.\n"
    + "\n"
    + "2) The free-energy profile is smoothed using a Savitzky-Golay filter.\n"
    + "\n"
    + "3) Free-energy extrema are identified on the basis of their peak\n"
    + "prominence and width and the distance between neighboring peaks.\n"
    + "\n"
    + "Parameters for data smoothing with scipy.signal.savgol_filter\n"
    + "polyorder:     {:d}\n".format(polyorder)
    + "window_length: {:d} sample points\n".format(wlen)
    + "\n"
    + "Parameters for peak finding with scipy.signal.find_peaks\n"
    + "distance_nm:  {:.9e} nm\n".format(distance_nm)
    + "prominence:   {:.9e} kT\n".format(prominence)
    + "min_width:    {:d} sample points\n".format(min_width)
    + "max_width_nm: {:.9e} nm\n".format(max_width_nm)
    + "rel_height:   {:.3f}\n".format(rel_height)
    + "\n\n"
    + "The columns contain:\n"
    + "  1 Number of ether oxygens per PEO chain\n"
    + "  2 Free-energy barrier / kT\n"
    + "  3 Standard deviation of the free-energy barrier / kT\n"
    + "\n"
    + "Column numbers:\n"
    + "{:>14d}".format(1)
)
for col_num in range(2, data.shape[1] + 1):
    header += " {:>16d}".format(col_num)
leap.io_handler.savetxt(outfile_txt, data, header=header)
print("Created {}".format(outfile_txt))


print("Creating plot(s)...")
legend_title = (
    r"$\sigma_s = -%.2f$ $e$/nm$^2$, " % Sims.surfqs[0]
    + r"$r = %.2f$" % Sims.Li_O_ratios[0]
    + "\n"
    + r"$n_{EO}$"
)
legend_loc = "upper center"
n_legend_cols = Sims.n_sims // 2

cmap = plt.get_cmap()
c_vals = np.arange(Sims.n_sims)
c_norm = max(1, Sims.n_sims - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile_pdf)
with PdfPages(outfile_pdf) as pdf:
    xlabel = r"Ether Oxygens per Chain $n_{EO}$"
    xlim = (1, 200)

    # Plot slab widths.
    ylabel = "Layer Width / nm"
    fig, ax = plt.subplots(clear=True)
    ax.plot(Sims.O_per_chain, slab_widths, marker="o")
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    equalize_yticks(ax)
    legend = ax.legend(title=legend_title.split("\n")[0].replace(", ", "\n"))
    legend.get_title().set_multialignment("center")
    pdf.savefig(fig)
    plt.close(fig)

    # Plot number of free-energy extrema.
    ylabel = "No. of Peaks / nm"
    fig, ax = plt.subplots(clear=True)
    ax.plot(Sims.O_per_chain, n_peaks, marker="o")
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    equalize_yticks(ax)
    legend = ax.legend(title=legend_title.split("\n")[0].replace(", ", "\n"))
    legend.get_title().set_multialignment("center")
    pdf.savefig(fig)
    plt.close(fig)

    # Plot free-energy barriers.
    ylabel = (
        r"Barrier Height $\Delta F_{"
        + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
        + r"}(r) / k_B T$"
    )
    fig, ax = plt.subplots(clear=True)
    ax.errorbar(Sims.O_per_chain, barriers, yerr=barriers_sd, marker="o")
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    equalize_yticks(ax)
    legend = ax.legend(title=legend_title.split("\n")[0].replace(", ", "\n"))
    legend.get_title().set_multialignment("center")
    pdf.savefig(fig)
    plt.close(fig)

    # Plot inverse transition rates.
    ylabel = "Inv. Trans. Rate / a.u."
    fig, ax = plt.subplots(clear=True)
    # Propagation of uncertainty
    # (https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae).
    # Std[1/A] = 1/|A| * Std[A]/|A| = Std[A]/A^2
    # Std[exp(bA)] = exp(bA) * |b| * Std[A]
    # Std[1/exp(bA)] = Std[exp(bA)]/exp(bA)^2
    #               = exp(bA) * |b| * Std[A] / exp(bA)^2
    #               = |b| * Std[A] / exp(bA)
    ax.errorbar(
        Sims.O_per_chain,
        1 / np.exp(-beta * barriers),
        yerr=np.abs(-beta) * barriers_sd / np.exp(-beta * barriers),
        marker="o",
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    equalize_yticks(ax)
    legend = ax.legend(title=legend_title.split("\n")[0].replace(", ", "\n"))
    legend.get_title().set_multialignment("center")
    pdf.savefig(fig)
    plt.close(fig)

    xlabel = r"$r$ / nm"
    ylabel = (
        r"Free Energy $F_{"
        + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
        + r"}(r) / k_B T$"
    )
    # xlim_full = (0, 19 * r0)
    xlim = (0, 6 * r0)
    ylim = (-1.8, 1.1)  # If free-energy maxima are set to zero.
    # ylim = (-0.2, 2.8)  # If free-energy minima are set to zero.

    # Plot free energy.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    fig_peaks, ax_peaks = plt.subplots(clear=True)
    ax_peaks.set_prop_cycle(color=colors)
    fig_peaks_only, ax_peaks_only = plt.subplots(clear=True)
    ax_peaks_only.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        if Sim.O_per_chain == 3:
            linestyle = "dashed"
        else:
            linestyle = None
        if Sim.O_per_chain == 4:
            color = "tab:red"
            ax.plot([], [])  # Increment color cycle.
        elif Sim.O_per_chain == 5:
            color = "tab:orange"
            ax.plot([], [])  # Increment color cycle.
        elif Sim.O_per_chain == 8:
            color = "tab:brown"
            ax.plot([], [])  # Increment color cycle.
        else:
            color = None
        ax.plot(
            xdata[sim_ix],
            ydata[sim_ix],
            label=r"$%d$" % Sims.O_per_chain[sim_ix],
            color=color,
            linestyle=linestyle,
            alpha=leap.plot.ALPHA,
        )
        lines = ax_peaks.plot(
            xdata[sim_ix],
            ydata[sim_ix],
            label=r"$%d$" % Sims.O_per_chain[sim_ix],
            linewidth=1,
            alpha=leap.plot.ALPHA,
        )
        ax_peaks.plot(
            xdata[sim_ix],
            ydata_smoothed[sim_ix],
            label="Smoothed" if sim_ix == Sims.n_sims - 1 else None,
            color=lines[0].get_color(),
            linestyle="dashed",
            alpha=leap.plot.ALPHA,
        )
        for fac_ix, fac in enumerate(peak_finding_factors):
            if fac == -1:
                marker = "2"
            elif fac == 1:
                marker = "1"
            else:
                raise ValueError("Unknown factor: {}".format(fac))
            ax_peaks.scatter(
                xdata[sim_ix][peak_ix[fac_ix][sim_ix]],
                ydata_smoothed[sim_ix][peak_ix[fac_ix][sim_ix]],
                color=lines[0].get_color(),
                marker=marker,
                alpha=leap.plot.ALPHA,
                zorder=2.001,
                # `zorder` of lines is 2, `zorder` of major ticks is
                # 2.01 -> set `zorder` of the scatter points to 2.001 to
                # ensure that they lie above lines but below the major
                # ticks.
            )
            ax_peaks_only.scatter(
                xdata[sim_ix][peak_ix[fac_ix][sim_ix]],
                ydata_smoothed[sim_ix][peak_ix[fac_ix][sim_ix]],
                label=(
                    r"$%d$" % Sims.O_per_chain[sim_ix] if fac_ix == 0 else None
                ),
                marker=marker,
                alpha=leap.plot.ALPHA,
            )

    # Original free-energy profiles.
    ax.set(xlabel=xlabel, ylabel=ylabel, ylim=ylim)
    ax.margins(x=0)
    equalize_yticks(ax)
    legend = ax.legend(
        title=legend_title,
        loc=legend_loc,
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig(fig)

    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    pdf.savefig(fig)
    plt.close(fig)

    # Smoothed free-energy profiles and peak positions.
    ax_peaks.set(xlabel=xlabel, ylabel=ylabel, ylim=ylim)
    ax_peaks.margins(x=0)
    equalize_yticks(ax_peaks)
    legend = ax_peaks.legend(
        title=legend_title,
        loc=legend_loc,
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig(fig_peaks)

    ax_peaks.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    pdf.savefig(fig_peaks)
    plt.close(fig_peaks)

    # Peak positions only.
    ax_peaks_only.set(xlabel=xlabel, ylabel=ylabel, ylim=ylim)
    ax_peaks_only.margins(x=0)
    equalize_yticks(ax_peaks_only)
    legend = ax_peaks_only.legend(
        title=legend_title,
        loc=legend_loc,
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig(fig_peaks_only)

    ax_peaks_only.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    pdf.savefig(fig_peaks_only)
    plt.close(fig_peaks_only)

print("Created {}".format(outfile_pdf))
print("Done")
