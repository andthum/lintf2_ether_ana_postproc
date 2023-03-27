#!/usr/bin/env python3


"""
Extract the positions of extrema of free-energy profiles for different
compounds of a single simulation.
"""


# Standard libraries
import argparse
import glob
import os
import warnings
from copy import deepcopy

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
from scipy.signal import find_peaks, peak_widths, savgol_filter

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Extract the positions of extrema of free-energy profiles for"
        " different different compounds of a single simulation."
    )
)
parser.add_argument(
    "--system",
    type=str,
    required=True,
    help="Name of the simulated system, e.g. lintf2_g1_20-1_gra_q1_sc80.",
)
parser.add_argument(
    "--settings",
    type=str,
    required=False,
    default="pr_nvt423_nh",
    help=(
        "String describing the used simulation settings.  Default:"
        " %(default)s."
    ),
)
args = parser.parse_args()

analysis = "density-z"  # Analysis name.
analysis_suffix = "_number"  # Analysis name specification.
tool = "gmx"  # Analysis software.
outfile_base = args.settings + "_" + args.system + "_free_energy_"
outfile_pdf = outfile_base + "extrema.pdf"

cols = (  # Columns to read from the input file(s).
    0,  # bin edges [nm]
    2,  # Li number density [nm^-3]
    5,  # NBT number density [nm^-3]
    6,  # OBT number density [nm^-3]
    7,  # OE number density [nm^-3]
)
compounds = ("Li", "NBT", "OBT", "OE")
if len(compounds) != len(cols) - 1:
    raise ValueError(
        "`len(compounds)` ({}) != `len(cols) - 1`"
        " ({})".format(len(compounds), len(cols) - 1)
    )

# Parameters for peak finding with `scipy.signal.find_peaks`.
# Vertical properties are given in data units, horizontal properties
# are given in number of sample points if not specified otherwise.
prominence = 0.25  # kT/2 = Thermal energy in one dimension (E=3kT/2).
min_width = 2  # Required minimum width in number of sample points.
max_width_nm = 2  # Maximum width at `rel_height` in nm.
rel_height = 0.2  # Relative height at which the peak width is measured.

# Parameters for finding outliers with `scipy.signal.find_peaks`.
sd_factor = 4  # threshold = sd_factor * bulk_standard_deviation.
outlier_max_width = min_width
outlier_rel_height = rel_height

# Parameters for data smoothing with a Savitzky-Golay filter.
# The advantage of a Savitzky-Golay filter compared to a moving average
# is that it preserves relative extrema while the moving average
# flattens them.  Note that a Savitzky-Golay filter with a polynomial
# order of zero is identical to a moving average.
polyorder = 3  # Order or the polynomial used to fit the samples.
wlen = 9  # Length of the filter window.


print("Creating Simulation instance(s)...")
SimPaths = leap.simulation.SimPaths()
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    top_path = "q%g" % surfq
else:
    surfq = None
    top_path = "bulk"
pattern_settings = "[0-9][0-9]_" + args.settings + "_" + args.system
pattern = os.path.join(SimPaths.PATHS[top_path], args.system, pattern_settings)
paths = glob.glob(pattern)
if len(paths) < 1:
    raise ValueError(
        "Could not find a directory that matches the pattern"
        " '{}'".format(pattern)
    )
elif len(paths) > 1:
    raise ValueError(
        "Found more than one directory that matches the pattern"
        " '{}'".format(pattern)
    )
Sim = leap.simulation.Simulation(paths[0])


print("Assembling input file name(s)...")
file_suffix = analysis + analysis_suffix + ".xvg.gz"
fname = Sim.fname_ana_base + file_suffix
infile = os.path.join(Sim.path_ana, tool, analysis, fname)
if not os.path.isfile(infile):
    raise FileNotFoundError("No such file: '{}'".format(infile))


print("Reading data...")
data = np.loadtxt(infile, comments=["#", "@"], usecols=cols, unpack=True)
xdata, ydata = data[0], data[1:]
n_samples_half = len(xdata) // 2

# Maximum peak width for peak finding with `scipy.signal.find_peaks`.
x_sample_spacing = np.mean(np.diff(xdata))
n_samples_per_nm = round(1 / x_sample_spacing)
max_width = max_width_nm * n_samples_per_nm
peak_finding_factors = (-1, 1)  # -1 for finding minima, 1 for maxima.

# Calculate free-energy profiles from density profiles.
bulk_region = Sim.bulk_region / 10  # A -> nm
for i, y in enumerate(ydata):
    ydata[i] = leap.misc.dens2free_energy(xdata, y, bulk_region=bulk_region)

# Calculate the threshold for outlier detection from the standard
# deviation of the free-energy profile in the bulk region.
bulk_begin_ix, bulk_end_ix = leap.misc.find_nearest(xdata, bulk_region)
outlier_thresholds = np.nanstd(
    ydata[:, bulk_begin_ix:bulk_end_ix], axis=-1, ddof=1
)
outlier_thresholds *= sd_factor


print("Finding extrema and creating plot(s)...")
Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm
box_z = Sim.box[2] / 10  # A -> nm

plot_sections = ("left", "right")
xmin = 0
xmax = Elctrd.BULK_START / 10  # A -> nm

ix_shift = np.zeros((len(plot_sections), len(compounds)), dtype=np.uint32)
peaks = [
    [[None for fac in peak_finding_factors] for cmp in compounds]
    for plt_sec in plot_sections
]
# Peak heights must be stored explicitly, because the smoothed y data
# are not stored and thus the peak heights cannot be recovered by
# `y[peaks]`.
peak_heights = deepcopy(peaks)
properties = deepcopy(peaks)
base_widths = deepcopy(peaks)
fwhm = deepcopy(peaks)

mdt.fh.backup(outfile_pdf)
with PdfPages(outfile_pdf) as pdf:
    for plt_ix, plt_sec in enumerate(plot_sections):
        for cmp_ix, cmp in enumerate(compounds):
            x = np.copy(xdata)
            y = np.copy(ydata[cmp_ix])
            if plt_sec == "left":
                y = y[:n_samples_half]
                x = x[:n_samples_half]
                x -= elctrd_thk
            elif plt_sec == "right":
                ix_shift[plt_ix][cmp_ix] += n_samples_half
                y = y[n_samples_half:]
                x = x[n_samples_half:]
                x += elctrd_thk
                x -= box_z
            else:
                raise ValueError("Unknown plot section: '{}'".format(plt_sec))
            if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                # `scipy.signal.find_peaks` cannot handle NaN values.
                raise ValueError(
                    "Compound: {}\n"
                    "Plot section: {}\n"
                    "Encountered NaN values in free-energy"
                    " profile".format(cmp, plt_sec)
                )

            # Remove infinite values that stem from taking the
            # logarithm of zero when calculating the free-energy
            # profile from the density profile.
            valid = np.isfinite(y)
            n_samples_half_plt_sec = len(valid) // 2
            if not np.all(valid[:n_samples_half_plt_sec]):
                start = np.argmin(valid[n_samples_half_plt_sec - 1 :: -1])
                start = n_samples_half_plt_sec - start
            else:
                start = 0
            if not np.all(valid[n_samples_half_plt_sec:]):
                stop = np.argmin(valid[n_samples_half_plt_sec:])
                stop = n_samples_half_plt_sec + stop
            else:
                stop = len(x)
            x = x[start:stop]
            y = y[start:stop]
            ix_shift[plt_ix][cmp_ix] += start
            del valid

            # Prepare data for peak finding.
            # Replace outliers with interpolated values.
            y_original = np.copy(y)
            y = leap.misc.interp_outliers(
                x,
                y,
                inplace=True,
                threshold=outlier_thresholds[cmp_ix],
                width=(None, outlier_max_width),
                rel_height=outlier_rel_height,
            )
            # Smooth data with Savitzky-Golay filter.
            y = savgol_filter(
                y, window_length=wlen, polyorder=polyorder, mode="nearest"
            )

            # Find free-energy extrema.
            for fac_ix, fac in enumerate(peak_finding_factors):
                peaks_tmp, properties_tmp = find_peaks(
                    fac * y,
                    prominence=prominence,
                    width=(
                        min_width,  # Discard noise.
                        max_width,  # Discard broad peaks.
                    ),
                    rel_height=rel_height,
                )
                base_widths_tmp = peak_widths(
                    fac * y,
                    peaks_tmp,
                    rel_height=1,
                    prominence_data=(
                        properties_tmp["prominences"],
                        properties_tmp["left_bases"],
                        properties_tmp["right_bases"],
                    ),
                )
                # Pseudo Full Width at Half Maximum.  Pseudo, because
                # it's the width at half prominence and not at half of
                # the peak height.
                fwhm_tmp = peak_widths(
                    fac * y,
                    peaks_tmp,
                    rel_height=0.5,
                    prominence_data=(
                        properties_tmp["prominences"],
                        properties_tmp["left_bases"],
                        properties_tmp["right_bases"],
                    ),
                )
                peaks[plt_ix][cmp_ix][fac_ix] = peaks_tmp
                peak_heights[plt_ix][cmp_ix][fac_ix] = np.copy(y[peaks_tmp])
                properties[plt_ix][cmp_ix][fac_ix] = properties_tmp
                base_widths[plt_ix][cmp_ix][fac_ix] = base_widths_tmp
                fwhm[plt_ix][cmp_ix][fac_ix] = fwhm_tmp

            # Plot peak finding results.
            fig, ax = plt.subplots(clear=True)
            leap.plot.elctrd_left(ax)
            if plt_sec == "right":
                x *= -1  # Ensure positive distance values.
            lines = ax.plot(
                x, y_original, color="tab:blue", label="Free Energy"
            )
            ax.plot(
                x,
                y,
                color="tab:cyan",
                label="Smoothed",
                linewidth=lines[0].get_linewidth() / 2,
                alpha=leap.plot.ALPHA,
            )
            for fac_ix, fac in enumerate(peak_finding_factors):
                leap.plot.peaks(
                    ax,
                    x,
                    y,
                    peaks[plt_ix][cmp_ix][fac_ix],
                    properties[plt_ix][cmp_ix][fac_ix],
                    base_widths[plt_ix][cmp_ix][fac_ix],
                    peak_type="min" if fac == -1 else "max",
                    kwargs_scatter={
                        "label": "Extremum" if fac == -1 else None
                    },
                    kwargs_proms={
                        "label": "Prominence" if fac == -1 else None
                    },
                    kwargs_widths={
                        "label": r"Width at $%.2f$ Prom." % rel_height
                        if fac == -1
                        else None
                    },
                    kwargs_hlines={
                        "label": "Lowest Contour Line" if fac == -1 else None
                    },
                )

            if plt_sec == "left":
                xlim = (xmin, xmax)
            elif plt_sec == "right":
                xlim = (xmax, xmin)  # Reverse x-axis.
            else:
                raise ValueError("Unknown plot section: '{}'".format(plt_sec))
            ax.set(
                xlabel=r"Distance to Electrode / nm",
                ylabel=(
                    r"Free Energy $F_{"
                    + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp]
                    + r"}$ / $k_B T$"
                ),
                xlim=xlim,
            )

            # Equalize x- and y-ticks so that plots can be stacked
            # together.
            xlim_diff = np.diff(ax.get_xlim())
            if xlim_diff > 2.5 and xlim_diff < 5:
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.xaxis.set_minor_locator(MultipleLocator(0.2))
            ylim_diff = np.diff(ax.get_ylim())
            if ylim_diff > 10 and ylim_diff < 20:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if all(np.abs(ax.get_ylim()) < 10) and ylim_diff > 2:
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

            legend_title = (
                r"%.2f$ $e$/nm$^2$" % surfq
                + "\n"
                + r"$r = %.4f$" % Sim.Li_O_ratio
                + "\n"
                + r"$n_{EO} = %d$" % Sim.O_per_chain
            )
            if plt_sec == "left":
                if np.isclose(surfq, 0, rtol=0, atol=1e-6):
                    legend_title = r"$\sigma_s = " + legend_title
                else:
                    legend_title = r"$\sigma_s = +" + legend_title
                legend_loc = "right"
            elif plt_sec == "right":
                if np.isclose(surfq, 0, rtol=0, atol=1e-6):
                    legend_title = r"$\sigma_s = " + legend_title
                else:
                    legend_title = r"$\sigma_s = -" + legend_title
                legend_loc = "left"
            else:
                raise ValueError("Unknown plot section: '{}'".format(plt_sec))
            if abs(ax.get_ylim()[1]) > abs(ax.get_ylim()[0]):
                legend_loc = "upper " + legend_loc
            else:
                legend_loc = "lower " + legend_loc
            legend = ax.legend(
                title=legend_title,
                loc=legend_loc,
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend.get_title().set_multialignment("center")

            pdf.savefig()
            plt.close()

    # Combine extrema that were found in the left and right section of
    # the free-energy profile for each compound.
    peaks_full = [[[] for fac in peak_finding_factors] for cmp in compounds]
    peak_heights_full = deepcopy(peaks_full)
    for cmp_ix in range(len(compounds)):
        for fac_ix in range(len(peak_finding_factors)):
            for plt_ix in range(len(plot_sections)):
                pks = peaks[plt_ix][cmp_ix][fac_ix] + ix_shift[plt_ix][cmp_ix]
                peaks_full[cmp_ix][fac_ix].append(pks)
                pks_h = peak_heights[plt_ix][cmp_ix][fac_ix]
                peak_heights_full[cmp_ix][fac_ix].append(pks_h)
            peaks_full[cmp_ix][fac_ix] = np.concatenate(
                peaks_full[cmp_ix][fac_ix]
            )
            peak_heights_full[cmp_ix][fac_ix] = np.concatenate(
                peak_heights_full[cmp_ix][fac_ix]
            )

    properties_full = [
        [
            {key: [] for key in properties[0][0][0].keys()}
            for fac in peak_finding_factors
        ]
        for cmp in compounds
    ]
    for cmp_ix in range(len(compounds)):
        for fac_ix in range(len(peak_finding_factors)):
            for plt_ix in range(len(plot_sections)):
                for key, val in properties[plt_ix][cmp_ix][fac_ix].items():
                    if key in (
                        "left_bases",
                        "right_bases",
                        "left_ips",
                        "right_ips",
                        "left_edges",
                        "right_edges",
                    ):
                        props = val + ix_shift[plt_ix][cmp_ix]
                    else:
                        props = val
                    properties_full[cmp_ix][fac_ix][key].append(props)
            for key, val in properties_full[cmp_ix][fac_ix].items():
                properties_full[cmp_ix][fac_ix][key] = np.concatenate(val)

    base_widths_full = [
        [[[] for bw in base_widths[0][0][0]] for fac in peak_finding_factors]
        for cmp in compounds
    ]
    fwhm_full = deepcopy(base_widths_full)
    for cmp_ix in range(len(compounds)):
        for fac_ix in range(len(peak_finding_factors)):
            for plt_ix in range(len(plot_sections)):
                for w_ix in range(len(base_widths[plt_ix][cmp_ix][fac_ix])):
                    bw = base_widths[plt_ix][cmp_ix][fac_ix][w_ix]
                    fw = fwhm[plt_ix][cmp_ix][fac_ix][w_ix]
                    if w_ix >= 2 and w_ix <= 3:
                        bw = bw + ix_shift[plt_ix][cmp_ix]
                        fw = fw + ix_shift[plt_ix][cmp_ix]
                    base_widths_full[cmp_ix][fac_ix][w_ix].append(bw)
                    fwhm_full[cmp_ix][fac_ix][w_ix].append(fw)
            for w_ix in range(len(base_widths_full[cmp_ix][fac_ix])):
                base_widths_full[cmp_ix][fac_ix][w_ix] = np.concatenate(
                    base_widths_full[cmp_ix][fac_ix][w_ix]
                )
                fwhm_full[cmp_ix][fac_ix][w_ix] = np.concatenate(
                    fwhm_full[cmp_ix][fac_ix][w_ix]
                )

    # Plot the full free-energy profile.
    x = xdata
    for cmp_ix, cmp in enumerate(compounds):
        y = ydata[cmp_ix]
        fig, ax = plt.subplots(clear=True)
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
        lines = ax.plot(x, y, color="tab:blue", label="Free Energy")
        for fac_ix, fac in enumerate(peak_finding_factors):
            leap.plot.peaks(
                ax,
                x,
                y,
                peaks_full[cmp_ix][fac_ix],
                properties_full[cmp_ix][fac_ix],
                base_widths_full[cmp_ix][fac_ix],
                peak_type="min" if fac == -1 else "max",
                kwargs_scatter={"label": "Extremum" if fac == -1 else None},
                kwargs_proms={"label": "Prominence" if fac == -1 else None},
                kwargs_widths={
                    "label": r"Width at $%.2f$ Prom." % rel_height
                    if fac == -1
                    else None
                },
                kwargs_hlines={
                    "label": "Lowest Contour Line" if fac == -1 else None
                },
            )

        ax.set(
            xlabel=r"$z$ / nm",
            ylabel=(
                r"Free Energy $F_{"
                + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp]
                + r"}$ / $k_B T$"
            ),
            xlim=(0, box_z),
        )

        # Equalize x- and y-ticks so that plots can be stacked together.
        xlim_diff = np.diff(ax.get_xlim())
        if xlim_diff > 2.5 and xlim_diff < 5:
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ylim_diff = np.diff(ax.get_ylim())
        if ylim_diff > 10 and ylim_diff < 20:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if all(np.abs(ax.get_ylim()) < 10) and ylim_diff > 2:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        legend_title = (
            r"%.2f$ $e$/nm$^2$" % surfq
            + "\n"
            + r"$r = %.4f$" % Sim.Li_O_ratio
            + "\n"
            + r"$n_{EO} = %d$" % Sim.O_per_chain
        )
        if np.isclose(surfq, 0, rtol=0, atol=1e-6):
            legend_title = r"$\sigma_s = " + legend_title
        else:
            legend_title = r"$\sigma_s = \pm" + legend_title
        if abs(ax.get_ylim()[1]) > abs(ax.get_ylim()[0]):
            legend_loc = "upper center"
        else:
            legend_loc = "lower center"
        legend = ax.legend(
            title=legend_title,
            loc=legend_loc,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")

        pdf.savefig()
        plt.close()
print("Created {}".format(outfile_pdf))

# Write peak properties to file.
print("Creating output file(s)...")
x = xdata
ndx = np.arange(len(x))
for cmp_ix, cmp in enumerate(compounds):
    for fac_ix, fac in enumerate(peak_finding_factors):
        peak_type = "minima" if fac == -1 else "maxima"
        outfile = outfile_base + peak_type + "_" + cmp + ".txt.gz"
        header = (
            "Free-energy {:s}.\n".format(peak_type)
            + "\n"
            + "System:        {:s}\n".format(args.system)
            + "Settings:      {:s}\n".format(args.settings)
            + "Input file:    {:s}\n".format(infile)
            + "Read Columns:  {:d}, {:d}\n".format(
                cols[0] + 1, cols[1:][cmp_ix] + 1
            )
            + "\n"
            + "Compound:                      {:s}\n".format(cmp)
            + "Surface charge:                {:.2f} e/nm^2\n".format(surfq)
            + "Lithium-to-ether-oxygen ratio: {:.4f}\n".format(Sim.Li_O_ratio)
            + "Ether oxygens per PEO chain:   {:d}\n".format(Sim.O_per_chain)
            + "\n"
            + "\n"
            + "The free-energy profile F(z) is calculated from the density\n"
            + "profile p(z) according to\n"
            + "  F(z)/kT = -ln[p(z)/p0]\n"
            + "where k is the Boltzmann constant and T is the temperature.\n"
            + "p0 is the average density in the bulk region.\n"
            + "\n"
            + "Afterwards, outliers are identified by comparing each F(z)\n"
            + "value to its two neighboring values.  If the value sticks out\n"
            + "by more than {:>.2f} times the standard deviation of\n".format(
                sd_factor
            )
            + "all F(z) values in the bulk region, it is regarded as\n"
            + "outlier.  The identified outliers are replaced by a linear\n"
            + "interpolation between their two neighboring values.\n"
            + "\n"
            + "Subsequently, the free-energy profile is smoothed using a\n"
            + "Savitzky-Golay filter.\n"
            + "\n"
            + "Finally, free-energy {:s} are identified on the basis\n".format(
                peak_type
            )
            + "of their peak prominence and width.\n"
            + "\n"
            + "\n"
            + "(Average) bin width and number of sample points per nm:\n"
            + "x_sample_spacing: {:>16.9e} nm\n".format(x_sample_spacing)
            + "n_samples_per_nm:  {:d}\n".format(n_samples_per_nm)
            + "\n"
            + "Start of the bulk region as distance to the electrodes:\n"
            + "bulk_start:  {:>.2f} nm\n".format(Elctrd.BULK_START / 10)
            + "Bulk region in absolute coordinates:\n"
            + "z_bulk_min: {:>16.9e} nm\n".format(bulk_region[0])
            + "z_bulk_max: {:>16.9e} nm\n".format(bulk_region[1])
            + "Box length in z direction in nm\n"
            + "box_z:      {:>16.9e} nm\n".format(box_z)
            + "\n"
            + "Parameters for finding outliers with scipy.signal.find_peaks\n"
            + "Standard deviation: {:>16.9e} kT\n".format(
                outlier_thresholds[cmp_ix] / sd_factor
            )
            + "sd_factor:           {:>.2f}\n".format(sd_factor)
            + "outlier_threshold:  {:>16.9e} kT\n".format(
                outlier_thresholds[cmp_ix]
            )
            + "outlier_max_width:   {:d} sample points\n".format(
                outlier_max_width
            )
            + "outlier_rel_height:  {:>.2f}\n".format(outlier_rel_height)
            + "\n"
            + "Parameters for data smoothing with scipy.signal.savgol_filter\n"
            + "polyorder:      {:d}\n".format(polyorder)
            + "window_length:  {:d} sample points\n".format(wlen)
            + "\n"
            + "Parameters for peak finding with scipy.signal.find_peaks\n"
            + "prominence:    {:>.2f} kT\n".format(prominence)
            + "min_width:     {:d} sample points\n".format(min_width)
            + "max_width_nm:  {:>.2f} nm\n".format(max_width_nm)
            + "max_width:     {:d} sample points\n".format(max_width)
            + "rel_height:    {:>.2f}\n".format(rel_height)
            + "\n"
            + "The columns contain:\n"
            + "(Direct return values from `scipy.signal.find_peaks` and from\n"
            + "`scipy.signal.peak_widths` are indicated by `backticks`)\n"
            + "\n"
            + "   1 `peaks`: The indices of the peak positions\n"
            + "   2 Peak positions in nm\n"
            + "   3 Peak heights in kT (i.e. free-energy values at the peak\n"
            + "     positions)\n"
            + "\n"
            + "   4 `prominences`: Peak prominences in kT\n"
            + "   5 `left_bases`: The indices of the peaks' left bases\n"
            + "   6 `right_bases`: The indices of the peaks' right bases\n"
            + "     (The higher base is a peak's lowest contour line)\n"
            + "   7 The peaks' left bases in nm\n"
            + "   8 The peaks' right bases in nm\n"
            + "   9 The peaks' base width in sample points\n"
            + "  10 The peaks' base width in nm\n"
            + "\n"
            + "  Peak widths at {:.0f} % of the peak prominence:\n".format(
                rel_height * 100
            )
            + "  (This width was used for peak identification using the\n"
            + "  parameters given above)\n"
            + "  11 `widths`: The width of each peak in sample points\n"
            + "  12 The width of each peak in nm\n"
            + "  13 `width_heights`: The heights (i.e. the free-energy\n"
            + "     values) at which the peak widths were measured in kT\n"
            + "  14 `left_ips`: Linearly interpolated positions in sample\n"
            + "     points of the left intersection points of a horizontal\n"
            + "     line at the respective evaluation height with the\n"
            + "     free-energy profile\n"
            + "  15 `right_ips`: Linearly interpolated positions of the\n"
            + "     right intersection points\n"
            + "  16 Left intersection points in nm\n"
            + "  17 Right intersection points in nm\n"
            + "\n"
            + "  Peak widths at 50 % of the peak prominence:\n"
            + "  18 `widths`: The width of each peak in sample points\n"
            + "  19 The width of each peak in nm\n"
            + "  20 `width_heights`: Heights for width measuring in kT\n"
            + "  21 `left_ips`: Left intersection points in sample points\n"
            + "  22 `right_ips`: Right intersection points in sample points\n"
            + "  23 Left intersection points in nm\n"
            + "  24 Right intersection points in nm\n"
            + "\n"
            + "  Peak widths at 100 % of the peak prominence:\n"
            + "  (i.e. the widths of the peaks' lowest contour lines):\n"
            + "  25 `widths`: The width of each peak in sample points\n"
            + "  26 The width of each peak in nm\n"
            + "  27 `width_heights`: Heights for width measuring in kT\n"
            + "  28 `left_ips`: Left intersection points in sample points\n"
            + "  29 `right_ips`: Right intersection points in sample points\n"
            + "  30 Left intersection points in nm\n"
            + "  31 Right intersection points in nm\n"
            + "\n"
            + "Column numbers:\n"
            + "{:>14d}".format(1)
        )
        pks = peaks_full[cmp_ix][fac_ix]
        pks_h = peak_heights_full[cmp_ix][fac_ix]
        props = properties_full[cmp_ix][fac_ix]
        bw = base_widths_full[cmp_ix][fac_ix]
        fwhm = fwhm_full[cmp_ix][fac_ix]

        if np.any(x[pks] <= elctrd_thk):
            raise ValueError(
                "At least one peak lies within the left electrode.  Peak"
                " positions: {}.  Left electrode:"
                " {}".format(x[pks], elctrd_thk)
            )
        if np.any(x[pks] >= box_z - elctrd_thk):
            raise ValueError(
                "At least one peak lies within the right electrode.  Peak"
                " positions: {}.  Right electrode:"
                " {}".format(x[pks], box_z - elctrd_thk)
            )
        if np.any((x[pks] >= bulk_region[0]) & (x[pks] <= bulk_region[1])):
            warnings.warn(
                "At least one peak lies within the bulk region.  Peak"
                " positions: {}.  Bulk region:"
                " {}".format(x[pks], bulk_region),
                RuntimeWarning,
                stacklevel=2,
            )
        if surfq == 0:
            pk_is_left = x[pks] <= (box_z / 2)
            n_pks_left = np.count_nonzero(pk_is_left)
            n_pks_right = len(pk_is_left) - n_pks_left
            if n_pks_left != n_pks_right:
                warnings.warn(
                    "The surface charge is {} e/nm^2 but the number of left"
                    " ({}) and right free-energy {} ({}) do not match for"
                    " compound {}.".format(
                        surfq, n_pks_left, peak_type, n_pks_right, cmp
                    ),
                    UserWarning,
                    stacklevel=2,
                )

        data = np.column_stack([pks, x[pks], pks_h])

        left_bases_nm = x[props["left_bases"]]
        right_bases_nm = x[props["right_bases"]]
        widths_bases = props["right_bases"] - props["left_bases"]
        widths_bases_nm = right_bases_nm - left_bases_nm
        left_ips_nm = np.interp(props["left_ips"], ndx, x)
        right_ips_nm = np.interp(props["right_ips"], ndx, x)
        widths_nm = right_ips_nm - left_ips_nm
        for key, val in props.items():
            if key == "width_heights":
                val = fac * val
            data = np.column_stack((data, val))
            if key == "right_bases":
                data = np.column_stack(
                    [
                        data,
                        left_bases_nm,
                        right_bases_nm,
                        widths_bases,
                        widths_bases_nm,
                    ]
                )
            if key == "widths":
                data = np.column_stack([data, widths_nm])
            if key == "right_ips":
                data = np.column_stack([data, left_ips_nm, right_ips_nm])

        for wth in (fwhm, bw):
            left_ips_nm = np.interp(wth[2], ndx, x)
            right_ips_nm = np.interp(wth[3], ndx, x)
            widths_nm = right_ips_nm - left_ips_nm
            for w_ix, w in enumerate(wth):
                if w_ix == 1:  # "width_heights"
                    w = fac * w
                data = np.column_stack((data, w))
                if w_ix == 0:  # "widths"
                    data = np.column_stack([data, widths_nm])
                elif w_ix == 3:  # "right_ips"
                    data = np.column_stack([data, left_ips_nm, right_ips_nm])

        for i in range(data.shape[-1] - 1):
            header += " {:>16d}".format(i + 2)

        leap.io_handler.savetxt(outfile, data, header=header)
        print("Created {}".format(outfile))

print("Done")
