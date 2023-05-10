#!/usr/bin/env python3


"""
Extract the positions of extrema of free-energy profiles for different
compounds of a single simulation.
"""


# Standard libraries
import argparse
from copy import deepcopy

# Third-party libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
from scipy.signal import find_peaks, peak_widths, savgol_filter

# First-party libraries
import lintf2_ether_ana_postproc as leap


def get_peaks(y, prominence, **kwargs):
    """
    Find peaks in a signal based on peak properties.

    This is a wrapper function around :func:`scipy.signal.find_peaks`
    and :func:`scipy.signal.peak_widths`.

    Parameters
    ----------
    y : array_like
        A signal with peaks.
    prominence :
        See :func:`scipy.signal.find_peaks`.
    kwargs : dict, optional
        Additional keyword arguments to parse to
        :func:`scipy.signal.find_peaks` besides `prominence`.

    Returns
    -------
    peaks, properties : numpy.ndarray, dict of numpy.ndarrays
        See :func:`scipy.signal.find_peaks`.
    base_widths : numpy.ndarray
        The width of each peak at the peak's full prominence, i.e. the
        width at the lowest contour line.  `base_widths` is the output
        of :func:`scipy.signal.peak_widths` converted to a 2-dimensional
        array by parsing it to :func:`numpy.array`.
    fwhm : numpy.ndarray
        Pseudo Full Width at Half Maximum for each peak.  "Pseudo",
        because it is actually the width at the peak's half prominence
        and not at the peak's half height.  `fwhm` is the output of
        :func:`scipy.signal.peak_widths` converted to a 2-dimensional
        array by parsing it to :func:`numpy.array`.
    """
    peaks, properties = find_peaks(y, prominence=prominence, **kwargs)
    base_widths = peak_widths(
        y,
        peaks,
        rel_height=1,
        prominence_data=(
            properties["prominences"],
            properties["left_bases"],
            properties["right_bases"],
        ),
    )
    fwhm = peak_widths(
        y,
        peaks,
        rel_height=0.5,
        prominence_data=(
            properties["prominences"],
            properties["left_bases"],
            properties["right_bases"],
        ),
    )
    return peaks, properties, np.array(base_widths), np.array(fwhm)


def ensure_last_max(
    y,
    minima,
    maxima,
    prominences,
    cutoff,
    max_gap=None,
    reverse=False,
    **kwargs,
):
    """
    Ensure that the last extremum in `y` is a maximum.

    Parameters
    ----------
    y : array_like
        A signal with peaks.
    minima, maxima : array_like, array_like
        Array of indices indicating minima/maxima in `y`.
    prominences : array_like
        Array of prominences.  If the last found extremum in `y` is a
        minimum, search for a maximum after the last minimum by
        successively varying the prominence threshold according to the
        given prominences.  Usually, `prominences` should be an array of
        decreasing prominence thresholds and the highest threshold
        should be lower than the one used to find the initial minima and
        maxima.  If multiple maxima are found for a given prominence
        threshold, the maximum closest to the last minimum is chosen.
    cutoff : scalar
        If no maximum is found after the last minimum with a prominence
        higher than ``np.min(prominences)``, the last "maximum" is set
        to the point where `y` reaches the first time `cutoff` after the
        last minimum.
    max_gap : int or None, optional
        Greatest allowed gap between the last minimum and the found
        maximum in sample points.  If provided and the found maximum is
        further away from the last maximum than `max_gap`, the last
        "maximum" is set to the position of the last minimum plus
        `max_gap`.
    reverse : bool, optional
        If ``True``, reverse `y`.  This corresponds to using
        ``y[::-1]``, ``len(y) - minima[::-1]`` and
        ``len(y) - maxima[::-1]`` for finding the last maximum.  Note
        that the return values of this function (like the index of the
        last maximum) always refer to the original input array `y`.
    kwargs : dict, optional
        Additional keyword arguments to parse to
        :func:`get_fre-energy_extrema.get_peaks` besides `prominence`.

    Returns
    -------
    peaks, properties : int, dict of numpy.ndarrays
        The index and the properties of the last maximum.  See
        :func:`get_fre-energy_extrema.get_peaks`.
    base_widths : numpy.ndarray
        The width of the last maximum at its full prominence, i.e. the
        width at the lowest contour line.  See
        :func:`get_fre-energy_extrema.get_peaks`.
    fwhm : numpy.ndarray
        Pseudo Full Width at Half Maximum of the last maximum.  See
        :func:`get_fre-energy_extrema.get_peaks`.

    If the last extremum in `y` is already a maximum, all return values
    are ``None``.

    Notes
    -----
    This function was originally written to ensure that the layering
    region and the bulk region of a free-energy profile are separated by
    a free-energy maximum.

    Sometimes, the layering region ends with a minimum.  This means,
    when going from the electrode to the bulk region, the last found
    peak is a minimum.  This can happen, when the difference of the
    free-energy values at the last found minimum and the next
    (not-found) maximum is greater than the required prominence, but the
    difference of the free-energy values at this not-found maximum and
    the next not-found minimum is less than this threshold.  Hence, the
    prominence of the not-found maximum is less than the required
    prominence threshold which is the reason why this maximum is not
    found.  However, a particle traveling from the electrode to the bulk
    region has to overcome this not-found maximum which in the direction
    of travel is higher than the required prominence.  Thus, this
    not-found maximum should actually be found.  In short, the layering
    region must not end with a minimum, because then there exists a
    free-energy barrier that a particle has to overcome when traveling
    from the electrode to the bulk that is higher than the prominence
    threshold.
    """
    y = np.array(y, copy=True)
    minima, maxima = np.asarray(minima), np.asarray(maxima)
    prominences = np.asarray(prominences)

    if reverse:
        ix_last_min = minima[0]
        if len(maxima) > 0:
            ix_last_max = maxima[0]
        else:
            ix_last_max = len(y)
        if ix_last_min > ix_last_max:
            # The last extremum in `y` is already a maximum.
            return None, None, None, None
    else:
        ix_last_min = minima[-1]
        if len(maxima) > 0:
            ix_last_max = maxima[-1]
        else:
            ix_last_max = -1
        if ix_last_min < ix_last_max:
            # The last extremum in `y` is already a maximum.
            return None, None, None, None
    if ix_last_min == ix_last_max:
        raise ValueError(
            "The last minimum ({}) and the last maximum ({}) have the same"
            " index".format(ix_last_min, ix_last_max)
        )

    # Only the region after the last minimum is of interest.  However,
    # to get the correct indices for the peak positions, their bases,
    # intersection points, etc., we cannot simply discard everything
    # before the last minimum.  This would make things more complicate,
    # because we would need to know which return values of
    # :func:`get_free-energy_extrema.get_peaks` would actually be
    # affected by truncating the input array and which would not.
    # Instead, we set every value of `y` before the last minimum to the
    # value at the last minimum so that we cannot find a maximum there.
    # For the same reason, we don't simply reverse `y` by using
    # ``y[::-1]`` if `reverse` is ``True``.
    if reverse:
        y[ix_last_min + 1 :] = y[ix_last_min]
    else:
        y[:ix_last_min] = y[ix_last_min]

    # Search for maxima with a prominence of at least
    # ``np.min(prom_min)``.
    found_max = False
    for prom in prominences:
        pks, prop, bw, fwhm = get_peaks(y, prom, **kwargs)
        if len(pks) > 0:
            found_max = True
            if (reverse and np.any(pks >= ix_last_min)) or (
                not reverse and np.any(pks <= ix_last_min)
            ):
                raise ValueError(
                    "Found a maximum before the last minimum.  This should not"
                    " have happened."
                )
            if reverse:
                ix = -1
                start, stop = -1, None
            else:
                ix = 0
                start, stop = 0, 1
            pks = pks[ix]
            for key, val in prop.items():
                prop[key] = val[ix]
            bw = mdt.nph.take(bw, start=start, stop=stop, axis=-1)
            fwhm = mdt.nph.take(fwhm, start=start, stop=stop, axis=-1)
            break

    if not found_max:
        # Set the last "maximum" to the point where `y` reaches the
        # first time `cutoff` after the last minimum.
        if reverse:
            y_after_last_min = y[:ix_last_min]
            y_after_last_min = y_after_last_min[::-1]
        else:
            y_after_last_min = y[ix_last_min + 1 :]
        pks = np.argmax(y_after_last_min >= cutoff)
        if reverse:
            pks = len(y_after_last_min) - pks
        else:
            pks += ix_last_min + 1

    max_gap_exceeded = False
    if max_gap is not None and abs(pks - ix_last_min) > max_gap:
        # Set the last "maximum" to the position of the last minimum
        # plus `max_gap`.
        max_gap_exceeded = True
        if reverse:
            pks = ix_last_min - max_gap
        else:
            pks = ix_last_min + max_gap
        if pks < 0 or pks >= len(y):
            raise ValueError(
                "`max_gap` ({}) is too large for an input array of length {}"
                " and the last minimum being at position {}.  This results in"
                " the last maximum being at position {}, which is outside the"
                " input array".format(max_gap, len(y), ix_last_min, pks)
            )

    if not found_max or max_gap_exceeded:
        # Set all peak properties to the peak position.
        for key in prop.keys():
            if key == "peak_heights":
                prop[key] = np.array(y[pks])
            elif key == "left_thresholds":
                try:
                    prop[key] = np.array(y[pks - 1])
                except IndexError:
                    prop[key] = np.array(y[pks])
            elif key == "right_thresholds":
                try:
                    prop[key] = np.array(y[pks + 1])
                except IndexError:
                    prop[key] = np.array(y[pks])
            elif key in ("prominences", "widths", "plateau_sizes"):
                prop[key] = np.array([0])
            elif key == "width_heights":
                prop[key] = np.array([kwargs.pop("rel_height", 0.0)])
            elif key in (
                "right_bases",
                "left_bases",
                "left_ips",
                "right_ips",
                "left_edges",
                "right_edges",
            ):
                prop[key] = np.array([pks])
            else:
                raise KeyError("Unknown key: '{}'".format(key))
        bw = np.zeros((4, 1))
        bw[0, 0] = 0  # widths
        bw[1, 0] = 1.0  # width_heights
        bw[2, 0] = 0  # left_ips
        bw[3, 0] = 0  # right_ips
        fwhm = np.copy(bw)
        fwhm[1, 0] = 0.5  # width_heights

    return pks, prop, bw, fwhm


def insert_into_arr_in_dct(dct1, ix, dct2):
    """
    Insert values into the arrays in a dictionary.

    Parameters
    ----------
    dct1 : dict
        Dictionary containing the arrays into which the arrays of `dct2`
        should be inserted.  `dct1` is changed inplace but the arrays
        are copied.
    ix : int
        The index of the arrays of `dct1` before which to insert the
        arrays of `dct2`.
    dct2 : dict
        Dictionary containing the arrays that should be inserted into
        the arrays of `dct1`.  `dct2` must contain all keys of `dct1`.

    Returns
    -------
    dct1 : dict
        The input dictionary with appended arrays.

    Notes
    -----
    This function was originally written to insert values into the
    arrays in a dictionary of peak properties as returned by
    :func:`scipy.signal.find_peaks`.
    """
    if not all(key1 in dct2.keys() for key1 in dct1.keys()):
        raise ValueError("`dct2` must contain all keys of `dct1`")

    for key1, val1 in dct1.items():
        dct1[key1] = np.insert(val1, ix, dct2[key1])

    return dct1


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

# Factor for multiplying the standard deviation of the free-energy
# profile in the bulk region for finding outliers and for finding the
# last maximum that separates the layering and bulk region.
sd_factor = 4
# In the interval [mean-4*sd, mean+4*sd] lie 99.993666 % of all values
# of a normally distributed random variable.
# See https://de.wikipedia.org/wiki/Normalverteilung#Streuintervalle

# Parameters for peak finding with `scipy.signal.find_peaks`.
# Vertical properties are given in data units, horizontal properties
# are given in number of sample points if not specified otherwise.
# The prominence is chosen such that 100*`prob_thresh` percent of the
# particles have a higher kinetic energy in the z direction than the
# prominence.
prob_thresh = 0.5
prominence = leap.misc.e_kin(prob_thresh)  # Prominence in kT.
# Another possible way to choose the prominence is:
# <T> = 1/2 kT (Average kinetic energy from the equipartition theorem)
# <T^2> = (1/2 + (1/2)^2) * (kT)^2
# sigma_T^2 = <T^2> - <T>^2 = 1/2 (kT)^2
# sigma_T = 1/sqrt(2) kT
# `prominence` / kT = <T> - 1/2 sigma_T = 1/2 - 1/2 * 1/sqrt(2) = 0.1464
# See pages 122-125 of my "derivation book".
min_width = 2  # Required minimum width at `rel_height` in sample points
max_width_nm = 2  # Maximum width at `rel_height` in nm.
rel_height = 0.2  # Relative height at which the peak width is measured.

# Parameters for finding the last maximum in case the layering and bulk
# region are separated by a minium.
# See :func:`get_free-energy_extrema.ensure_last_max`.
prob_thresh_max = 0.9
prob_thresh_step_size = 0.1
prob_thresholds = np.arange(
    prob_thresh,
    prob_thresh_max + prob_thresh_step_size / 2,
    prob_thresh_step_size,
)
# Discard the first prominence threshold, because this is already used
# for the initial peak finding.
prob_thresholds = prob_thresholds[1:]
prominences_last_max = np.asarray(leap.misc.e_kin(prob_thresholds))
# height_min = 0 - outlier_threshold.
plateau_size_max = 2
# cutoff = height_min.
max_gap_nm = 1  # Allowed gap between the last minimum and maximum in nm

# Parameters for finding outliers with `scipy.signal.find_peaks`.
# outlier_threshold = sd_factor * bulk_sd.
outlier_max_width = min_width
outlier_rel_height = rel_height

# Parameters for data smoothing with a Savitzky-Golay filter.
# The advantage of a Savitzky-Golay filter compared to a moving average
# is that it preserves relative extrema while the moving average
# flattens them.  Note that a Savitzky-Golay filter with a polynomial
# order of zero is identical to a moving average.
polyorder = 3  # Order or the polynomial used to fit the samples.
wlen = 15  # Length of the filter window.


print("Creating Simulation instance(s)...")
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    top_path = "q%g" % surfq
else:
    surfq = None
    top_path = "bulk"
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, top_path)


print("Reading data...")
file_suffix = analysis + analysis_suffix + ".xvg.gz"
infile = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)

data = np.loadtxt(infile, comments=["#", "@"], usecols=cols, unpack=True)
xdata, ydata = data[0], data[1:]

# Average the density profile in the left and right half of the
# simulation box for bulk simulations and surface simulations with
# uncharged surfaces.
box_z = Sim.box[2] / 10  # A -> nm
if Sim.surfq is None or Sim.surfq == 0:
    xdata_bfr_sym = np.copy(xdata)  # Data before before symmetrization.
    ydata_bfr_sym = np.copy(ydata)
    xdata, ydata = leap.misc.symmetrize_data(
        xdata, ydata, x2shift=box_z, reassemble=True
    )
    symmetrized = True
else:
    symmetrized = False

# Calculate free-energy profiles from density profiles.
bulk_region = Sim.bulk_region / 10  # A -> nm
for i, y in enumerate(ydata):
    ydata[i] = leap.misc.dens2free_energy(xdata, y, bulk_region=bulk_region)
    if symmetrized:
        ydata_bfr_sym[i] = leap.misc.dens2free_energy(
            xdata_bfr_sym, ydata_bfr_sym[i], bulk_region=bulk_region
        )

# Maximum peak width for peak finding with `scipy.signal.find_peaks`.
x_sample_spacing = np.mean(np.diff(xdata))
n_samples_per_nm = round(1 / x_sample_spacing)
max_width = int(max_width_nm * n_samples_per_nm)
width = (min_width, max_width)  # Discard noise and broad peaks.
# Maximum allowed gap between the last minimum and the last maximum.
max_gap = int(max_gap_nm * n_samples_per_nm)

# Calculate the threshold for outlier detection from the standard
# deviation of the free-energy profile in the bulk region.
bulk_begin_ix, bulk_end_ix = leap.misc.find_nearest(xdata, bulk_region)
bulk_sds = np.nanstd(ydata[:, bulk_begin_ix:bulk_end_ix], axis=-1, ddof=1)
outlier_thresholds = sd_factor * bulk_sds
# Cutoff and minimal required peak height for finding the last maximum.
height_mins = 0 - outlier_thresholds
cutoffs = height_mins


print("Finding extrema and creating plot(s)...")
Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm
# Increase the layering region for peak finding by a small buffer.
bulk_begin_ix += 2
bulk_end_ix -= 2

plot_sections = ("left", "right")
xmin = 0
xmax = Elctrd.BULK_START / 10  # A -> nm
free_en_kwargs = {"label": "Free Energy", "color": "tab:blue"}

peak_finding_factors = (-1, 1)  # -1 for finding minima, 1 for maxima.
minima_ix = peak_finding_factors.index(-1)
maxima_ix = peak_finding_factors.index(1)

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
                y = y[:bulk_begin_ix]
                x = x[:bulk_begin_ix]
                x -= elctrd_thk
                if symmetrized:
                    # Only required for plotting the original data.
                    y_bfr_sym = np.copy(ydata_bfr_sym[cmp_ix][:bulk_begin_ix])
                    x_bfr_sym = np.copy(xdata_bfr_sym[:bulk_begin_ix])
                    x_bfr_sym -= elctrd_thk
            elif plt_sec == "right":
                ix_shift[plt_ix][cmp_ix] += bulk_end_ix
                y = y[bulk_end_ix:]
                x = x[bulk_end_ix:]
                x += elctrd_thk
                x -= box_z
                # Don't multiply `x` with -1 here, because then the
                # interpolation of outliers does not work.
                if symmetrized:
                    # Only required for plotting the original data.
                    y_bfr_sym = np.copy(ydata_bfr_sym[cmp_ix][bulk_end_ix:])
                    x_bfr_sym = np.copy(xdata_bfr_sym[bulk_end_ix:])
                    x_bfr_sym += elctrd_thk
                    x_bfr_sym -= box_z
                    x_bfr_sym *= -1  # Ensure positive distance values.
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

            # Remove infinite values that stem from taking the logarithm
            # of zero when calculating the free-energy profile from the
            # density profile.
            valid = np.isfinite(y)
            if np.all(valid):
                start = 0
                stop = len(y)
            else:
                # Remove infinite values at the data edges.
                start = np.argmax(valid)
                stop = len(y) - np.argmax(valid[::-1])
            x = x[start:stop]
            y = y[start:stop]
            ix_shift[plt_ix][cmp_ix] += start
            invalid = ~valid[start:stop]
            if np.any(invalid):
                # Interpolate intermediate infinite values.
                y = leap.misc.interp_invalid(x, y, invalid)
            del valid, invalid

            # Prepare data for peak finding.
            # Replace outliers with interpolated values.
            y_bfr_smooth = np.copy(y)  # y data before smoothing.
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
                pks_tmp, prop_tmp, bw_tmp, fwhm_tmp = get_peaks(
                    fac * y,
                    prominence=prominence,
                    # Use same arguments as for finding the last maximum
                    width=width,
                    rel_height=rel_height,
                )
                peaks[plt_ix][cmp_ix][fac_ix] = pks_tmp
                peak_heights[plt_ix][cmp_ix][fac_ix] = np.copy(y[pks_tmp])
                properties[plt_ix][cmp_ix][fac_ix] = prop_tmp
                base_widths[plt_ix][cmp_ix][fac_ix] = bw_tmp
                fwhm[plt_ix][cmp_ix][fac_ix] = fwhm_tmp

            # Ensure that the layering region and the bulk region are
            # separated by a free-energy maximum.
            pks_tmp, prop_tmp, bw_tmp, fwhm_tmp = ensure_last_max(
                y,
                minima=peaks[plt_ix][cmp_ix][minima_ix],
                maxima=peaks[plt_ix][cmp_ix][maxima_ix],
                prominences=prominences_last_max,
                cutoff=cutoffs[cmp_ix],
                max_gap=max_gap,
                reverse=(plt_sec == "right"),
                # Use same arguments as for peak finding.
                width=width,
                rel_height=rel_height,
                # Additional arguments.
                height=height_mins[cmp_ix],
                plateau_size=(None, plateau_size_max),
            )
            if pks_tmp is not None:
                if plt_sec == "left":
                    insert_ix = len(peaks[plt_ix][cmp_ix][maxima_ix])
                elif plt_sec == "right":
                    insert_ix = 0
                else:
                    raise ValueError(
                        "Unknown plot section: '{}'".format(plt_sec)
                    )
                peaks[plt_ix][cmp_ix][maxima_ix] = np.insert(
                    peaks[plt_ix][cmp_ix][maxima_ix], insert_ix, pks_tmp
                )
                peak_heights[plt_ix][cmp_ix][maxima_ix] = np.insert(
                    peak_heights[plt_ix][cmp_ix][maxima_ix],
                    insert_ix,
                    y[pks_tmp],
                )
                properties[plt_ix][cmp_ix][maxima_ix] = insert_into_arr_in_dct(
                    properties[plt_ix][cmp_ix][maxima_ix], insert_ix, prop_tmp
                )
                base_widths[plt_ix][cmp_ix][maxima_ix] = np.insert(
                    base_widths[plt_ix][cmp_ix][maxima_ix],
                    insert_ix,
                    np.squeeze(bw_tmp),
                    axis=-1,
                )
                fwhm[plt_ix][cmp_ix][maxima_ix] = np.insert(
                    fwhm[plt_ix][cmp_ix][maxima_ix],
                    insert_ix,
                    np.squeeze(fwhm_tmp),
                    axis=-1,
                )

            # Plot peak finding results.
            fig, ax = plt.subplots(clear=True)
            if surfq is not None:
                leap.plot.elctrd_left(ax)
            if plt_sec == "right":
                x *= -1  # Ensure positive distance values.
            if symmetrized:
                lines = ax.plot(
                    x_bfr_sym,
                    y_bfr_sym,
                    linewidth=1.5 * mpl.rcParams["lines.linewidth"],
                    **free_en_kwargs,
                )
                lines = ax.plot(
                    x,
                    y_bfr_smooth,
                    color="dodgerblue",
                    label="Symmetrized",
                    alpha=leap.plot.ALPHA,
                )
            else:
                lines = ax.plot(x, y_bfr_smooth, **free_en_kwargs)
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

            # Consistency checks.
            pks_min = peaks[plt_ix][cmp_ix][minima_ix]
            pks_max = peaks[plt_ix][cmp_ix][maxima_ix]
            if len(pks_min) != len(pks_max):
                raise ValueError(
                    "Compound: {}\n"
                    "Plot section: {}\n"
                    "The number of minima ({}) does not match the number of"
                    " maxima"
                    " ({})".format(cmp, plt_sec, len(pks_min), len(pks_max))
                )
            if plt_sec == "left" and np.any(pks_min >= pks_max):
                raise ValueError(
                    "Compound: {}\n"
                    "Plot section: {}\n"
                    "Either the first extremum is not a minium or minima and"
                    " maxima are not ordered alternately.  Minima: {}."
                    "  Maxima: {}".format(cmp, plt_sec, pks_min, pks_max)
                )
            elif plt_sec == "right" and np.any(pks_min <= pks_max):
                raise ValueError(
                    "Compound: {}\n"
                    "Plot section: {}\n"
                    "Either the first extremum is not a maximum or minima and"
                    " maxima are not ordered alternately.  Minima: {}."
                    "  Maxima: {}".format(cmp, plt_sec, pks_min, pks_max)
                )
            else:
                raise ValueError("Unknown plot section: '{}'".format(plt_sec))

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
    for cmp_ix, cmp in enumerate(compounds):
        if symmetrized:
            x = xdata_bfr_sym
            y = ydata_bfr_sym[cmp_ix]
        else:
            x = xdata
            y = ydata[cmp_ix]
        fig, ax = plt.subplots(clear=True)
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
        lines = ax.plot(x, y, **free_en_kwargs)
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
# Write out the x data that were used for peak finding.  Don't write out
# `xdata_bfr_sym`.
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
            + "Peak finding procedure:\n"
            + "\n"
            + "1.) For bulk simulations and surface simulations with\n"
            + "uncharged surfaces, the density profile is symmetrized with\n"
            + "respect to the center of the simulation box by taking the\n"
            + "average of the density profile in the left and right half of\n"
            + "the simulation box.\n"
            + "\n"
            + "2.) The free-energy profile F(z) is calculated from the\n"
            + "density profile p(z) according to\n"
            + "  F(z)/kT = -ln[p(z)/p0]\n"
            + "where k is the Boltzmann constant and T is the temperature.\n"
            + "p0 is the average density in the bulk region.  Thus, the free\n"
            + "energy in the bulk region is effectively set to zero.\n"
            + "\n"
            + "3.) Outliers are identified by comparing each F(z) value to\n"
            + "its two neighboring values.  If the value sticks out by more\n"
            + "than {:.2f} times the standard deviation of all F(z)\n".format(
                sd_factor
            )
            + "values in the bulk region, it is regarded as outlier.  The\n"
            + "identified outliers are replaced by a linear interpolation\n"
            + "between their two neighboring values.\n"
            + "\n"
            + "4.) The free-energy profile is smoothed using a\n"
            + "Savitzky-Golay filter.\n"
            + "\n"
            + "5.) Free-energy {:s} are identified on the basis of\n".format(
                peak_type
            )
            + "their peak prominence and width.  The prominence is chosen\n"
            + "such that that {:.2f} percent of the particles have a\n".format(
                100 * prob_thresh
            )
            + "higher kinetic energy in the z direction than the prominence.\n"
            + "\n"
            + "6.) The layering region must not end with a minimum, because\n"
            + "then there exists a free-energy barrier that a particle has\n"
            + "to overcome when traveling from the electrode to the bulk\n"
            + "that is higher than the prominence threshold.  Thus, if the\n"
            + "layering and bulk region are separated by a free-energy\n"
            + "minimum, search for a free-energy maximum between this\n"
            + "minimum and the bulk region.  This is done by gradually\n"
            + "decreasing the prominence threshold from a value that\n"
            + "corresponds to a kinetic energy that {:.2f} percent\n".format(
                100 * prob_thresh
            )
            + "of the particles exceed to a value that corresponds to a\n"
            + "kinetic energy that {:.2f} percent of the particles\n".format(
                100 * prob_thresh_max
            )
            + "exceed in steps of {:.2f} percent.\n".format(
                100 * prob_thresh_step_size
            )
            + "The maximum must have a height of at least {:.9e} kT.\n".format(
                height_mins[cmp_ix]
            )
            + "If no such maximum is found, the last 'maximum' between the\n"
            + "layering and bulk region is set to the point where the\n"
            + "free-energy profile reaches the first time\n"
            + "{:.9e} kT after the last minimum.  However, if the\n".format(
                cutoffs[cmp_ix]
            )
            + "last maximum is further away from the last minimum than\n"
            + "{:.2f} nm, the last maximum is set to the position of\n".format(
                max_gap_nm
            )
            + "the last minimum plus {:.2f} nm.\n".format(max_gap_nm)
            + "\n"
            + "\n"
            + "(Average) bin width and number of sample points per nm:\n"
            + "sample_spacing:   {:.9e} nm\n".format(x_sample_spacing)
            + "n_samples_per_nm: {:d}\n".format(n_samples_per_nm)
            + "\n"
            + "Start of the bulk region as distance to the electrodes:\n"
            + "bulk_start: {:.2f} nm\n".format(Elctrd.BULK_START / 10)
            + "Bulk region in absolute coordinates:\n"
            + "z_bulk_min: {:.9e} nm\n".format(bulk_region[0])
            + "z_bulk_max: {:.9e} nm\n".format(bulk_region[1])
            + "Box length in z direction in nm\n"
            + "box_z:      {:.9e} nm\n".format(box_z)
            + "Standard deviation of the free energy in the bulk region\n"
            + "bulk_sd:    {:.9e} kT\n".format(bulk_sds[cmp_ix])
            + "sd_factor:  {:.2f}\n".format(sd_factor)
            + "\n"
            + "Parameters for finding outliers with scipy.signal.find_peaks\n"
            + "outlier_threshold:  {:.9e} kT\n".format(
                outlier_thresholds[cmp_ix]
            )
            + "outlier_max_width:  {:d} sample points\n".format(
                outlier_max_width
            )
            + "outlier_rel_height: {:.2f}\n".format(outlier_rel_height)
            + "\n"
            + "Parameters for data smoothing with scipy.signal.savgol_filter\n"
            + "polyorder:     {:d}\n".format(polyorder)
            + "window_length: {:d} sample points\n".format(wlen)
            + "\n"
            + "Parameters for peak finding with scipy.signal.find_peaks\n"
            + "prob_thresh:  {:.2f}\n".format(prob_thresh)
            + "prominence:   {:.9e} kT\n".format(prominence)
            + "min_width:    {:d} sample points\n".format(min_width)
            + "max_width_nm: {:.2f} nm\n".format(max_width_nm)
            + "max_width:    {:d} sample points\n".format(max_width)
            + "rel_height:   {:.2f}\n".format(rel_height)
            + "\n"
            + "Parameters for finding the last maximum in case the layering\n"
            + "and bulk region are separated by a free-energy minimum.\n"
            + "prob_thresh_max:       {:.2f}\n".format(prob_thresh_max)
            + "prob_thresh_step_size: {:.2f}\n".format(prob_thresh_step_size)
            + "prominence_min:        {:.9e} kT\n".format(
                np.min(prominences_last_max)
            )
            + "height_min:            {:.9e} kT\n".format(height_mins[cmp_ix])
            + "plateau_size_max:      {:d} sample points\n".format(
                plateau_size_max
            )
            + "cutoff:                {:.9e} kT\n".format(cutoffs[cmp_ix])
            + "max_gap_nm:            {:.2f} nm\n".format(max_gap_nm)
            + "max_gap:               {:d} sample points\n".format(max_gap)
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
        pk_pos = x[pks]
        pks_h = peak_heights_full[cmp_ix][fac_ix]
        props = properties_full[cmp_ix][fac_ix]
        bw = base_widths_full[cmp_ix][fac_ix]
        fwhm = fwhm_full[cmp_ix][fac_ix]

        data = np.column_stack([pks, pk_pos, pks_h])

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

        # Consistency checks.
        if np.any(pk_pos <= elctrd_thk):
            raise ValueError(
                "Compound: {}\n"
                "Peak type: {}\n"
                "At least one peak lies within the left electrode.  Peak"
                " positions: {}.  Left electrode:"
                " {}".format(cmp, peak_type, pk_pos, elctrd_thk)
            )
        if np.any(pk_pos >= box_z - elctrd_thk):
            raise ValueError(
                "Compound: {}\n"
                "Peak type: {}\n"
                "At least one peak lies within the right electrode.  Peak"
                " positions: {}.  Right electrode:"
                " {}".format(cmp, peak_type, pk_pos, box_z - elctrd_thk)
            )
        if np.any((pk_pos >= bulk_region[0]) & (pk_pos <= bulk_region[1])):
            raise ValueError(
                "Compound: {}\n"
                "Peak type: {}\n"
                "At least one peak lies within the bulk region.  Peak"
                " positions: {}.  Bulk region:"
                " {}".format(cmp, peak_type, pk_pos, bulk_region)
            )
        if surfq == 0:
            pk_is_left = pk_pos <= (box_z / 2)
            n_pks_left = np.count_nonzero(pk_is_left)
            n_pks_right = len(pk_is_left) - n_pks_left
            if n_pks_left != n_pks_right:
                raise ValueError(
                    "The surface charge is {} e/nm^2 but the number of left"
                    " ({}) and right free-energy {} ({}) do not match for"
                    " compound {}.".format(
                        surfq, n_pks_left, peak_type, n_pks_right, cmp
                    )
                )

print("Done")
