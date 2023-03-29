"""Module containing functions for plotting."""


# Third-party libraries
import numpy as np

# First-party libraries
import lintf2_ether_ana_postproc as leap


ALPHA = 0.75  # Default transparency for transparent objects.
atom_type2display_name = {
    "Li": r"Li",
    "NBT": r"N_{TFSI}",
    "OBT": r"O_{TFSI}",
    "OE": r"O_{PEO}",
}


def plot_elctrd_left(ax, offset=0, **kwargs):
    """
    Plot the position of the electrode layers of the left electrode as
    vertical lines into an :class:`matplotlib.axes.Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` into which to plot the
        electrode positions.
    offset : float, optional
        x value at which to position the rightmost layer of the left
        electrode.
    kwargs : dict
        Keyword arguments to parse to
        :func:`matplotlib.axes.Axes.axvline`.  See there for possible
        options.  By default, `color` is set to ``"tab:gray"`` and
        `linestyle` is set to ``"dashed"``.
    """
    kwargs.setdefault("color", "tab:gray")
    kwargs.setdefault("linestyle", "dashed")

    Elctrd = leap.simulation.Electrode()
    elctrd_pos_z = offset
    for _ in range(Elctrd.GRA_LAYERS_N):
        ax.axvline(x=elctrd_pos_z, **kwargs)
        elctrd_pos_z -= Elctrd.GRA_LAYER_DIST / 10  # nm -> Angstrom


def plot_elctrd_right(ax, offset=0, **kwargs):
    """
    Plot the position of the electrode layers of the right electrode as
    vertical lines into an :class:`matplotlib.axes.Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` into which to plot the
        electrode positions.
    offset : float, optional
        x value at which to position the leftmost layer of the right
        electrode.
    kwargs : dict
        Keyword arguments to parse to
        :func:`matplotlib.axes.Axes.axvline`.  See there for possible
        options.  By default, `color` is set to ``"tab:gray"`` and
        `linestyle` is set to ``"dashed"``.
    """
    kwargs.setdefault("color", "tab:gray")
    kwargs.setdefault("linestyle", "dashed")

    Elctrd = leap.simulation.Electrode()
    elctrd_pos_z = offset
    for _ in range(Elctrd.GRA_LAYERS_N):
        ax.axvline(x=elctrd_pos_z, **kwargs)
        elctrd_pos_z += Elctrd.GRA_LAYER_DIST / 10  # nm -> Angstrom


def plot_elctrds(ax, offset_right, offset_left=0, **kwargs):
    """
    Plot the position of the electrode layers of the left and right
    electrode as vertical lines into an :class:`matplotlib.axes.Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` into which to plot the
        electrode positions.
    offset_left, offset_right : float
        x values at which to position the rightmost layer of the left
        electrode and the leftmost layer of the right electrode.
    kwargs : dict
        Keyword arguments to parse to
        :func:`lintf2_ether_ana_postproc.plot.plot_elctrd_left` and
        :func:`lintf2_ether_ana_postproc.plot.plot_elctrd_right`.  See
        there for possible options.  By default, `color` is set to
        ``"tab:gray"`` and `linestyle` is set to ``"dashed"``.
    """
    kwargs.setdefault("color", "tab:gray")
    kwargs.setdefault("linestyle", "dashed")
    leap.plot.plot_elctrd_left(ax, offset_left, **kwargs)
    leap.plot.plot_elctrd_right(ax, offset_right, **kwargs)


def peak_proms(ax, x, y, peaks, properties, peak_type=None, **kwargs):
    """
    Plot the peak prominences as calculated by
    :func:`scipy.signal.find_peaks` into an
    :class:`matplotlib.axes.Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` into which to plot the peak
        prominences.
    x, y : array_like
        1-dimensional arrays containing the x and y data.
    peaks : array_like
        Indices of the peaks as returned by
        :func:`scipy.signal.find_peaks`.
    properties : dict
        Dictionary containing the properties of peaks as returned by
        :func:`scipy.signal.find_peaks`.  The dictionary must contain
        the key "prominences".
    peak_type : (None, "min", "max"), optional
        Specify whether the peaks are minima or maxima.  If ``None``,
        the peak type will be guessed by comparing the peak values to
        their neighboring values.
    kwargs : dict, optional
        Keyword arguments to parse to
        :func:`matplotlib.axes.Axes.vlines`.  See there for possible
        options.  By default, `alpha` is set to 0.75 and `color` is set
        to "limegreen" for maxima and `darkgreen` for minima.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    peaks = np.asarray(peaks)

    if peak_type is None:
        if leap.misc.peaks_are_max(y, peaks):
            peak_type = "max"
        else:
            peak_type = "min"
    if peak_type.lower() == "max":
        ymin = y[peaks] - properties["prominences"]
        kwargs.setdefault("color", "limegreen")
    elif peak_type.lower() == "min":
        ymin = y[peaks] + properties["prominences"]
        kwargs.setdefault("color", "darkgreen")
    else:
        raise ValueError("Unknown `peak_type`: {}".format(peak_type))
    kwargs.setdefault("alpha", ALPHA)
    ax.vlines(x=x[peaks], ymin=ymin, ymax=y[peaks], **kwargs)


def peak_widths(ax, x, y, peaks, properties, peak_type=None, **kwargs):
    """
    Plot the peak widths as calculated by
    :func:`scipy.signal.find_peaks` into an
    :class:`matplotlib.axes.Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` into which to plot the peak
        widths.
    x, y : array_like
        1-dimensional arrays containing the x and y data.
    peaks : array_like
        Indices of the peaks as returned by
        :func:`scipy.signal.find_peaks`.
    properties : dict
        Dictionary containing the properties of peaks as returned by
        :func:`scipy.signal.find_peaks`.  The dictionary must contain
        the keys "width_heights", "left_ips" and "right_ips".
    peak_type : (None, "min", "max"), optional
        Specify whether the peaks are minima or maxima.  If ``None``,
        the peak type will be guessed by comparing the peak values to
        their neighboring values.
    kwargs : dict, optional
        Keyword arguments to parse to
        :func:`matplotlib.axes.Axes.hlines`.  See there for possible
        options.  By default, `alpha` is set to 0.75 and `color` is set
        to "red" for maxima and `darkred` for minima.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    peaks = np.asarray(peaks)

    if peak_type is None:
        if leap.misc.peaks_are_max(y, peaks):
            peak_type = "max"
        else:
            peak_type = "min"
    if peak_type.lower() == "max":
        heights = properties["width_heights"]
        kwargs.setdefault("color", "red")
    elif peak_type.lower() == "min":
        heights = -properties["width_heights"]
        kwargs.setdefault("color", "darkred")
    else:
        raise ValueError("Unknown `peak_type`: {}".format(peak_type))
    kwargs.setdefault("alpha", ALPHA)

    ndx = np.arange(len(x), dtype=np.uint32)
    left_ips = np.interp(properties["left_ips"], ndx, x)
    right_ips = np.interp(properties["right_ips"], ndx, x)

    ax.hlines(y=heights, xmin=left_ips, xmax=right_ips, **kwargs)


def peak_proms_widths(
    ax,
    x,
    y,
    peaks,
    properties,
    peak_type=None,
    kwargs_proms=None,
    kwargs_widths=None,
):
    """
    Plot the peak prominences and widths as calculated by
    :func:`scipy.signal.find_peaks` into an
    :class:`matplotlib.axes.Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` into which to plot the peak
        prominences and widths.
    x, y : array_like
        1-dimensional arrays containing the x and y data.
    peaks : array_like
        Indices of the peaks as returned by
        :func:`scipy.signal.find_peaks`.
    properties : dict
        Dictionary containing the properties of peaks as returned by
        :func:`scipy.signal.find_peaks`.  The dictionary must contain
        the keys "prominences", "width_heights", "left_ips" and
        "right_ips".
    peak_type : (None, "min", "max"), optional
        Specify whether the peaks are minima or maxima.  If ``None``,
        the peak type will be guessed by comparing the peak values to
        their neighboring values.
    kwargs_proms : dict or None, optional
        Keyword arguments to parse to
        :func:`lintf2_ether_ana_postproc.plot.peak_proms`.  See there
        for possible options.
    kwargs_widths : dict or None, optional
        Keyword arguments to parse to
        :func:`lintf2_ether_ana_postproc.plot.peak_widths`.  See there
        for possible options.
    """
    if kwargs_proms is None:
        kwargs_proms = {}
    if kwargs_widths is None:
        kwargs_widths = {}
    leap.plot.peak_proms(
        ax, x, y, peaks, properties, peak_type, **kwargs_proms
    )
    leap.plot.peak_widths(
        ax, x, y, peaks, properties, peak_type, **kwargs_widths
    )


def peaks(
    ax,
    x,
    y,
    peaks,
    properties,
    widths=None,
    peak_type=None,
    kwargs_scatter=None,
    kwargs_proms=None,
    kwargs_widths=None,
    kwargs_hlines=None,
):
    """
    Plot the peaks extrema, prominences and widths as calculated by
    :func:`scipy.signal.find_peaks` into an
    :class:`matplotlib.axes.Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` into which to plot the peak
        extrema, prominences and widths.
    x, y : array_like
        1-dimensional arrays containing the x and y data.
    peaks : array_like
        Indices of the peaks as returned by
        :func:`scipy.signal.find_peaks`.
    properties : dict
        Dictionary containing the properties of peaks as returned by
        :func:`scipy.signal.find_peaks`.  The dictionary must contain
        the keys "prominences", "width_heights", "left_ips" and
        "right_ips".
    widths : array_like or None, optional
        Optionally, additional widths of the peaks as returned by
        :func:`scipy.signal.peak_widths`.  This could for example be the
        widths of the lowest contour lines.
    peak_type : (None, "min", "max"), optional
        Specify whether the peaks are minima or maxima.  If ``None``,
        the peak type will be guessed by comparing the peak values to
        their neighboring values.
    kwargs_scatter : dict or None, optional
        Keyword arguments to parse to
        :func:`matplotlib.axes.Axes.scatter`.  See there for possible
        options.
    kwargs_proms : dict or None, optional
        Keyword arguments to parse to
        :func:`lintf2_ether_ana_postproc.plot.peak_prom`.  See there for
        possible options.
    kwargs_widths : dict or None, optional
        Keyword arguments to parse to
        :func:`lintf2_ether_ana_postproc.plot.peak_width`.  See there
        for possible options.
    kwargs_hlines : dict or None, optional
        Keyword arguments to parse to
        :func:`matplotlib.axes.Axes.hlines`.  See there for possible
        options.  Only relevant if `widths` is provided.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    peaks = np.asarray(peaks)

    if kwargs_scatter is None:
        kwargs_scatter = {}
    if kwargs_hlines is None:
        kwargs_hlines = {}

    if peak_type is None:
        if leap.misc.peaks_are_max(y, peaks):
            peak_type = "max"
        else:
            peak_type = "min"
    if peak_type.lower() == "max":
        kwargs_scatter.setdefault("marker", "1")
        kwargs_scatter.setdefault("color", "orange")
        kwargs_hlines.setdefault("color", "violet")
        fac = 1
    elif peak_type.lower() == "min":
        kwargs_scatter.setdefault("marker", "2")
        kwargs_scatter.setdefault("color", "darkorange")
        kwargs_hlines.setdefault("color", "darkviolet")
        fac = -1
    else:
        raise ValueError("Unknown `peak_type`: {}".format(peak_type))
    kwargs_scatter.setdefault("alpha", ALPHA)
    # `zorder` of lines is 2, `zorder` of major ticks is 2.01 -> set
    # `zorder` of the scatter points to 2.001 to ensure that they lie
    # above lines but below the major ticks.
    kwargs_scatter.setdefault("zorder", 2.001)
    kwargs_hlines.setdefault("alpha", ALPHA)

    ax.scatter(x[peaks], y[peaks], **kwargs_scatter)
    leap.plot.peak_proms_widths(
        ax, x, y, peaks, properties, peak_type, kwargs_proms, kwargs_widths
    )
    if widths is not None:
        ndx = np.arange(len(x), dtype=np.uint32)
        left_ips = np.interp(widths[2], ndx, x)
        right_ips = np.interp(widths[3], ndx, x)
        ax.hlines(
            y=fac * widths[1], xmin=left_ips, xmax=right_ips, **kwargs_hlines
        )
