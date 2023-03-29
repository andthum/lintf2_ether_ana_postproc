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
