"""Module containing functions for plotting."""


# Third-party libraries
import numpy as np

# First-party libraries
import lintf2_ether_ana_postproc as leap


ALPHA = 0.75  # Default transparency for transparent objects.
ATOM_TYPE2DISPLAY_NAME = {
    "Li": "Li",
    "NBT": r"N_{TFSI}",
    "OBT": r"O_{TFSI}",
    "OE": r"O_{PEO}",
    "NTf2": "TFSI",
    "TFSI": "TFSI",
    "ether": "PEO",
    "PEO": "PEO",
}


def elctrd_left(ax, offset=0, **kwargs):
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


def elctrd_right(ax, offset=0, **kwargs):
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


def elctrds(ax, offset_right, offset_left=0, **kwargs):
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
    leap.plot.elctrd_left(ax, offset_left, **kwargs)
    leap.plot.elctrd_right(ax, offset_right, **kwargs)


def bins(
    ax, bins=None, Sim=None, infile=None, conv=1, kwargs_txt=None, **kwargs
):
    """
    Plot the position of the given bin edges as vertical lines into an
    :class:`matplotlib.axes.Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` into which to plot the
        bin edges
    bins : array_like, optional
        1-dimensional array of bin edges.
    Sim : lintf2_ether_ana_postproc.simulation.Simulation
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        instance.  If provided, the bin edges will be read from the
        first column of
        ``Sim.settings + "_" + Sim.system +
        "_density-z_number_Li_binsA.txt.gz"``.
    infile : str, optional
        If provided, the bin edges will be read from the given file.
    conv : float, optional
        Conversion factor for the bin edges (e.g. to convert from
        Angstroms to nm).
    kwargs_txt : None or dict, optional
        Keyword arguments to parse to :func:`numpy.loadtxt` when reading
        the bin edges from `infile`.  See there for possible options.
    kwargs : dict
        Keyword arguments to parse to
        :func:`matplotlib.axes.Axes.axvline`.  See there for possible
        options.  By default, `color` is set to ``"black"`` and
        `linestyle` is set to ``"dotted"``.

    Notes
    -----
    If multiple of `bins`, `Sim` or `infile` are provided, `bins` takes
    precedence over `Sim` and `Sim` takes precedence over `infile`.
    """
    kwargs.setdefault("color", "black")
    kwargs.setdefault("linestyle", "dotted")

    if bins is not None:
        bins = np.array(bins, copy=True)
    elif Sim is not None:
        analysis = "density-z"  # Analysis name.
        analysis_suffix = "_number"  # Analysis name specification.
        tool = "gmx"  # Analysis software.
        file_suffix = analysis + analysis_suffix + "_Li_binsA.txt.gz"
        infile = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
        bins = np.loadtxt(infile, usecols=0)
    elif infile is not None:
        if kwargs_txt is None:
            kwargs_txt = {}
        bins = np.loadtxt(infile, **kwargs_txt)
    else:
        raise ValueError("Either `bins`, `Sim` or `infile` must be provided.")

    bins *= conv
    for bn in bins:
        ax.axvline(x=bn, **kwargs)


def profile(
    ax,
    x=None,
    profile=None,
    Sim=None,
    cmp="Li",
    infile=None,
    conv=1,
    free_en=False,
    kwargs_txt=None,
    kwargs_set=None,
    **kwargs_plt,
):
    """
    Plot the given profile (e.g. density or free energy) into an
    :class:`matplotlib.axes.Axes`.

    The axes and spines of the plot are removed so that only the profile
    is visible.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` into which to plot the
        bin edges
    x, profile : array_like, optional
        x values and the corresponding profile values.
    Sim : lintf2_ether_ana_postproc.simulation.Simulation
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        instance.  If provided, the `x` and `profile` will be read from
        ``Sim.settings + "_" + Sim.system +
        "_density-z_number.xvg.gz"``.  `x` will be read from the first
        column, `profile` will be read from the column specified by
        `cmp`.
    cmp : str, optional
        The compound whose profile to read from
        ``Sim.settings + "_" + Sim.system + "_density-z_number.xvg.gz"``
        if `Sim` is provided.
    infile : str, optional
        If provided, `x` and `profile` will be read from the given file.
    conv : float, optional
        Conversion factor for the x values (e.g. to convert from
        Angstroms to nm).
    free_en : bool, optional
        If ``True``, the negative logarithm of `profile` will be
        plotted.  If `profile` was a density profile, this means
        conversion to a free-energy profile.
    kwargs_txt : None or dict, optional
        Keyword arguments to parse to :func:`numpy.loadtxt` when reading
        the bin edges from `infile`.  See there for possible options.
    kwargs_set : None or dict, optional
        Keyword arguments to parse to :func:`matplotlib.axes.Axes.set`.
        See there for possible arguments.  By default, `ylim` is set to
        the minimum and maximum value of `profile`.
    kwargs : dict
        Keyword arguments to parse to
        :func:`matplotlib.axes.Axes.plot`.  See there for possible
        options.  By default, `color` is set to ``"black"``.

    Notes
    -----
    If multiple of `bins`, `Sim` or `infile` are provided, `bins` takes
    precedence over `Sim` and `Sim` takes precedence over `infile`.
    """
    kwargs_plt.setdefault("color", "black")

    if profile is not None:
        if x is None:
            raise ValueError("`x` must be provided if `profile` is provided")
        x = np.array(x, copy=True)
        profile = np.asarray(profile)
    elif Sim is not None:
        analysis = "density-z"  # Analysis name.
        analysis_suffix = "_number"  # Analysis name specification.
        tool = "gmx"  # Analysis software.
        file_suffix = analysis + analysis_suffix + ".xvg.gz"
        infile = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
        cols = (0, Sim.dens_file_cmp2col[cmp])
        x, profile = np.loadtxt(
            infile, comments=["#", "@"], usecols=cols, unpack=True
        )
    elif infile is not None:
        if kwargs_txt is None:
            kwargs_txt = {}
        x, profile = np.loadtxt(infile, **kwargs_txt)
    else:
        raise ValueError(
            "Either `profile`, `Sim` or `infile` must be provided."
        )

    if free_en:
        profile = -np.log(profile)

    if kwargs_set is None:
        kwargs_set = {}
    ylim = kwargs_set.pop("ylim", None)
    if ylim is None:
        valid = np.isfinite(profile)
        ymin = np.min(profile[valid])
        ymin = ymin - 0.4
        ymax = np.max(profile[valid])
        kwargs_set.setdefault("ylim", (ymin, ymax))

    x *= conv
    ax.plot(x, profile, **kwargs_plt)
    ax.set(**kwargs_set)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
