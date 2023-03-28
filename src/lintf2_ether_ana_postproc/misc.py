"""Miscellaneous functions."""


# Third-party libraries
import mdtools as mdt
import numpy as np
from scipy.signal import find_peaks

# First-party libraries
import lintf2_ether_ana_postproc as leap


def generate_equidistant_bins(start=0, stop=None, bin_width_desired=10):
    """
    Generate equidistant bins.

    Parameters
    ----------
    start, stop : float
        First and last bin edge.
    bin_width_desired : float
        Desired bin width.
    """
    if stop >= start:
        raise ValueError(
            "`stop` ({}) must be less than `start` ({})".format(stop, start)
        )
    if bin_width_desired <= 0:
        raise ValueError(
            "`bin_width_desired` ({}) must be greater than"
            " zero".format(bin_width_desired)
        )

    dist = start - stop
    n_bins = round(dist / bin_width_desired)
    bin_width_actual = dist / n_bins
    print("Binning distance:  {:>11.6f}".format(dist))
    print("Desired bin width: {:>11.6f}".format(bin_width_desired))
    print("Actual bin width:  {:>11.6f}".format(bin_width_actual))
    print("Number of bins:    {:>4d}".format(n_bins))
    print("Equidistant Bins:")
    edge = start
    while edge <= stop:
        print("{:>16.9e}".format(edge))
        edge += bin_width_actual


def find_nearest(a, vals, tol=0.01):
    """
    Get the indices of the values of an array that are closest to the
    given values.

    Parameters
    ----------
    a : array_like
        The input array.  The search is performed over the flattened
        array.
    vals : array_like
        The values for which to get the indices.  If a value is not
        contained in `a`, the index of the next closest value is
        returned.  If a value is contained multiple times in `a`, only
        the index of the first occurrence is returned.
    tol : float, optional
        Tolerance how much values in `vals` and the actually found
        values in `a` can differ.

    Returns
    -------
    ix : numpy.ndarray
        Indices of the values in `a` that are closest to `vals`.

    See Also
    --------
    :func:`mdtools.numpy_helper_functions.find_nearest` :
        Find the values in an array which are closest to a given value
        along an axis
    """
    ix = np.zeros_like(vals, dtype=int)
    for i, val in enumerate(vals):
        val_at_ix, ix[i] = mdt.nph.find_nearest(a, val, return_index=True)
        if not np.isclose(val_at_ix, val, rtol=0, atol=tol):
            raise ValueError(
                "`val_at_ix` ({}) != `val` ({})".format(val_at_ix, val)
            )
    return ix


def dens2free_energy(x, dens, bulk_region=None):
    r"""
    Calculate free-energy profiles from density profiles.

    Parameters
    ----------
    x : array_like
        x values at which the density profile was sampled.
    dens : array_like
        Density profile.
    bulk_region : None or 2-tuple of floats, optional
        Start and end of the bulk region in units of `x`.  If provided,
        the free-energy profile will be shifted such that the free
        energy in the bulk region is zero.

    Returns
    -------
    free_en : numpy.ndarray
        Free-energy profile :math:`F(z)` in units of :math:`k_B T`, i.e.
        :math:`\frac{F(z)}{k_B T}`.

    Notes
    -----
    The free-energy profile :math:`F(z)` is calculated from the density
    profile :math:`\rho(z)` according to

    .. math::

        \frac{F(z)}{k_B T} =
        -\ln\left[ \frac{\rho(z)}{\rho^\circ} \right]

    Here, :math:`k_B` is the Boltzmann constant and :math:`T` is the
    temperature.  If `bulk_start` is given, :math:`\rho^\circ` is set to
    the average density in the bulk region (i.e. the free energy in the
    bulk region is effectively set to zero).  If `bulk_start` is None,
    :math:`\rho^\circ` is set to one.
    """
    free_en = -np.log(dens)
    if bulk_region is not None:
        try:
            if len(bulk_region) != 2:
                raise ValueError(
                    "`bulk_region` ({}) must be None or a 2-tuple of"
                    " floats.".format(bulk_region)
                )
        except TypeError:
            raise TypeError(
                "`bulk_region` ({}) must be None or a 2-tuple of"
                " floats.".format(bulk_region)
            )
        start, stop = leap.misc.find_nearest(x, bulk_region, tol=0.01)
        free_en_bulk = np.mean(free_en[start : stop + 1])
        free_en -= free_en_bulk
    return free_en


def interp_outliers(x, y, inplace=False, **kwargs):
    r"""
    Find outliers and replace them with linearly interpolated values.

    First, find outliers in `y` using :func:`scipy.signal.find_peaks`.
    Second, replace these outliers by linearly interpolating their
    neighboring `y` values using :func:`numpy.interp`.

    Parameters
    ----------
    x, y : array_like
        1-dimensional input arrays containing the x- and y-coordinates
        of the data points.  `x` must be monotonically increasing.  `x`
        and `y` must not contain NaN values.
    inplace : bool, optional
        If ``True``, change `y` in place (therefore, `y` must be a
        :class:`numpy.ndarray`).
    kwargs : dict, optional
        Keyword arguments to parse to :func:`scipy.signal.find_peaks`.
        See there for possible options.  By default, `threshold` is set
        to ``4 * numpy.std(y)``, `width` is set to ``(None, 2)`` and
        `rel_height` is set to ``0.2``.

    Returns
    -------
    y_interp : numpy.ndarray
        y-coordinates where outliers are replaced by interpolated
        values.

    Notes
    -----
    The default `kwargs` should be fine for detecting outliers in
    normally distributed `y` data.  With the default value of
    `threshold` of :math:`4 \sigma`, 0.006334 % of the `y` data would
    be detected as outliers.  See the table at
    https://de.wikipedia.org/wiki/Normalverteilung#Streuintervalle
    for the expected percentage of values of a normally distributed
    variable that lie within the interval
    :math:`[\mu - z \sigma, \mu + z \sigma]`.  However, note that
    :math:`\sigma` is susceptible to outliers and will increase when you
    have too much or too large outliers.  Thus, the :math:`4 \sigma`
    criterion is not suitable for all data.
    """
    x = np.asarray(x)
    y = np.array(y, copy=not inplace)
    if y.shape != x.shape:
        raise ValueError(
            "`y` ({}) must have the same shape as `x`"
            " ({})".format(y.shape, x.shape)
        )
    if np.any(np.isnan(y)):
        # `scipy.signal.find_peaks` cannot handle NaN values.
        raise ValueError("`y` must not contain NaN values")
    if np.any(np.isnan(x)):
        # `numpy.interp` cannot handle NaN values.
        raise ValueError("`x` must not contain NaN values")
    if np.any(np.diff(x) <= 0):
        # `x` must be monotonically increasing for `numpy.interp`.
        raise ValueError("`x` must be monotonically increasing")

    kwargs.setdefault("width", (None, 2))
    kwargs.setdefault("rel_height", 0.2)
    if kwargs.get("threshold") is None:
        kwargs["threshold"] = 4 * np.std(y)

    for factor in (-1, 1):  # -1 for finding minima, 1 for maxima.
        outliers, _properties = find_peaks(factor * y, **kwargs)
        valid = np.ones_like(y, dtype=bool)
        valid[outliers] = False
        y[outliers] = np.interp(x[outliers], x[valid], y[valid])

    return y


def peaks_are_max(y, peaks):
    """
    Check if all given peaks are maxima or minima.

    Parameters
    ----------
    y : array_like
        1-dimensional array of y coordinates.
    peaks : array_like
        1-dimensional array of peak indices as returned by
        :func:`scipy.signal.find_peaks`.

    Returns
    -------
    peaks_are_max : bool
        ``True`` if the y values of all peaks are lower or equal to
        their neighboring y values.  ``False`` if the y values of all
        peaks are higher or equal to their neighboring y values.

    Raises
    ------
    ValueError :
        If the given peaks are a mix of maxima and minima or if the
        given peaks have no neighboring values.
    """
    y = np.asarray(y)
    peaks = np.asarray(peaks)

    try:
        y[peaks]
    except IndexError:
        raise IndexError("`peaks` is not a suitable index array for `y`")

    ix_left = peaks - 1
    ix_right = peaks + 1
    valid_ix = (ix_left > 0) & (ix_right < len(y))
    if not np.any(valid_ix):
        raise ValueError("The given peaks have no neighboring values")

    left_is_lower = np.all(y[ix_left[valid_ix]] <= y[peaks[valid_ix]])
    right_is_lower = np.all(y[ix_right[valid_ix]] <= y[peaks[valid_ix]])
    if left_is_lower and right_is_lower:
        return True

    left_is_higher = np.all(y[ix_left[valid_ix]] >= y[peaks[valid_ix]])
    right_is_higher = np.all(y[ix_right[valid_ix]] >= y[peaks[valid_ix]])
    if left_is_higher and right_is_higher:
        return False

    raise ValueError("The given peaks are a mix of maxima and minima")
