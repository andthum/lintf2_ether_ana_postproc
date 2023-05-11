"""Miscellaneous functions."""


# Third-party libraries
import mdtools as mdt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import norm

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


def symmetrize_data(x, y, x2shift=None, reassemble=False, tol=1e-4):
    """
    Compute the average of the first and second half of the given data.

    Cut the given data in two halves.  Mirror the second half and take
    the average of both halves.  This can e.g. be useful when the data
    are expected to be symmetric to the center, but due to statistical
    noise or other reasons the symmetry is broken.  By averaging the
    first and second half of the data, the symmetry can be restored.

    Parameters
    ----------
    x : array_like
        1-dimensional array of x values.
    y : array_like
        Array of corresponding y values.  `y` can be 1-dimensional or
        2-dimensional.  In both cases, ``y.shape[-1]`` must match
        ``len(x)``.  In the 2-dimensional case, `y` is interpreted as
        array of y arrays.
    x2shift : float or None, optional
        A shift value for the second half of the x values.  If provided,
        the second half of the x data will be
        ``x2 = x2shift - x[len(x) - len(x) // 2 :][::-1]``.
        Otherwise, the second half of the x data will be
        ``x2 = x[len(x) - len(x) // 2 :][::-1]``.
    reassemble : bool, optional
        If ``True``, join the computed average data with their mirror
        image to reassemble the input data in a symmetrized manner.
        This means, join `x_av` with ``x_av[::-1]`` (or with
        ``(x2shift - x_av)[::-1]`` if `x2shift` was provided) and join
        `y_av` with ``y_av[::-1]`` (or with ``y_av[:, ::-1]`` if `y` is
        2-dimensional).
    tol : float or None, optional
        If provided, the x values in the first and second half must be
        equal within the given tolerance.  Good values for `tol` are
        1e-4 if `x` is given in nanometers or 1e-3 if `x` is given in
        Angstroms.

    Returns
    -------
    x_av : numpy.ndarray
        Array containing the averaged x values.  If `reassemble` is
        ``True``, the shape of `x_av` will be the same as that of `x`.
        Otherwise, it will be ``(np.ceil(len(x) / 2),)``.
    y_av : numpy.ndarray
        Array containing the averaged y values.  If `reassemble` is
        ``True``, the shape of `y_av` will be the same as that of `y`.
        Otherwise, the length of the last dimension (i.e.
        ``y_av.shape[-1]``) will be ``np.ceil(len(x) / 2)``.

    Notes
    -----
    This function was written to symmetrize the density (or free-energy)
    profiles of surface simulations with uncharged surfaces.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1:
        raise ValueError(
            "`x` has {} dimension(s) but must be 1-dimensional".format(x.ndim)
        )
    if y.ndim not in (1, 2):
        raise ValueError(
            "`y` has {} dimension(s) but must be 1-dimensional or"
            " 2-dimensional".format(y.ndim)
        )
    if y.shape[-1] != len(x):
        raise ValueError(
            "`y.shape[-1]` ({}) must be equal to `len(x)`"
            " ({})".format(y.shape[-1], len(x))
        )

    n_data = len(x)
    n_half = n_data // 2
    x1 = x[:n_half]
    x2 = x[n_data - n_half :]
    x2 = x2[::-1]
    if x2shift is not None:
        x2 = x2shift - x2
    if n_data % 2 != 0:
        # `x_mid` must be an array, otherwise the call of
        # `np.concatenate` below will fail.  `mdt.nph.take will` always
        # returns a view of the input array, thus the result will also
        # be an array.  In contrast, `x_mid[n_half]` will return a
        # scalar.
        x_mid = mdt.nph.take(x, start=n_half, stop=n_half + 1, axis=-1)
    if x1.shape != x2.shape:
        raise ValueError(
            "The number of x data points in the first and second half do not"
            " match: `x1.shape` ({}) != `x2.shape` ({}).  This should not"
            " have happened".format(x1.shape, x2.shape)
        )
    if tol is not None and not np.allclose(x1, x2, rtol=0, atol=tol):
        raise ValueError(
            "The x values in the first and second half do not math within the"
            " given tolerance of {}.  `x1` = {}.  `x2` ="
            " {}".format(tol, x1, x2)
        )
    x = np.mean([x1, x2], axis=0)
    del x1, x2

    y1 = mdt.nph.take(y, start=0, stop=n_half, axis=-1)
    y2 = mdt.nph.take(y, start=y.shape[-1] - n_half, stop=None, axis=-1)
    y2 = mdt.nph.take(y2, step=-1, axis=-1)
    if n_data % 2 != 0:
        y_mid = mdt.nph.take(y, start=n_half, stop=n_half + 1, axis=-1)
    if y1.shape != y2.shape:
        raise ValueError(
            "The number of y data points in the first and second half do not"
            " match: `y1.shape` ({}) != `y2.shape` ({}).  This should not"
            " have happened".format(y1.shape, y2.shape)
        )
    y = np.mean([y1, y2], axis=0)
    del y1, y2

    if reassemble:
        if x2shift is not None:
            # x2 = x2shift - x2 can mathematically be expressed as
            # x2' = x2shift - x2
            # => x2 = x2shift - x2'
            x2 = x2shift - x
            x2 = x2[::-1]
        else:
            x2 = x[::-1]
        if n_data % 2 != 0:
            x = np.concatenate([x, x_mid, x2], axis=-1)
            y = np.concatenate(
                [y, y_mid, mdt.nph.take(y, step=-1, axis=-1)], axis=-1
            )
        else:
            x = np.concatenate([x, x2], axis=-1)
            y = np.concatenate([y, mdt.nph.take(y, step=-1, axis=-1)], axis=-1)
    elif n_data % 2 != 0:  # and not reassemble
        x = np.concatenate([x, x_mid], axis=-1)
        y = np.concatenate([y, y_mid], axis=-1)

    return x, y


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


def e_kin(p):
    r"""
    Return the kinetic energy that 100*`p` percent of the particles of a
    system in a canonical (:math:`NVT`) ensemble exceed.

    Parameters
    ----------
    p : scalar or array_like
        The fraction of particles that should have a higher kinetic
        energy than the returned one.

    Returns
    -------
    e_kin : scalar or numpy.ndarray
        The kinetic energy threshold in :math:`k_B T` that 100*`p`
        percent of the particles exceed.

    Notes
    -----
    In the canonical ensemble, the probability density distribution of
    the kinetic energy :math:`E_{kin}` of a particle moving along one
    spatial dimension is given by

    .. math::

        \rho(E_{kin}) = \frac{1}{\sqrt{\pi k_B T}} \frac{1}{E_{kin}}
        \exp{\left( -\frac{E_kin}{k_B T} \right)}

    where, :math:`k_B` is the Boltzmann constant and :math:`T` is the
    temperature.  From this follows the probability that a particle has
    a kinetic energy higher than :math:`c` along this spatial dimension.

    .. math::

        p(E_{kin} > c) = 1 - p(E_{kin} \leq c) = 2
        \left[
            1 - \Phi_{0,1}
            \left(
                \sqrt{\frac{2c}{k_B T}}
            \right)
        \right]

    Here, :math:`\Phi_{0,1}(z)` denotes the cumulative distribution
    function of the normal distribution.  Thus, given a probability
    :math:`p(E_{kin} > c)` that a particle has a kinetic energy higher
    than :math:`c`, the corresponding kinetic energy :math:`c` can be
    calculated by

    .. math::

        c = \frac{k_B T}{2}
        \left(
            \Phi_{0,1}^{-1}
            \left[
                1 - \frac{1}{2} p(E_{kin} > c)
            \right]
        \right)^2

    See the notes on pages 142-147 in my "derivation book" for a
    derivation of the above formulas.
    """
    if p < 0 or p > 1:
        raise ValueError("`p` ({}) must be between 0 and 1".format(p))
    return 0.5 * norm.ppf(1 - 0.5 * p) ** 2
