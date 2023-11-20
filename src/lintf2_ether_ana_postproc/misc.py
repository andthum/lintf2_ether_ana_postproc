"""Miscellaneous functions."""


# Standard libraries
from copy import deepcopy

# Third-party libraries
import mdtools as mdt
import numpy as np
from scipy import constants
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks
from scipy.stats import norm

# First-party libraries
import lintf2_ether_ana_postproc as leap


def straight_line(x, m, c):
    """
    Straight Line.

    Calculate the y values that belong to the given x values for a
    straight line with the a given slope `m` and intercept `c`.

    Parameters
    ----------
    x : scalar or array_like
        x values.
    m : scalar or array_like
        Slope.
    c : scalar or array_like
        y axis intercept.

    Returns
    -------
    y : scalar or numpy.ndarray
        ``y = m * x + c``.

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    return np.add(np.multiply(m, x), c)


def line_inv(y, xp1, xp2, fp1, fp2):
    """
    Inverse straight line.

    Return the `x` values that belong to the given `y` values for a
    straight line that is defined by the two points (`xp1`, `fp1`) and
    (`xp2`, `fp2`).

    Parameters
    ----------
    y : scalar or array_like
        `y` values for which to get the `x` values.
    xp1, xp2 : scalar
        The two x data points defining the straight line.
    fp1, fp2 : scalar
        The two y data points defining the straight line.

    Returns
    -------
    x : scalar or array_like
        The `x` values that belong to the given `y` values.

    See Also
    --------
    :func:`numpy.interp` :
        1-dimensional linear interpolation for monotonically increasing
        sample points
    """
    slope = (fp2 - fp1) / (xp2 - xp1)
    intercept = fp1 - slope * xp1
    return (y - intercept) / slope


def exp_law(x, m, c):
    """
    Exponential law.

    Calculate the y values that belong to the given x values for a
    exponential law with the a given exponent-pre-factor `m` and
    pre-factor `c`.

    Parameters
    ----------
    x : scalar or array_like
        x values.
    m : scalar or array_like
        Exponent pre-factor.
    c : scalar or array_like
        Pre-factor.

    Returns
    -------
    y : scalar or numpy.ndarray
        ``y = c * exp(m * x)``.

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    return np.multiply(c, np.exp(np.multiply(m, x)))


def power_law(x, m, c):
    """
    Power law.

    Calculate the y values that belong to the given x values for a power
    law with the a given exponent `m` and pre-factor `c`.

    Parameters
    ----------
    x : scalar or array_like
        x values.
    m : scalar or array_like
        Exponent.
    c : scalar or array_like
        Pre-factor.

    Returns
    -------
    y : scalar or numpy.ndarray
        ``y = c * x**m``.

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    return np.multiply(c, np.power(x, m))


def power_spectrum(data, dt):
    """
    Calculate the power spectrum of the data.

    Parameters
    ----------
    data : array_like
        The data.
    dt : float
        Time step between recorded data points.

    Returns
    -------
    frequencies : numpy.ndarray
        The frequency values.
    amplitudes : numpy.ndarray
        The corresponding power spectrum.
    """
    data = np.asarray(data)
    # The zero-frequency term is the sum of the signal => Remove the
    # mean to get a zero-frequency term that is zero.
    mean = np.mean(data)
    amplitudes = np.abs(np.fft.rfft(data - mean))
    amplitudes **= 2
    frequencies = np.fft.rfftfreq(len(data), dt)
    return frequencies, amplitudes


def gen_equidist_bins(start, stop, bw_desired):
    """
    Generate equidistant bins.

    Parameters
    ----------
    start, stop : float
        First and last bin edge.
    bw_desired : float
        Desired bin width.

    Returns
    -------
    bin_edges : numpy.ndarray
        Bin edges including `start` and `stop`.
    bw_actual : float
        The actual bin width.

    See Also
    --------
    :func:`numpy.linspace` :
        Return evenly spaced numbers over a specified interval

    Notes
    -----
    `start` and `stop` is kept fixed while `bw_desired` is adjusted such
    that ``round((stop - start) / bw_desired)`` bins are created.  The
    number of bin edges is the number of bins plus one.
    """
    if stop <= start:
        raise ValueError(
            "`stop` ({}) must be greater than `start` ({})".format(stop, start)
        )
    if bw_desired <= 0:
        raise ValueError(
            "`bw_desired` ({}) must be greater than zero".format(bw_desired)
        )
    n_bins = round((stop - start) / bw_desired)
    return np.linspace(start, stop, n_bins + 1, retstep=True)


def extend_bins(bins, prepend=None, append=None):
    """
    Extend `bins` with the given prepend and append values.

    Parameters
    ----------
    bins : array_like
        1D array of sorted bin edges.
    prepend : scalar or array_like or None
        (Sorted) bin edge(s) to prepend to `bins`.
    append : scalar or array_like or None
        (Sorted) bin edge(s) to append to `bins`.

    Returns
    -------
    bins_ext : numpy.ndarray
        Extended 1D array of bin edges.
    """
    bins = np.asarray(bins)
    if not np.allclose(np.sort(bins), bins, rtol=0):
        raise ValueError("`bins` ({}) must be a sorted 1D array".format(bins))

    if prepend is not None:
        prepend = np.asarray(prepend)
        if prepend.ndim > 0 and not np.allclose(
            np.sort(prepend), prepend, rtol=0
        ):
            raise ValueError(
                "`prepend` ({}) must be a scalar or a sorted 1D"
                " array".format(prepend)
            )
        if np.any(prepend >= bins[0]):
            raise ValueError(
                "`prepend` ({}) must be smaller than `bins[0]`"
                " ({})".format(prepend, bins[0])
            )
        bins = np.insert(bins, 0, prepend)

    if append is not None:
        append = np.asarray(append)
        if append.ndim > 0 and not np.allclose(
            np.sort(append), append, rtol=0
        ):
            raise ValueError(
                "`append` ({}) must be a scalar or a sorted 1D"
                " array".format(append)
            )
        if np.any(append <= bins[-1]):
            raise ValueError(
                "`append` ({}) must be greater than `bins[-1]`"
                " ({})".format(append, bins[-1])
            )
        bins = np.append(bins, append)

    return bins


def fit_goodness(data, fit):
    """
    Calculate quantities to assess the goodness of a fit.

    Parameters
    ----------
    data : array_like
        The true data.
    fit : array_like
        Array of the same shape as `data` containing the corresponding
        fit/model values.

    Returns
    -------
    r2 : scalar
        Coefficient of determination.
    rmse : scalar
        The root-mean-square error, also known as root-mean-square
        residuals.
    """
    data = np.asarray(data)
    fit = np.asarray(fit)
    # Residual sum of squares.
    ss_res = np.sum((data - fit) ** 2)
    # Root-mean-square error / root-mean-square residuals.
    rmse = np.sqrt(ss_res / len(fit))
    # Total sum of squares.
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    # (Pseudo) coefficient of determination (R^2).
    # https://www.r-bloggers.com/2021/03/the-r-squared-and-nonlinear-regression-a-difficult-marriage/
    r2 = 1 - (ss_res / ss_tot)
    return r2, rmse


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


def qdens2field(x, qdens, bulk_region=None, tol=0.005):
    r"""
    Calculate the electric field resulting from a given charge-density
    profile.

    Parameters
    ----------
    x : array_like
        x values in nm at which the charge-density profile was sampled.
    qdens : array_like
        Charge-density profile in e/nm³.
    bulk_region : None or 2-tuple of floats, optional
        Start and end of the bulk region in units of `x`.  If provided,
        the electric field will be shifted such that the bulk region is
        field-free.
    tol : float, optional
        Tolerance value for finding the indices of `x` that correspond
        to the values provided by `bulk_region`.  Is ignored, if
        `bulk_region` is ``None``.  This function uses
        :func:`lintf2_ether_ana_postproc.misc.find_nearest` to find the
        indices of the bulk region.  Good values for `tol` are 0.005 if
        `x` is given in nanometers or 0.05 if `x` is given in Angstroms.

    Returns
    -------
    field : numpy.ndarray
        The electric field :math:`\epsilon_r E(z)` in V/nm.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.mise.qdens2pot` :
        Calculate the electric potential resulting from a given
        charge-density profile

    Notes
    -----
    The electric field :math:`E(z)` is calculated from the given charge
    density :math:`\rho(z)` based on Poisson's equation

    .. math::

        \frac{\text{d}^2 \phi(z)}{\text{d}z^2} =
        -\frac{\rho(z)}{\epsilon_r \epsilon_0}

    where :math:`\epsilon_r` is the relative permittivity of the medium
    and :math:`\epsilon_0` is the vacuum permittivity.

    Given a charge density :math:`\rho(z)`, the resulting electric field
    can be calculated by:

    .. math::

        E(z) = -\frac{\text{d} \phi(z)}{\text{d}z}
        = -\int{\frac{\text{d}^2 \phi(z)}{\text{d}z^2} \text{ d}z} - c
        = \frac{1}{\epsilon_r \epsilon_0} \int{\rho(z) \text{ d}z} - c

    The integration constant :math:`c` is determined by the boundary
    conditions.  If `bulk_region` is provided, :math:`c` is set to the
    average electric field in the bulk region.  Thus, the electric field
    in the bulk region is effectively set to zero.  If `bulk_region` is
    not provided, :math:`c` is set to zero.

    Note that this function returns :math:`\epsilon_r E(z)` rather than
    :math:`E(z)`.
    """
    # Vacuum permittivity in e/(V*nm).
    eps_0 = constants.epsilon_0 / (constants.e * 1e9)
    field = cumulative_trapezoid(y=qdens, x=x, initial=0)
    field /= eps_0
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
        start, stop = leap.misc.find_nearest(x, bulk_region, tol=tol)
        field_bulk = np.mean(field[start : stop + 1])
        field -= field_bulk
    return field


def qdens2pot(x, qdens, bulk_region=None, tol=0.005, return_field=False):
    r"""
    Calculate the electric potential resulting from a given
    charge-density profile.

    Parameters
    ----------
    x : array_like
        x values in nm at which the charge-density profile was sampled.
    qdens : array_like
        Charge-density profile in e/nm³.
    bulk_region : None or 2-tuple of floats, optional
        Start and end of the bulk region in units of `x`.  If provided,
        the electric field and the electric potential will be shifted
        such that they are zero in the bulk region.
    tol : float, optional
        Tolerance value for finding the indices of `x` that correspond
        to the values provided by `bulk_region`.  Is ignored, if
        `bulk_region` is ``None``.  This function uses
        :func:`lintf2_ether_ana_postproc.misc.find_nearest` to find the
        indices of the bulk region.  Good values for `tol` are 0.005 if
        `x` is given in nanometers or 0.05 if `x` is given in Angstroms.
    return_field : bool, optional
        If ``True``, return the electric field.  The electric field is
        the integral of the charge density and is thus calculated as an
        intermediate step during the computation of the electric
        potential.

    Returns
    -------
    pot : numpy.ndarray
        The electric potential :math:`\epsilon_r \phi(z)` in V.
    field : numpy.ndarray
        The electric field :math:`\epsilon_r E(z)` in V/nm.  Only
        returned if `return_field` is ``True``.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.mise.qdens2field` :
        Calculate the electric field resulting from a given
        charge-density profile

    Notes
    -----
    The electric potential :math:`\phi(z)` is calculated from the given
    charge density :math:`\rho(z)` based on Poisson's equation

    .. math::

        \frac{\text{d}^2 \phi(z)}{\text{d}z^2} =
        -\frac{\rho(z)}{\epsilon_r \epsilon_0}

    where :math:`\epsilon_r` is the relative permittivity of the medium
    and :math:`\epsilon_0` is the vacuum permittivity.

    Given a charge density :math:`\rho(z)`, the resulting electric
    potential can be calculated by:

    .. math::

        \phi(z)
        = \int{
            \int{
                \frac{\text{d}^2 \phi(z)}{\text{d}z^2}
            \text{ d}z} + c
        \text{ d}z} - k
        = \int{
            \int{
                -\frac{\rho(z)}{\epsilon_r \epsilon_0}
            \text{ d}z} + c
        \text{ d}z} - k
        = -\frac{1}{\epsilon_r \epsilon_0}
        \int{\int{\rho(z) \text{ d}z} \text{ d}z}
        + cz - k

    The integration constants :math:`c` and :math:`k` are determined by
    the boundary conditions.  If `bulk_region` is provided, :math:`c` is
    set to the average electric field in the bulk region.  Thus, the
    electric field in the bulk region is effectively set to zero.
    Likewise, :math:`k` is set to the average electric potential in the
    bulk region.  If `bulk_region` is not provided, :math:`c` and
    :math:`k` are set to zero.

    Note that this function returns :math:`\epsilon_r \phi(z)` rather
    than :math:`\phi(z)`.
    """
    field = leap.misc.qdens2field(x, qdens, bulk_region, tol)
    pot = -cumulative_trapezoid(y=field, x=x, initial=0)
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
        start, stop = leap.misc.find_nearest(x, bulk_region, tol=tol)
        pot_bulk = np.mean(pot[start : stop + 1])
        pot -= pot_bulk
    return pot


def dens2free_energy(x, dens, bulk_region=None, tol=0.005):
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
    tol : float, optional
        Tolerance value for finding the indices of `x` that correspond
        to the values provided by `bulk_region`.  Is ignored, if
        `bulk_region` is ``None``.  This function uses
        :func:`lintf2_ether_ana_postproc.misc.find_nearest` to find the
        indices of the bulk region.  Good values for `tol` are 0.005 if
        `x` is given in nanometers or 0.05 if `x` is given in Angstroms.

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
    temperature.  If `bulk_region` is given, :math:`\rho^\circ` is set
    to the average density in the bulk region (i.e. the free energy in
    the bulk region is effectively set to zero).  If `bulk_region` is
    None, :math:`\rho^\circ` is set to one.
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
        start, stop = leap.misc.find_nearest(x, bulk_region, tol=tol)
        free_en_bulk = np.mean(free_en[start : stop + 1])
        free_en -= free_en_bulk
    return free_en


def rdf2free_energy(x, rdf, bulk_start=None, tol=0.005):
    r"""
    Calculate the potential of mean force (PMF, free-energy profile)
    from a radial distribution function (RDF).

    Parameters
    ----------
    x : array_like
        x values at which the RDF was sampled.
    rdf : array_like
        RDF values.
    bulk_start : None or 2-tuple of floats, optional
        Start of the bulk region, i.e. the region where the RDF is one,
        in units of `x`.  If provided, the PMF will be shifted such that
        it is zero in the bulk region.
    tol : float, optional
        Tolerance value for finding the index of `x` that corresponds to
        the value provided by `bulk_start`.  Is ignored, if `bulk_start`
        is ``None``.  This function uses
        :func:`lintf2_ether_ana_postproc.misc.find_nearest` to find the
        index of the bulk region.  Good values for `tol` are 0.005 if
        `x` is given in nanometers or 0.05 if `x` is given in Angstroms.

    Returns
    -------
    free_en : numpy.ndarray
        Free-energy profile :math:`F(r)` in units of :math:`k_B T`, i.e.
        :math:`\frac{F(r)}{k_B T}`.

    Notes
    -----
    The free-energy profile :math:`F(r)` is calculated from the RDF
    :math:`g(r)` according to

    .. math::

        \frac{F(r)}{k_B T} = -\ln\left[ \frac{g(r)}{g^\circ} \right]

    Here, :math:`k_B` is the Boltzmann constant and :math:`T` is the
    temperature.  If `bulk_start` is given, :math:`g^\circ` is set to
    the average value of the RDF in the bulk region (i.e. the free
    energy in the bulk region is effectively set to zero).  If
    `bulk_start` is None, :math:`g^\circ` is set to one.
    """
    free_en = -np.log(rdf)
    if bulk_start is not None:
        start = leap.misc.find_nearest(x, [bulk_start], tol=tol)[0]
        free_en_bulk = np.mean(free_en[start:])
        free_en -= free_en_bulk
    return free_en


def free_energy_barriers(
    minima, maxima, pkp_col_ix, pkh_col_ix, thresh=0, absolute_pos=False
):
    """
    Calculate the barrier heights between the minima and maxima of a
    free-energy profile.

    Parameters
    ----------
    minima, maxima : list
        3-dimensional list of minima/maxima data as returned by
        :func:`lintf2_ether_ana_postproc.simulation.read_free_energy_extrema`.
        The first index must address the columns read from the output
        file of
        :file:`scripts/gmx/density-z/get_free-energy_extrema.py`.  The
        second index must address the peak-position type, i.e. whether
        the peak is in the left or right half of the simulation box.
        The third index must address the simulation.
    pkp_col_ix : int
        The index for the first dimension of `minima`/`maxima` that
        returns the peak positions.  The positions of minima and maxima
        must be alternating.
    pkh_col_ix : int
        The index for the first dimension of `minima`/`maxima`  that
        returns the peak heights.
    thresh : float, optional
        Remove free-energy barriers from the final output that are not
        greater than the given threshold.  Set to ``None`` to not remove
        any free-energy barriers.
    absolute_pos : bool, optional
        If ``True``, assume that the peak minima/maxima positions are
        given as absolute coordinates rather than as distance to the
        electrode as returned by
        :func:`lintf2_ether_ana_postproc.simulation.read_free_energy_extrema`.

    Returns
    -------
    barriers_l2r : list
        Same list as `maxima`, but with the maxima heights (index
        `pkh_col_ix`) replaced by the barrier heights as they appear
        when moving from left to right.
    barriers_r2l : list
        Same list as `maxima`, but with the maxima heights (index
        `pkh_col_ix`) replaced by the barrier heights as they appear
        when moving from right to left.
    """
    try:
        n_pkp_types_minima = len(minima[0])
        n_pkp_types_maxima = len(maxima[0])
        n_pkp_types = n_pkp_types_maxima
    except (TypeError, IndexError):
        raise TypeError("`minima` and `maxima` must be a 3-dimensional lists")
    if n_pkp_types_minima != n_pkp_types_maxima:
        raise ValueError(
            "The length of the second dimension of `minima` ({}) and `maxima`"
            " ({}) must be the"
            " same".format(n_pkp_types_minima, n_pkp_types_maxima)
        )
    try:
        n_sims_minima = len(minima[0][0])
        n_sims_maxima = len(maxima[0][0])
        n_sims = n_sims_maxima
    except (TypeError, IndexError):
        raise TypeError("`minima` and `maxima` must be a 3-dimensional lists")
    if n_sims_minima != n_sims_maxima:
        raise ValueError(
            "The length of the third dimension of `minima` ({}) and `maxima`"
            " ({}) must be the same".format(n_sims_minima, n_sims_maxima)
        )

    try:
        pk_pos_minima = minima[pkp_col_ix]
        pk_pos_maxima = maxima[pkp_col_ix]
    except IndexError:
        raise IndexError(
            "`pkp_col_ix` ({}) is out of bounds for axis 0 of `minima` (size"
            " {}) or `maxima` (size"
            " {})".format(pkp_col_ix, len(minima), len(maxima))
        )
    try:
        pk_height_minima = minima[pkh_col_ix]
        pk_height_maxima = maxima[pkh_col_ix]
    except IndexError:
        raise IndexError(
            "`pkh_col_ix` ({}) is out of bounds for axis 0 of `minima` (size"
            " {}) or `maxima` (size"
            " {})".format(pkh_col_ix, len(minima), len(maxima))
        )

    if absolute_pos:
        pk_pos_types = ("left", "right")  # Peak at left or right electrode.
    else:
        # Peak positions are given as distance to the electrode and
        # therefore peaks at the right electrode behave as peaks at the
        # left electrode.
        pk_pos_types = ("left", "left")
    if n_pkp_types != len(pk_pos_types):
        raise TypeError(
            "`minima` and `maxima` must contain values for {} peak-position"
            " types but contain values for {} peak-position"
            " type(s)".format(len(pk_pos_types), n_pkp_types)
        )

    barriers_l2r = deepcopy(maxima)
    barriers_r2l = deepcopy(maxima)
    if thresh is not None:
        bars_l2r_valid = [
            [None for pkh_sim in pkh_pkt] for pkh_pkt in pk_height_maxima
        ]
        bars_r2l_valid = deepcopy(bars_l2r_valid)

    for pkt_ix, pkp_type in enumerate(pk_pos_types):
        for sim_ix in range(n_sims):
            pkp_minima = pk_pos_minima[pkt_ix][sim_ix]
            pkp_maxima = pk_pos_maxima[pkt_ix][sim_ix]
            pkh_minima = pk_height_minima[pkt_ix][sim_ix]
            pkh_maxima = pk_height_maxima[pkt_ix][sim_ix]

            if len(pkp_minima) != len(pkp_maxima):
                raise ValueError(
                    "Simulation: '{}'.\n"
                    "Peak-position type: {}\n"
                    "The number of minima ({}) does not match the number of"
                    " maxima ({})".format(
                        sim_ix, pkp_type, len(pkp_minima), len(pkp_maxima)
                    )
                )
            if pkp_type == "left":
                if np.any(pkp_minima >= pkp_maxima):
                    raise ValueError(
                        "Simulation: '{}'.\n"
                        "Peak-position type: {}\n"
                        "Either the first extremum is not a minium or minima"
                        " and maxima are not ordered alternately.  Minima: {}."
                        "  Maxima:"
                        " {}".format(sim_ix, pkp_type, pkp_minima, pkp_maxima)
                    )
                bars_l2r = pkh_maxima - pkh_minima
                barriers_l2r[pkh_col_ix][pkt_ix][sim_ix] = bars_l2r
                bars_r2l = pkh_maxima[:-1] - pkh_minima[1:]
                bars_r2l = np.append(bars_r2l, pkh_maxima[-1] - 0)
                barriers_r2l[pkh_col_ix][pkt_ix][sim_ix] = bars_r2l
                if thresh is not None:
                    bars_l2r_valid[pkt_ix][sim_ix] = bars_l2r > thresh
                    bars_r2l_valid[pkt_ix][sim_ix] = bars_r2l > thresh
            elif pkp_type == "right":
                if np.any(pkp_minima <= pkp_maxima):
                    raise ValueError(
                        "Simulation: '{}'.\n"
                        "Peak-position type: {}\n"
                        "Either the first extremum is not a maximum or minima"
                        " and maxima are not ordered alternately.  Minima: {}."
                        "  Maxima:"
                        " {}".format(sim_ix, pkp_type, pkp_minima, pkp_maxima)
                    )
                bars_l2r = pkh_maxima[1:] - pkh_minima[:-1]
                bars_l2r = np.insert(bars_l2r, 0, pkh_maxima[0] - 0)
                barriers_l2r[pkh_col_ix][pkt_ix][sim_ix] = bars_l2r
                bars_r2l = pkh_maxima - pkh_minima
                barriers_r2l[pkh_col_ix][pkt_ix][sim_ix] = bars_r2l
                if thresh is not None:
                    bars_l2r_valid[pkt_ix][sim_ix] = bars_l2r > thresh
                    bars_r2l_valid[pkt_ix][sim_ix] = bars_r2l > thresh
            else:
                raise ValueError(
                    "Unknown peak-position type: '{}'".format(pkp_type)
                )

    if thresh is not None:
        for col_ix, bars_l2r_col in enumerate(barriers_l2r):
            for pkt_ix, bars_l2r_pkt in enumerate(bars_l2r_col):
                for sim_ix, bars_l2r_sim in enumerate(bars_l2r_pkt):
                    valid_l2r = bars_l2r_valid[pkt_ix][sim_ix]
                    barriers_l2r[col_ix][pkt_ix][sim_ix] = bars_l2r_sim[
                        valid_l2r
                    ]
                    bars_r2l_sim = barriers_r2l[col_ix][pkt_ix][sim_ix]
                    valid_r2l = bars_r2l_valid[pkt_ix][sim_ix]
                    barriers_r2l[col_ix][pkt_ix][sim_ix] = bars_r2l_sim[
                        valid_r2l
                    ]

    return barriers_l2r, barriers_r2l


def interp_invalid(x, y, invalid, inplace=True):
    """
    Replace invalid values in `y` with linearly interpolated values.

    Parameters
    ----------
    x, y : array_like
        1-dimensional input arrays containing the x- and y-coordinates
        of the data points.  `x` must be monotonically increasing.  `x`
        and `y` must not contain NaN values.
    invalid : array_like
        Boolean array of the same shape as `x` and `y` indicating
        invalid `y` values.
    inplace : bool, optional
        If ``True``, change `y` in place (therefore, `y` must be a
        :class:`numpy.ndarray`).

    Returns
    -------
    y_interp : numpy.ndarray
        y-coordinates where invalid values are replaced by interpolated
        values.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.misc.interp_outliers` :
        Find outliers and replace them with linearly interpolated values
    """
    x = np.asarray(x)
    y = np.array(y, copy=not inplace)
    invalid = np.asarray(invalid)
    valid = ~invalid
    if y.shape != x.shape:
        raise ValueError(
            "`y` ({}) must have the same shape as `x`"
            " ({})".format(y.shape, x.shape)
        )
    if invalid.shape != y.shape:
        raise ValueError(
            "`invalid` ({}) must have the same shape as `y`"
            " ({})".format(invalid.shape, y.shape)
        )
    if np.any(np.isnan(y[valid])):
        # `scipy.signal.find_peaks` cannot handle NaN values.
        raise ValueError("`y[~invalid]` must not contain NaN values")
    if np.any(np.isnan(x)):
        # `numpy.interp` cannot handle NaN values.
        raise ValueError("`x` must not contain NaN values")
    if np.any(np.diff(x) <= 0):
        # `x` must be monotonically increasing for `numpy.interp`.
        raise ValueError("`x` must be monotonically increasing")

    if np.all(invalid):
        raise ValueError("All y data are marked invalid")
    elif np.any(invalid):
        y[invalid] = np.interp(x[invalid], x[valid], y[valid])
    else:  # all valid
        y = y
    return y


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

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.misc.interp_invalid` :
        Replace invalid values in `y` with linearly interpolated values

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
