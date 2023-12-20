"""Functions related to lifetime analyses."""

# Standard libraries
import warnings

# Third-party libraries
import mdtools as mdt
import numpy as np
from scipy import stats

# First-party libraries
import lintf2_ether_ana_postproc as leap


def dist_characs(a, axis=-1, n_moms=4):
    """
    Estimate distribution characteristics from a given sample.

    Parameters
    ----------
    a : array_like
        Array of samples.
    axis : int, optional
        The axis along which to compute the distribution
        characteristics.
    n_moms : int, optional
        Number of raw moments to calculate.

    Returns
    -------
    characs : numpy.ndarray
        Array of shape ``(11 + n_moms-1, )`` containing the following
        distribution characteristics:

            1. Sample mean (unbiased 1st raw moment)
            2. Uncertainty of the sample mean (standard error)
            3. Corrected sample standard deviation
            4. Corrected coefficient of variation
            5. Unbiased sample skewness (Fisher)
            6. Unbiased sample excess kurtosis (according to Fisher)
            7. Sample median
            8. Non-parametric skewness
            9. 2nd raw moment (biased estimate)
            10. 3rd raw moment (biased estimate)
            11. 4th raw moment (biased estimate)
            12. Sample minimum
            13. Sample maximum
            14. Number of samples

        The number of calculated raw moments depends on `n_moms`.  The
        first raw moment (mean) is always calculated.
    """
    a = np.asarray(a)
    nobs, min_max, mean, var, skew, kurt = stats.describe(
        a, axis=axis, ddof=1, bias=False
    )
    std = np.sqrt(var)
    cv = std / mean
    median = np.median(a, axis=axis)
    skew_non_param = np.divide((mean - median), std)
    raw_moments = [np.mean(a**n) for n in range(2, n_moms + 1)]
    characs = np.array(
        [
            mean,  # Sample mean.
            np.divide(std, np.sqrt(nobs)),  # Uncertainty of sample mean
            std,  # Corrected sample standard deviation.
            cv,  # Corrected coefficient of variation.
            skew,  # Unbiased sample skewness (Fisher).
            kurt,  # Unbiased sample excess kurtosis (Fisher).
            median,  # Median of the sample.
            skew_non_param,  # Non-parametric skewness.
            *raw_moments,  # 2nd to 4th raw moment (biased).
            *min_max,  # Minimum and maximum value of the sample.
            nobs,  # Number of observations (sample points).
        ]
    )
    return characs


def count_method_state_average(dtrj, n_moms=4, time_conv=1, **kwargs):
    """
    Estimate characteristics of the underlying lifetime distribution
    from a sample of lifetimes.

    Take a discrete trajectory and count the number of frames that a
    given compound stays in a given state.  Estimate characteristics of
    the underlying lifetime distribution from the obtained sample.

    The difference to
    :func:`lintf2_ether_ana_postproc.lifetimes.count_method` is that
    this function returns a mean lifetime averaged over all states.

    Parameters
    ----------
    dtrj : array_like
        The discrete trajectory.  Array of shape ``(n, f)``, where ``n``
        is the number of compounds and ``f`` is the number of frames.
        The shape can also be ``(f,)``, in which case the array is
        expanded to shape ``(1, f)``.   The elements of `dtrj` are
        interpreted as the indices of the states in which a given
        compound is at a given frame.
    n_moms : int, optional
        Number of raw moments to calculate.
    time_conv : scalar, optional
        Time conversion factor.  All lifetimes are multiplied by this
        factor.
    kwargs : dict, optional
        Keyword arguments to parse to :func:`mdtools.dtrj.lifetimes`.
        See there for possible choices.  Not allowed are keyword
        arguments that change the number of return values of
        :func:`mdtools.dtrj.lifetimes`.

    Returns
    -------
    characs : numpy.ndarray
        Estimated distribution characteristics.  See
        :func:`lintf2_ether_ana_postproc.lifetimes.dist_characs` for
        more details.
    """
    lts = mdt.dtrj.lifetimes(dtrj, **kwargs)
    lts = lts * time_conv
    characs = leap.lifetimes.dist_characs(lts, n_moms=n_moms)
    return characs


def count_method(
    dtrj, uncensored=False, n_moms=4, time_conv=1, states_check=None
):
    """
    Estimate characteristics of the underlying lifetime distribution
    from a sample of lifetimes.

    Take a discrete trajectory and count the number of frames that a
    given compound stays in a given state.  Estimate characteristics of
    the underlying lifetime distribution from the obtained sample.

    The difference to
    :func:`lintf2_ether_ana_postproc.lifetimes.count_method_state_average`
    is that this function returns a mean lifetime for each individual
    state.

    Parameters
    ----------
    dtrj : array_like
        The discrete trajectory.  Array of shape ``(n, f)``, where ``n``
        is the number of compounds and ``f`` is the number of frames.
        The shape can also be ``(f,)``, in which case the array is
        expanded to shape ``(1, f)``.   The elements of `dtrj` are
        interpreted as the indices of the states in which a given
        compound is at a given frame.
    uncensored : bool, optional
        If ``True`` only take into account uncensored states, i.e.
        states whose start and end lie within the trajectory.  In other
        words, discard the truncated (censored) states at the beginning
        and end of the trajectory.  For these states the start/end time
        is unknown.
    n_moms : int, optional
        Number of raw moments to calculate.
    time_conv : scalar, optional
        Time conversion factor.  All lifetimes are multiplied by this
        factor.
    states_check : array_like or None, optional
        Expected state indices.  If provided, the state indices in the
        discrete trajectory are checked against the provided state
        indices.

    Returns
    -------
    characs : numpy.ndarray
        Estimated distribution characteristics.  See
        :func:`lintf2_ether_ana_postproc.lifetimes.dist_characs` for
        more details.
    states : numpy.ndarray
        The state indices.

    Raises
    ------
    ValueError :
        If the state indices in the given discrete trajectory are not
        contained in the given array of state indices.
    """
    lts_per_state, states = mdt.dtrj.lifetimes_per_state(
        dtrj, uncensored=uncensored, return_states=True
    )
    lts_per_state = [lts * time_conv for lts in lts_per_state]
    if states_check is not None and not np.all(np.isin(states, states_check)):
        raise ValueError(
            "`states` ({}) is not fully contained in `states_check`"
            " ({})".format(states, states_check)
        )
    if states_check is not None:
        n_states = len(states_check)
    else:
        n_states = len(states)
    characs = np.full((n_states, 10 + n_moms), np.nan, dtype=np.float64)
    characs[:, -1] = 0  # Default number of observations.
    for i, lts in enumerate(lts_per_state):
        if len(lts) == 0:
            if not uncensored:
                raise ValueError(
                    "`len(lts) == 0` although `uncensored` is False"
                )
            continue
        else:
            characs[i] = leap.lifetimes.dist_characs(lts, n_moms=n_moms)
    return characs, states


def cross(y, x, f):
    r"""
    Return the `x` value where the array `f` falls below the given `y`
    value for the first time.

    If `f` never falls below the given `y` value, ``numpy.nan`` is
    returned.

    If `f` falls immediately below the given `y` value, ``0`` is
    returned.

    Parameters
    ----------
    y : scalar
        The `y` value for which to get the `x` value.
    x, f : array_like
        The `x` values and corresponding `f` values.

    Returns
    -------
    x_of_y : scalar
        The `x` value that belongs to the given `y` value.
    """
    ix_y = np.nanargmax(f <= y)
    if f[ix_y] > y:
        # `f` never falls below the given `y` value.
        return np.nan
    elif ix_y < 1:
        # `f` falls immediately below the given `y` value.
        return 0
    elif f[ix_y] == y:
        return x[ix_y]
    else:
        # Linearly interpolate between `f[ix_y]` and `f[ix_y - 1]` to
        # estimate the `x` value that belongs to the given `y` value.
        slope = f[ix_y] - f[ix_y - 1]
        slope /= x[ix_y] - x[ix_y - 1]
        intercept = f[ix_y] - slope * x[ix_y]
        return (y - intercept) / slope


def raw_moment_integrate(sf, x, n=1):
    r"""
    Calculate the :math:`n`-th raw moment through numerical integration
    of the survival function using of the alternative expectation
    formula.
    [1]_

    .. math::

        \langle x^n \rangle =
        n \int_{-\infty}^\infty x^{n-1} S(x) \text{ d}x

    Here, :math:`S(x)` is the survival function of the probability
    density function of :math:`x`.  The integral is evaluated
    numerically using :func:`numpy.trapz`.

    Parameters
    ----------
    sf : array_like
        Values of the survival function :math:`S(x)`.
    x : array_like
        Corresponding :math:`x` values.
    n : int, optional
        Order of the moment.

    Returns
    -------
    rm_n : float
        The :math:`n`-th raw moment.

    Notes
    -----
    Values were `sf` or `x` are NaN or infinite are removed prior to
    computing the integral.

    References
    ----------
    .. [1] S. Chakraborti, F. Jardim, E. Epprecht,
        `Higher-order moments using the survival function: The
        alternative expectation formula
        <https://doi.org/10.1080/00031305.2017.1356374>`_,
        The American Statistician, 2019, 73, 2, 191-194.
    """
    valid = np.isfinite(x) & np.isfinite(sf)
    if not np.any(valid):
        warnings.warn(
            "No valid values for numerical integration", stacklevel=2
        )
        return np.nan
    if n < 1 or np.any(np.modf(n)[0] != 0):
        raise ValueError(
            "The moment order, `n` ({}), must be a positive integer".format(n)
        )
    integrand = x[valid] ** (n - 1)
    integrand *= sf[valid]
    integral = np.trapz(y=integrand, x=x[valid])
    integral *= n
    return integral


def skewness(mu2, mu3):
    r"""
    Calculate Fisher's skewness of a distribution from the second and
    third central moment.

    .. math::

        \gamma_1 = \frac{\mu_3}{\mu_2^{3/2}}

    Here, :math:`\mu_n = \langle (x - \mu)^n \rangle` is the
    :math:`n`-th central moment.

    Parameters
    ----------
    mu2, mu3 : scalar or array_like
        The second and third central moment.

    Returns
    -------
    skew : scalar or numpy.ndarray
        The skewness of the distribution.

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    return np.divide(mu3, np.power(mu2, 3 / 2))


def kurtosis(mu2, mu4):
    r"""
    Calculate the excess kurtosis (according to Fisher) of a
    distribution from the second and fourth central moment.

    .. math::

        \gamma_2 = \frac{\mu_4}{\mu_2^2} - 3

    Here, :math:`\mu_n = \langle (x - \mu)^n \rangle` is the
    :math:`n`-th central moment.

    Parameters
    ----------
    mu2, mu4 : scalar or array_like
        The second and fourth central moment.

    Returns
    -------
    kurt : scalar or numpy.ndarray
        The excess kurtosis of the distribution.

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    return np.divide(mu4, np.power(mu2, 2)) - 3


def integral_method(surv_funcs, times, n_moms=4, int_thresh=0.01):
    r"""
    Estimate characteristics of the underlying lifetime distribution by
    numerically integrating/evaluating the survival function.

    Take a survival function and calculate the raw moments of the
    underlying lifetime distribution by numerically integrating the
    survival function according to the alternative expectation formula
    (see :func:`raw_moment_integrate` for more details).  The standard
    deviation, skewness and excess kurtosis are calculated from the raw
    moments.  The median is estimated as the lag time at which the
    survival function decays below 0.5.

    Parameters
    ----------
    surv_funcs : array_like
        Array of shape ``(t, s)`` where ``t`` is the number of lag times
        and ``s`` is the number of different states.  The ij-th element
        of `surv_funcs` is the value of the survival function of state
        j after a lag time of i frames.
    times : array_like
        Array of shape ``(t,)`` containing the corresponding lag times.
    n_moms : int, optional
        Number of moments to calculate.  Must be greater than zero.
    int_thresh : float, optional
        Only calculate raw moments by numerical integration if the
        survival function decays below the given threshold.

    Returns
    -------
    characs : numpy.ndarray
        Array of shape ``(7 + n_moms-1, )`` containing the following
        distribution characteristics:

            1. Mean (1st raw moment)
            2. Standard deviation
            3. Coefficient of variation
            4. Skewness (Fisher)
            5. Excess kurtosis (according to Fisher)
            6. Median
            7. Non-parametric skewness
            8. 2nd raw moment
            9. 3rd raw moment
            10. 4th raw moment

        The number of calculated raw moments depends on `n_moms`.  The
        first raw moment (mean) is always calculated.
    """
    surv_funcs = np.asarray(surv_funcs)
    times = np.asarray(times)
    if surv_funcs.ndim != 2:
        raise ValueError(
            "`surv_funcs` must be 2-dimensional but is"
            " {}-dimensional".format(surv_funcs.ndim)
        )
    if times.shape != surv_funcs.shape[:1]:
        raise ValueError(
            "`times.shape` ({}) != `surv_funcs.shape[:1]`"
            " ({})".format(times.shape, surv_funcs.shape[:1])
        )
    if n_moms < 1:
        raise ValueError(
            "`n_moms` ({}) must be greater than zero".format(n_moms)
        )

    n_frames, n_states = surv_funcs.shape
    characs = np.full((n_states, 6 + n_moms), np.nan, dtype=np.float64)
    for i, sf in enumerate(surv_funcs.T):
        raw_moms = np.full(n_moms, np.nan, dtype=np.float64)
        cen_moms = np.full_like(raw_moms, np.nan)
        if np.any(sf <= int_thresh):
            # Only calculate the moments by numerical integration if the
            # survival function decays below the given threshold.
            for n in range(len(raw_moms)):
                raw_moms[n] = leap.lifetimes.raw_moment_integrate(
                    sf=sf, x=times, n=n + 1
                )
                cen_moms[n] = mdt.stats.moment_raw2cen(raw_moms[: n + 1])
        skew = leap.lifetimes.skewness(mu2=cen_moms[1], mu3=cen_moms[2])
        kurt = leap.lifetimes.kurtosis(mu2=cen_moms[1], mu4=cen_moms[3])
        std = np.sqrt(cen_moms[1])
        cv = std / raw_moms[0]
        median = leap.lifetimes.cross(y=0.5, x=times, f=sf)
        skew_non_param = np.divide((raw_moms[0] - median), std)
        characs[i] = np.array(
            [
                raw_moms[0],  # Mean.
                std,  # Standard deviation.
                cv,  # Coefficient of variation.
                skew,  # Skewness (Fisher).
                kurt,  # Excess kurtosis (Fisher).
                median,  # Median
                skew_non_param,  # Non-parametric skewness.
                *raw_moms[1:],  # 2nd to `n_moms`-th raw moment.
            ]
        )
    return characs


def get_fit_region(surv_funcs, times, end_fit=None, stop_fit=0.01):
    """
    Get the start and end points of the region within which to fit the
    survival function.

    Parameters
    ----------
    surv_funcs : array_like
        Array of shape ``(t, s)`` where ``t`` is the number of lag times
        and ``s`` is the number of different states.  The ij-th element
        of `surv_funcs` is the value of the survival function of state
        j after a lag time of i frames.
    times : array_like
        Array of shape ``(t,)`` containing the corresponding lag times.
    end_fit : float or None, optional
        End time for fitting.  If None, the fit ends at 90% of the lag
        times.
    stop_fit : float, optional
        Stop fitting the survival function as soon as it falls below the
        given value.  The fitting is stopped by whatever happens
        earlier: `end_fit` or `stop_fit`.

    Returns
    -------
    fit_start_ix, fit_stop_ix : numpy.ndarray
        1-dimensional arrays containing for each state the index of the
        start (inclusive) and end point (exclusive) of the region to
        fit.
    """
    surv_funcs = np.asarray(surv_funcs)
    times = np.asarray(times)
    if surv_funcs.ndim != 2:
        raise ValueError(
            "`surv_funcs` must be 2-dimensional but is"
            " {}-dimensional".format(surv_funcs.ndim)
        )
    if times.shape != surv_funcs.shape[:1]:
        raise ValueError(
            "`times.shape` ({}) != `surv_funcs.shape[:1]`"
            " ({})".format(times.shape, surv_funcs.shape[:1])
        )

    if end_fit is None:
        end_fit_ix = int(0.9 * len(times))
    else:
        _, end_fit_ix = mdt.nph.find_nearest(times, end_fit, return_index=True)
    end_fit_ix += 1  # Make `end_fit_ix` inclusive.

    fit_start_ix = np.zeros(surv_funcs.shape[1], dtype=np.uint32)  # Inclusive.
    fit_stop_ix = np.zeros(surv_funcs.shape[1], dtype=np.uint32)  # Exclusive.
    for i, sf in enumerate(surv_funcs.T):
        stop_fit_ix = np.nanargmax(sf < stop_fit)
        if sf[stop_fit_ix] >= stop_fit:
            # The remain probability never falls below `stop_fit`.
            stop_fit_ix = len(sf)
        elif stop_fit_ix < 2:
            # The remain probability immediately falls below `stop_fit`.
            stop_fit_ix = 2
        fit_stop_ix[i] = min(end_fit_ix, stop_fit_ix)
    return fit_start_ix, fit_stop_ix


def weibull_fit_method(
    surv_funcs,
    times,
    fit_start,
    fit_stop,
    surv_funcs_var=None,
    n_moms=4,
    fit_method="trf",
):
    r"""
    Estimate characteristics of the underlying lifetime distribution
    assuming a Weibull distribution.

    Take a survival function and fit it with the survival function of a
    Weibull distribution.  Characteristics of the underlying lifetime
    distribution are thus described by characteristics of the fitted
    Weibull distribution.

    Parameters
    ----------
    surv_funcs : array_like
        Array of shape ``(t, s)`` where ``t`` is the number of lag times
        and ``s`` is the number of different states.  The ij-th element
        of `surv_funcs` is the value of the survival function of state
        j after a lag time of i frames.
    times : array_like
        Array of shape ``(t,)`` containing the corresponding lag times.
    fit_start_ix, fit_stop_ix : numpy.ndarray
        1-dimensional arrays containing for each state the index of the
        start (inclusive) and end point (exclusive) of the region to
        fit.
    surv_funcs_var : array_like or None, optional
        Array of the same shape as `surv_funcs` containing the variance
        of the survival function.
    n_moms : int, optional
        Number of moments to calculate.  Must be greater than zero.
    fit_method : str, optional
        Fit method of :func:`scipy.optimize.curve_fit` to use for
        fitting the survival function.  See there for possible options.

    Returns
    -------
    characs : numpy.ndarray
        Array of shape ``(7 + n_moms-1, )`` containing the following
        distribution characteristics:

            1. Mean (1st raw moment)
            2. Standard deviation
            3. Coefficient of variation
            4. Skewness (Fisher)
            5. Excess kurtosis (according to Fisher)
            6. Median
            7. Non-parametric skewness
            8. 2nd raw moment
            9. 3rd raw moment
            10. 4th raw moment

        The number of calculated raw moments depends on `n_moms`.  The
        first raw moment (mean) is always calculated.
    fit_quality : numpy.ndarray
        Array that contains quantities to assess the goodness of a fit.
        See :func:`fit_goodness` for details about the calculated
        quantities.
    popt : numpy.ndarray
        Optimal values for the fit parameters so that the sum of the
        squared residuals is minimized.  The first element of `popt` is
        the optimal :math:`\tau_0` value, the second element is the
        optimal :math:`\beta` value.
    perr : numpy.ndarray
        Standard deviation of the optimal parameters.

    See Also
    --------
    :func:`mdtools.fit.fit_kww` :
        The function that is used to fit the survival functions.

    Notes
    -----
    The probability distribution function of the Weibull distribution is
    given by

    .. math::

        f(t) =
        \frac{\beta}{\tau_0}
        \left( \frac{t}{\tau_0} \right)^{\beta - 1}
        \exp{\left[ \left( -\frac{t}{\tau_0} \right)^\beta \right]}

    It is a special form of the generalized gamma distribution with
    :math:`\delta = \beta`.

    .. math::

        f(t) =
        \frac{1}{\Gamma\left( \frac{\delta}{\beta} \right)}
        \frac{\beta}{\tau_0}
        \left( \frac{t}{\tau_0} \right)^{\delta - 1}
        \exp{\left[ \left( -\frac{t}{\tau_0} \right)^\beta \right]}

    If :math:`\beta = 1`, the Weibull distribution reduces to the
    exponential distribution.

    The survival function of the Weibull distribution is a stretched
    exponential, also known as Kohlrausch or Kohlrausch-Williams-Watts
    (KWW) function:

    .. math::

        S(t) = \exp\left[ -\left(\frac{t}{\tau_0})^\beta \right) \right]

    The :math:`n`-th raw moment of the Weibull distribution is

    .. math::

        \langle t^n \rangle = \tau_0^n \Gamma(1 + \frac{n}{\beta})

    The median of the Weibull distribution is

    .. math::

        t_med = tau_0 * \left[ \len(2) \right]^\frac{1}{\beta}

    The skewness of the Weibull distribution is

    .. math::

        \gamma_1 =
        \frac{
             2 \Gamma^3\left( 1 + \frac{1}{\beta} \right)
            -3 \Gamma  \left( 1 + \frac{1}{\beta} \right)
               \Gamma  \left( 1 + \frac{2}{\beta} \right)
            +  \Gamma  \left( 1 + \frac{3}{\beta} \right)
        }{
            \left[
                 \Gamma  \left( 1 + \frac{2}{\beta} \right)
                -\Gamma^2\left( 1 + \frac{1}{\beta} \right)
            \right]^{3/2}
        }

    The excess kurtosis (according to Fisher) of the Weibull
    distribution is

    .. math::

        \gamma_2 =
        \frac{
             -6 \Gamma^4\left( 1 + \frac{1}{\beta} \right)
            +12 \Gamma^2\left( 1 + \frac{1}{\beta} \right)
                \Gamma  \left( 1 + \frac{2}{\beta} \right)
             -3 \Gamma^2\left( 1 + \frac{2}{\beta} \right)
             -4 \Gamma  \left( 1 + \frac{1}{\beta} \right)
                \Gamma  \left( 1 + \frac{3}{\beta} \right)
             +  \Gamma  \left( 1 + \frac{4}{\beta} \right)
        }{
            \left[
                 \Gamma  \left( 1 + \frac{2}{\beta} \right) -
                -\Gamma^2\left( 1 + \frac{1}{\beta} \right)
            \right]^2
        }

    """
    surv_funcs = np.asarray(surv_funcs)
    times = np.asarray(times)
    fit_start = np.asarray(fit_start)
    fit_stop = np.asarray(fit_stop)
    if surv_funcs.ndim != 2:
        raise ValueError(
            "`surv_funcs` must be 2-dimensional but is"
            " {}-dimensional".format(surv_funcs.ndim)
        )
    if times.shape != surv_funcs.shape[:1]:
        raise ValueError(
            "`times.shape` ({}) != `surv_funcs.shape[:1]`"
            " ({})".format(times.shape, surv_funcs.shape[:1])
        )
    if fit_start.shape != surv_funcs.shape[1:]:
        raise ValueError(
            "`fit_start.shape` ({}) != `surv_funcs.shape[1:]`"
            " ({})".format(fit_start.shape, surv_funcs.shape[1:])
        )
    if fit_stop.shape != fit_start.shape:
        raise ValueError(
            "`fit_stop.shape` ({}) != `fit_start.shape`"
            " ({})".format(fit_stop.shape, fit_start.shape)
        )
    if n_moms < 1:
        raise ValueError(
            "`n_moms` ({}) must be greater than zero".format(n_moms)
        )

    if surv_funcs_var is not None:
        surv_funcs_var = np.asarray(surv_funcs_var)
        if surv_funcs_var.shape != surv_funcs.shape:
            raise ValueError(
                "`surv_funcs_var.shape` ({}) != `surv_funcs.shape`"
                " ({})".format(surv_funcs_var.shape, surv_funcs.shape)
            )

    n_frames, n_states = surv_funcs.shape
    bounds = ([0, 0], [np.inf, np.inf])
    popt = np.full((n_states, 2), np.nan, dtype=np.float64)
    perr = np.full_like(popt, np.nan)
    fit_quality = np.full((n_states, 2), np.nan, dtype=np.float64)
    characs = np.full((n_states, 6 + n_moms), np.nan, dtype=np.float64)
    for i, sf in enumerate(surv_funcs.T):
        # Do the fit.
        times_fit = times[fit_start[i] : fit_stop[i]]
        sf_fit = sf[fit_start[i] : fit_stop[i]]
        if surv_funcs_var is not None:
            sf_sd = np.sqrt(surv_funcs_var[:, i][fit_start[i] : fit_stop[i]])
        else:
            sf_sd = None
        popt[i], perr[i], valid = mdt.func.fit_kww(
            xdata=times_fit,
            ydata=sf_fit,
            ysd=sf_sd,
            return_valid=True,
            bounds=bounds,
            method=fit_method,
        )
        fit = mdt.func.kww(times_fit[valid], *popt[i])
        fit_quality[i] = np.array(
            leap.misc.fit_goodness(data=sf_fit[valid], fit=fit)
        )
        # Calculate distribution characteristics.
        dist = stats.gengamma(
            a=1,  # delta/beta = beta/beta = 1.
            c=popt[i, 1],  # beta.
            loc=0,
            scale=popt[i, 0],  # tau0.
        )
        raw_moms = [dist.moment(n) for n in range(1, n_moms + 1)]
        var, skew, kurt = dist.stats(moments="vsk")
        std = np.sqrt(var)
        cv = std / raw_moms[0]
        median = dist.median()
        skew_non_param = np.divide((raw_moms[0] - median), std)
        characs[i] = np.array(
            [
                raw_moms[0],  # Mean.
                std,  # Standard deviation.
                cv,  # Coefficient of variation.
                skew,  # Skewness (Fisher).
                kurt,  # Excess kurtosis (Fisher).
                median,  # Median
                skew_non_param,  # Non-parametric skewness.
                *raw_moms[1:],  # 2nd to `n_moms`-th raw moment.
            ]
        )
    return characs, fit_quality, popt, perr


def burr12_fit_method(
    surv_funcs,
    times,
    fit_start,
    fit_stop,
    surv_funcs_var=None,
    n_moms=4,
    fit_method="trf",
):
    r"""
    Estimate characteristics of the underlying lifetime distribution
    assuming a Burr Type XII distribution.

    Take a survival function and fit it with the survival function of a
    Burr Type XII distribution.  Characteristics of the underlying
    lifetime distribution are thus described by characteristics of the
    fitted Burr Type XII distribution.

    Parameters
    ----------
    surv_funcs : array_like
        Array of shape ``(t, s)`` where ``t`` is the number of lag times
        and ``s`` is the number of different states.  The ij-th element
        of `surv_funcs` is the value of the survival function of state
        j after a lag time of i frames.
    times : array_like
        Array of shape ``(t,)`` containing the corresponding lag times.
    fit_start_ix, fit_stop_ix : numpy.ndarray
        1-dimensional arrays containing for each state the index of the
        start (inclusive) and end point (exclusive) of the region to
        fit.
    surv_funcs_var : array_like or None, optional
        Array of the same shape as `surv_funcs` containing the variance
        of the survival function.
    n_moms : int, optional
        Number of moments to calculate.  Must be greater than zero.
    fit_method : str, optional
        Fit method of :func:`scipy.optimize.curve_fit` to use for
        fitting the survival function.  See there for possible options.

    Returns
    -------
    characs : numpy.ndarray
        Array of shape ``(7 + n_moms-1, )`` containing the following
        distribution characteristics:

            1. Mean (1st raw moment)
            2. Standard deviation
            3. Coefficient of variation
            4. Skewness (Fisher)
            5. Excess kurtosis (according to Fisher)
            6. Median
            7. Non-parametric skewness
            8. 2nd raw moment
            9. 3rd raw moment
            10. 4th raw moment

        The number of calculated raw moments depends on `n_moms`.  The
        first raw moment (mean) is always calculated.
    fit_quality : numpy.ndarray
        Array that contains quantities to assess the goodness of a fit.
        See :func:`fit_goodness` for details about the calculated
        quantities.
    popt : numpy.ndarray
        Optimal values for the fit parameters so that the sum of the
        squared residuals is minimized.  The first element of `popt` is
        the optimal :math:`\tau_0` value, the second element is the
        optimal :math:`\beta` value, the third value is the optimal
        :math:`d` value, where :math:`d = \beta\delta` (!).
    perr : numpy.ndarray
        Standard deviation of the optimal parameters.
    popt_converted : numpy.ndarray
        Same as `popt`, but instead of :math:`d = \beta\delta`, the
        third value is :math:`\delta`.
    perr_converted : numpy.ndarray
        Standard deviation of `popt_converted`.

    See Also
    --------
    :func:`mdtools.fit.fit_burr12_sf_alt` :
        The function that is used to fit the survival functions.

    Notes
    -----
    The probability distribution function of the Burr Type XII
    distribution is given by

    .. math::

        f(t) =
        \frac{\beta \delta}{\tau_0}
        \left( \frac{t}{\tau_0} \right)^{\beta - 1}
        \frac{
            1
        }{
            \left[
                1 + \left( \frac{t}{\tau_0} \right)^\beta
            \right]^{\delta - 1}
        }

    If :math:`\delta = 1`, the Burr Type XII distribution reduces to the
    log-logistic distribution.  If :math:`\beta = 1`, the Burr Type XII
    distribution reduces to the Lomax distribution.

    The survival function of the Burr Type XII distribution is

    .. math::

        S(t) =
        \frac{
            1
        }{
            \left[
                1 + \left( \frac{t}{\tau_0} \right)^\beta
            \right]^\delta
        }

    For :math:`\beta = 1`, the survival function is also known as
    Becquerel decay.

    The :math:`n`-th raw moment of the Burr Type XII distribution is

    .. math::

        \langle t^n \rangle =
        \tau_0^n
        \frac{
            \Gamma\left( \delta - \frac{n}{\beta} \right)
            \Gamma\left( 1      + \frac{n}{\beta} \right)
        }{
            \Gamma(\delta)
        }

    Note that the n-th raw moment of the Burr Type XII distribution only
    exists if :math:`n < \beta \delta`.

    The median of the Burr Type XII distribution is

    .. math::

        t_med = \frac{1}{\left( 2^{1/\delta} - 1 \right)^(1/\beta)}

    """
    surv_funcs = np.asarray(surv_funcs)
    times = np.asarray(times)
    fit_start = np.asarray(fit_start)
    fit_stop = np.asarray(fit_stop)
    if surv_funcs.ndim != 2:
        raise ValueError(
            "`surv_funcs` must be 2-dimensional but is"
            " {}-dimensional".format(surv_funcs.ndim)
        )
    if times.shape != surv_funcs.shape[:1]:
        raise ValueError(
            "`times.shape` ({}) != `surv_funcs.shape[:1]`"
            " ({})".format(times.shape, surv_funcs.shape[:1])
        )
    if fit_start.shape != surv_funcs.shape[1:]:
        raise ValueError(
            "`fit_start.shape` ({}) != `surv_funcs.shape[1:]`"
            " ({})".format(fit_start.shape, surv_funcs.shape[1:])
        )
    if fit_stop.shape != fit_start.shape:
        raise ValueError(
            "`fit_stop.shape` ({}) != `fit_start.shape`"
            " ({})".format(fit_stop.shape, fit_start.shape)
        )
    if n_moms < 1:
        raise ValueError(
            "`n_moms` ({}) must be greater than zero".format(n_moms)
        )

    if surv_funcs_var is not None:
        surv_funcs_var = np.asarray(surv_funcs_var)
        if surv_funcs_var.shape != surv_funcs.shape:
            raise ValueError(
                "`surv_funcs_var.shape` ({}) != `surv_funcs.shape`"
                " ({})".format(surv_funcs_var.shape, surv_funcs.shape)
            )

    n_frames, n_states = surv_funcs.shape
    bounds = ([0, 0, 1 + 1e-6], [np.inf, np.inf, np.inf])
    popt = np.full((n_states, 3), np.nan, dtype=np.float64)
    perr = np.full_like(popt, np.nan)
    fit_quality = np.full((n_states, 2), np.nan, dtype=np.float64)
    characs = np.full((n_states, 6 + n_moms), np.nan, dtype=np.float64)
    for i, sf in enumerate(surv_funcs.T):
        # Do the fit.
        times_fit = times[fit_start[i] : fit_stop[i]]
        sf_fit = sf[fit_start[i] : fit_stop[i]]
        if surv_funcs_var is not None:
            sf_sd = np.sqrt(surv_funcs_var[:, i][fit_start[i] : fit_stop[i]])
        else:
            sf_sd = None
        popt[i], perr[i], valid = mdt.func.fit_burr12_sf_alt(
            xdata=times_fit,
            ydata=sf_fit,
            ysd=sf_sd,
            return_valid=True,
            bounds=bounds,
            method=fit_method,
        )
        fit = mdt.func.burr12_sf_alt(times_fit[valid], *popt[i])
        fit_quality[i] = np.array(
            leap.misc.fit_goodness(data=sf_fit[valid], fit=fit)
        )
        # Calculate distribution characteristics.
        dist = stats.burr12(
            c=popt[i, 1],  # beta.
            d=popt[i, 2] / popt[i, 1],  # delta.
            loc=0,
            scale=popt[i, 0],  # tau0.
        )
        raw_moms = [dist.moment(n) for n in range(1, n_moms + 1)]
        var, skew, kurt = dist.stats(moments="vsk")
        std = np.sqrt(var)
        cv = std / raw_moms[0]
        median = dist.median()
        skew_non_param = np.divide((raw_moms[0] - median), std)
        characs[i] = np.array(
            [
                raw_moms[0],  # Mean.
                std,  # Standard deviation.
                cv,  # Coefficient of variation.
                skew,  # Skewness (Fisher).
                kurt,  # Excess kurtosis (Fisher).
                median,  # Median
                skew_non_param,  # Non-parametric skewness.
                *raw_moms[1:],  # 2nd to `n_moms`-th raw moment.
            ]
        )

    tau0, beta, d = popt.T
    tau0_sd, beta_sd, d_sd = perr.T
    delta = d / beta
    delta_sd = np.sqrt(  # Propagation of uncertainty.
        delta**2
        * (
            (d_sd / d) ** 2
            + (beta_sd / beta) ** 2
            - 2 * d_sd * beta_sd / (d * beta)
        )
    )
    popt_converted = np.column_stack([tau0, beta, delta])
    perr_converted = np.column_stack([tau0_sd, beta_sd, delta_sd])
    return characs, fit_quality, popt, perr, popt_converted, perr_converted


def histograms(dtrj_file, uncensored=False, intermittency=0, time_conv=1):
    """
    Calculate the lifetime histogram for each state in a discrete
    trajectory.

    Parameters
    ----------
    dtrj_file : str or bytes or os.PathLike
        The filename of the discrete trajectory.
    uncensored : bool, optional
        If ``True`` only take into account uncensored states, i.e.
        states whose start and end lie within the trajectory.  In other
        words, discard the truncated (censored) states at the beginning
        and end of the trajectory.  For these states the start/end time
        is unknown.
    intermittency : int, optional
        Maximum number of frames a compound is allowed to leave its
        state while still being considered to be in this state provided
        that it returns to this state after the given number of frames.
    time_conv : float, optional
        Time conversion factor to convert trajectory steps to a physical
        time unit (like ns).

    Returns
    -------
    hists : numpy.ndarray
        Array of histograms (one for each state in the discrete
        trajectory).
    bins : numpy.ndarray
        Bin edges used to generate the histograms.
    states : numpy.ndarray
        Array containing the state indices.
    """
    # Read discrete trajectory.
    dtrj = mdt.fh.load_dtrj(dtrj_file)
    n_frames = dtrj.shape[-1]

    if intermittency > 0:
        print("Correcting for intermittency...")
        dtrj = mdt.dyn.correct_intermittency(
            dtrj.T, intermittency, inplace=True, verbose=True
        )
        dtrj = dtrj.T

    # Get list of all lifetimes for each state.
    lts_per_state, states = mdt.dtrj.lifetimes_per_state(
        dtrj, uncensored=uncensored, return_states=True
    )
    states = states.astype(np.uint16)
    n_states = len(states)
    del dtrj

    # Calculate lifetime histogram for each state.
    # Binning is done in trajectory steps.
    # Linear bins.
    # step = 1
    # bins = np.arange(1, n_frames, step, dtype=np.float64)
    # Logarithmic bins.
    stop = int(np.ceil(np.log2(n_frames))) + 1
    bins = np.logspace(0, stop, stop + 1, base=2, dtype=np.float64)
    bins -= 0.5
    hists = np.full((n_states, len(bins) - 1), np.nan, dtype=np.float32)
    for state_ix, lts_state in enumerate(lts_per_state):
        if np.any(lts_state < bins[0]) or np.any(lts_state > bins[-1]):
            raise ValueError(
                "At least one lifetime lies outside the binned region"
            )
        hists[state_ix], _bins = np.histogram(
            lts_state, bins=bins, density=True
        )
        if not np.allclose(_bins, bins, rtol=0):
            raise ValueError(
                "`_bins` != `bins`.  This should not have happened"
            )
        if not np.isclose(np.sum(hists[state_ix] * np.diff(bins)), 1):
            raise ValueError(
                "The integral of the histogram ({}) is not close to"
                " one".format(np.sum(hists[state_ix] * np.diff(bins)))
            )
    del lts_per_state, lts_state, _bins
    bins *= time_conv
    return hists, bins.astype(np.float32), states
