#!/usr/bin/env python3


"""
Calculate bin residence times / lifetimes.

For a single simulation, calculate the average time that a given
compound stays in a given bin directly from the discrete trajectory
(Method 1-3) and from the corresponding remain probability function
(Method 4-7).
"""


# Standard libraries
import argparse
import warnings

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.special import gamma

# First-party libraries
import lintf2_ether_ana_postproc as leap


def dist_charac(a, axis=-1):
    """
    Calculate distribution characteristics of a given sample.

    Parameters
    ----------
    a : array_like
        Array of samples.
    axis : int, optional
        The axis along which to compute the distribution
        characteristics.

    Returns
    -------
    charac : numpy.ndarray
        Array of shape ``(9, )`` containing the

            1. Sample mean
            2. Uncertainty of the sample mean (standard error)
            3. Corrected sample standard deviation
            4. Unbiased sample skewness
            5. Unbiased sample excess kurtosis (according to Fisher)
            6. Sample median
            7. Sample minimum
            8. Sample maximum
            9. Number of samples

    """
    a = np.asarray(a)
    nobs, min_max, mean, var, skew, kurt = stats.describe(
        a, axis=axis, ddof=1, bias=False
    )
    median = np.median(a, axis=axis)
    charac = np.array(
        [
            mean,  # Sample mean.
            np.sqrt(np.divide(var, nobs)),  # Uncertainty of sample mean
            np.sqrt(var),  # Corrected sample standard deviation.
            skew,  # Unbiased sample skewness.
            kurt,  # Unbiased sample excess kurtosis (Fisher).
            median,  # Median of the sample.
            min_max[0],  # Minimum value of the sample.
            min_max[1],  # Maximum value of the sample.
            nobs,  # Number of observations (sample points).
        ]
    )
    return charac


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


def raw_moment_weibull(tau0, beta, n=1):
    r"""
    Calculate the :math:`n`-th raw moment of the Weibull distribution.

    .. math::

        \langle t^n \rangle =
        \tau_0^n \Gamma(1 + \frac{n}{\beta})

    Parameters
    ----------
    tau0 : scalar or array_like
        The scale parameter of the Weibull distribution.
    beta : scalar or array_like
        The shape parameter of the Weibull distribution.
    n : int or array of int, optional
        Order of the moment.

    Returns
    -------
    rm_n : scalar or numpy.ndarray
        The :math:`n`-th raw moment.

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    rm_n = np.power(tau0, n)
    rm_n *= gamma(1 + np.divide(n, beta))
    return rm_n


def raw_moment_burr12(tau0, beta, delta, n=1):
    r"""
    Calculate the :math:`n`-th raw moment of the Burr Type XII
    distribution.

    .. math::

        \langle t^n \rangle =
        \tau_0^n
        \frac{
            \Gamma\left( \delta - \frac{n}{\beta} \right)
            \Gamma\left( 1      + \frac{n}{\beta} \right)
        }{
            \Gamma(\delta)
        }

    Parameters
    ----------
    tau0 : scalar or array_like
        The scale parameter of the Burr Type XII distribution.
    beta, delta : scalar or array_like
        The shape parameters of the Burr Type XII distribution.
    n : int or array of int, optional
        Order of the moment.

    Returns
    -------
    rm_n : scalar or numpy.ndarray
        The :math:`n`-th raw moment.

    Notes
    -----
    If more than one input argument is an array, all arrays must be
    broadcastable.
    """
    rm_n = np.power(tau0, n)
    rm_n *= gamma(np.subtract(delta, np.divide(n, beta)))
    rm_n *= gamma(1 + np.divide(n, beta))
    rm_n /= gamma(delta)
    return rm_n


def skewness(mu2, mu3):
    r"""
    Calculate the skewness of a distribution from the second and third
    central moment.

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


def get_ydata_min_max(ax):
    """
    Get the minimum and maximum y value of the data plotted in an
    :class:`matplotlib.axes.Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` from which to get the data.

    Returns
    -------
    yd_min, yd_max : numpy.ndarray
        Array of minimum and maximum values of the y data plotted in the
        given :class:`~matplotlib.axes.Axes`.  Each value in the array
        corresponds to one plotted :class:`matplotlib.lines.Line2D` in
        the :class:`~matplotlib.axes.Axes`.
    """
    ydata = [line.get_ydata() for line in ax.get_lines()]
    yd_min, yd_max = [], []
    for yd in ydata:
        if isinstance(yd, np.ndarray) and np.any(yd > 0):
            yd_min.append(np.min(yd[yd > 0]))
            yd_max.append(np.max(yd[yd > 0]))
    yd_min, yd_max = np.array(yd_min), np.array(yd_max)
    return yd_min, yd_max


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "For a single simulation, calculate the average time that a given"
        " compound stays in a given bin."
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
parser.add_argument(
    "--cmp",
    type=str,
    required=False,
    default="Li",
    choices=("Li",),  # ("Li", "NBT", "OBT", "OE"),
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--continuous",
    required=False,
    default=False,
    action="store_true",
    help="Use the 'continuous' definition of the remain probability function.",
)
parser.add_argument(
    "--int-thresh",
    type=float,
    required=False,
    default=0.01,
    help=(
        "Only calculate the lifetime by directly integrating the remain"
        " probability if the remain probability decayed below the given"
        " threshold.  Default:  %(default)s."
    ),
)
parser.add_argument(
    "--end-fit",
    type=float,
    required=False,
    default=None,
    help=(
        "Last lag time (in ns) to include when fitting the remain probability."
        "  Default:  %(default)s (this means end at 90%% of the lag times)."
    ),
)
parser.add_argument(
    "--stop-fit",
    type=float,
    required=False,
    default=0.01,
    help=(
        "Stop fitting the remain probability as soon as it falls below this"
        " threshold.  The fitting is stopped by whatever happens earlier:"
        " --end-fit or --stop-fit.  Default: %(default)s"
    ),
)
args = parser.parse_args()

if args.continuous:
    con = "_continuous"
else:
    con = ""

analysis = "discrete-z"  # Analysis name.
# Common file suffix of analysis input files.
file_suffix_common = analysis + "_" + args.cmp
tool = "mdt"  # Analysis software.
outfile_base = (
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis
    + "_"
    + args.cmp
    + "_lifetimes"
    + con
)
outfile_txt = outfile_base + ".txt.gz"
outfile_pdf = outfile_base + ".pdf"

# Time conversion factor to convert trajectory steps to ns.
time_conv = 2e-3
# Number of moments to calculate.  For calculating the skewness, the 2nd
# and 3rd (central) moments are required, for the kurtosis the 2nd and
# 4th (central) moments are required.
n_moms = 4
# Fit method of `scipy.optimize.curve_fit` to use for all fits.
fit_method = "trf"


print("Creating Simulation instance(s)...")
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    path_key = "q%g" % surfq
else:
    surfq = None
    path_key = "bulk"
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, path_key)


print("Calculating lifetimes directly from `dtrj`...")
# Read discrete trajectory.
file_suffix = file_suffix_common + "_dtrj.npz"
infile_dtrj = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
dtrj = mdt.fh.load_dtrj(infile_dtrj)
n_frames = dtrj.shape[1]

# Method 1: Calculate the average lifetime by counting the number of
# frames that a given compound stays in a given state including
# truncated states at the trajectory edges -> censored.
lts_cnt_cen, states = mdt.dtrj.lifetimes_per_state(
    dtrj, uncensored=False, return_states=True
)
lts_cnt_cen = [lts * time_conv for lts in lts_cnt_cen]
n_states = len(states)
lts_cnt_cen_characs = np.full((n_states, 9), np.nan, dtype=np.float64)
lts_cnt_cen_characs[:, -1] = 0  # Default number of observations.
for i, lts in enumerate(lts_cnt_cen):
    lts_cnt_cen_characs[i] = dist_charac(lts)
del lts_cnt_cen

# Method 2: Calculate the average lifetime by counting the number of
# frames that a given compound stays in a given state excluding
# truncated states at the trajectory edges -> uncensored.
lts_cnt_unc, states_cnt_unc = mdt.dtrj.lifetimes_per_state(
    dtrj, uncensored=True, return_states=True
)
lts_cnt_unc = [lts * time_conv for lts in lts_cnt_unc]
if not np.all(np.isin(states_cnt_unc, states)):
    raise ValueError(
        "`states_cnt_unc` ({}) is not fully contained in `states`"
        " ({})".format(states_cnt_unc, states)
    )
lts_cnt_unc_characs = np.full((n_states, 9), np.nan, dtype=np.float64)
lts_cnt_unc_characs[:, -1] = 0  # Default number of observations.
for i, lts in enumerate(lts_cnt_unc):
    if len(lts) == 0:
        continue
    else:
        lts_cnt_unc_characs[i] = dist_charac(lts)
del lts_cnt_unc, states_cnt_unc

# Method 3: Calculate the transition rate as the number of transitions
# leading out of a given state divided by the number of frames that
# compounds have spent in this state.  The average lifetime is
# calculated as the inverse transition rate.
rates, states_k = mdt.dtrj.trans_rate_per_state(dtrj, return_states=True)
lts_k = time_conv / rates
if not np.array_equal(states_k, states):
    raise ValueError(
        "`states_k` ({}) != `states` ({})".format(states_k, states)
    )
del dtrj, rates, states_k


print("Calculating lifetimes from the remain probability...")
# Read remain probabilities (one for each bin).
file_suffix = file_suffix_common + "_state_lifetime_discrete" + con + ".txt.gz"
infile_rp = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
remain_props = np.loadtxt(infile_rp)
states_rp = remain_props[0, 1:]  # State indices.
times = remain_props[1:, 0]  # Lag times in trajectory steps.
remain_props = remain_props[1:, 1:]  # Remain probability functions.
if np.any(remain_props < 0) or np.any(remain_props > 1):
    raise ValueError(
        "Some values of the remain probability lie outside the interval [0, 1]"
    )
if not np.array_equal(times, np.arange(n_frames)):
    print("`n_frames` =", n_frames)
    print("`times` =")
    print(times)
    raise ValueError("`times` != `np.arange(n_frames)`")
times *= time_conv  # Trajectory steps -> ns.
if np.any(np.modf(states_rp)[0] != 0):
    raise ValueError(
        "Some state indices are not integers but floats.  `states_rp` ="
        " {}".format(states_rp)
    )
if not np.array_equal(states_rp, states):
    raise ValueError(
        "`states_rp` ({}) != `states` ({})".format(states_rp, states)
    )
del states_rp

# Method 4: Set the lifetime to the lag time at which the remain
# probability crosses 1/e.
lts_e = np.array([cross(y=1 / np.e, x=times, f=rp) for rp in remain_props.T])

# Method 5: Calculate the lifetime as the integral of the remain
# probability p(t).
lts_int_characs = np.full((n_states, 5), np.nan, dtype=np.float64)
for i, rp in enumerate(remain_props.T):
    raw_moms = np.full(n_moms, np.nan, dtype=np.float64)
    cen_moms = np.full(n_moms, np.nan, dtype=np.float64)
    if np.any(rp <= args.int_thresh):
        # Only calculate the (raw) moments by numerically integrating
        # the remain probability if the remain probability falls below
        # the given threshold.
        for n in range(n_moms):
            raw_moms[n] = raw_moment_integrate(sf=rp, x=times, n=n + 1)
            cen_moms[n] = mdt.stats.moment_raw2cen(raw_moms[: n + 1])
    skew = skewness(mu2=cen_moms[1], mu3=cen_moms[2])
    kurt = kurtosis(mu2=cen_moms[1], mu4=cen_moms[3])
    # Estimate of the median assuming that the remain probability is
    # equal to the survival function of the underlying distribution of
    # lifetimes.
    median = cross(y=0.5, x=times, f=rp)
    lts_int_characs[i] = np.array(
        [raw_moms[0], np.sqrt(cen_moms[1]), skew, kurt, median]
    )

# Get fit region for fitting methods.
if args.end_fit is None:
    end_fit = int(0.9 * len(times))
else:
    _, end_fit = mdt.nph.find_nearest(times, args.end_fit, return_index=True)
end_fit += 1  # Make `end_fit` inclusive.
fit_start = np.zeros(n_states, dtype=np.uint32)  # Inclusive.
fit_stop = np.zeros(n_states, dtype=np.uint32)  # Exclusive.
for i, rp in enumerate(remain_props.T):
    stop_fit = np.nanargmax(rp < args.stop_fit)
    if rp[stop_fit] >= args.stop_fit:
        # The remain probability never falls below `args.stop_fit`.
        stop_fit = len(rp)
    elif stop_fit < 2:
        # The remain probability immediately falls below
        # `args.stop_fit`.
        stop_fit = 2
    fit_stop[i] = min(end_fit, stop_fit)

# Method 6: Fit the remain probability with a Kohlrausch function
# stretched exponential) and calculate the lifetime as the integral of
# the fit:
#   I_kww(t) = exp[-(t/tau0_kww)^beta_kww]
#   <t^n> = n * int_0^inf t^(n-1) * I_kww(t) dt
#         = tau0_kww^n * Gamma(1 + n/beta_kww)
bounds_kww = ([0, 0], [np.inf, 10])
popt_kww = np.full((n_states, 2), np.nan, dtype=np.float64)
perr_kww = np.full((n_states, 2), np.nan, dtype=np.float64)
lts_kww_fit_goodness = np.full((n_states, 2), np.nan, dtype=np.float64)
lts_kww_characs = np.full((n_states, 5), np.nan, dtype=np.float64)
for i, rp in enumerate(remain_props.T):
    # Fit remain probability.
    times_fit = times[fit_start[i] : fit_stop[i]]
    rp_fit = rp[fit_start[i] : fit_stop[i]]
    popt_kww[i], perr_kww[i], valid = mdt.func.fit_kww(
        xdata=times_fit,
        ydata=rp_fit,
        return_valid=True,
        bounds=bounds_kww,
        method=fit_method,
    )
    fit = mdt.func.kww(times_fit[valid], *popt_kww[i])
    r2, rmse = fit_goodness(data=rp_fit[valid], fit=fit)
    lts_kww_fit_goodness[i] = np.array([r2, rmse])
    # Calculate distribution characteristics.
    raw_moms = np.full(n_moms, np.nan, dtype=np.float64)
    cen_moms = np.full(n_moms, np.nan, dtype=np.float64)
    for n in range(n_moms):
        raw_moms[n] = raw_moment_weibull(*popt_kww[i], n=n + 1)
        cen_moms[n] = mdt.stats.moment_raw2cen(raw_moms[: n + 1])
    skew = skewness(mu2=cen_moms[1], mu3=cen_moms[2])
    kurt = kurtosis(mu2=cen_moms[1], mu4=cen_moms[3])
    # Median = tau_0 * ln(2)^(1/beta)
    median = popt_kww[i][0] * np.log(2) ** (1 / popt_kww[i][1])
    lts_kww_characs[i] = np.array(
        [raw_moms[0], np.sqrt(cen_moms[1]), skew, kurt, median]
    )
tau0_kww, beta_kww = popt_kww.T
tau0_kww_sd, beta_kww_sd = perr_kww.T

# Method 7: Fit the remain probability with the survival function of a
# Burr Type XII distribution and calculate the lifetime as the integral
# fo the fit:
#   I_bur(t) = 1 / [1 + (t/tau0_bur)^beta_bur]^delta_bur
#   <t^n> = n * int_0^inf t^(n-1) * I_bur(t) dt
#         = tau0_bur^n * Gamma(delta_bur - n/beta_bur) *
#           Gamma(1 + n/beta_bur) / Gamma(delta_bur)
bounds_bur = ([0, 0, 1 + 1e-6], [np.inf, 10, 100])
popt_bur = np.full((n_states, 3), np.nan, dtype=np.float64)
perr_bur = np.full((n_states, 3), np.nan, dtype=np.float64)
lts_bur_fit_goodness = np.full((n_states, 2), np.nan, dtype=np.float64)
lts_bur_characs = np.full((n_states, 5), np.nan, dtype=np.float64)
for i, rp in enumerate(remain_props.T):
    # Fit remain probability.
    times_fit = times[fit_start[i] : fit_stop[i]]
    rp_fit = rp[fit_start[i] : fit_stop[i]]
    popt_bur[i], perr_bur[i], valid = mdt.func.fit_burr12_sf_alt(
        xdata=times_fit,
        ydata=rp_fit,
        return_valid=True,
        bounds=bounds_bur,
        method=fit_method,
    )
    fit = mdt.func.burr12_sf_alt(times_fit[valid], *popt_bur[i])
    r2, rmse = fit_goodness(data=rp_fit[valid], fit=fit)
    lts_bur_fit_goodness[i] = np.array([r2, rmse])
    # Calculate distribution characteristics.
    tau0 = popt_bur[i][0]
    beta = popt_bur[i][1]
    delta = popt_bur[i][2] / beta
    raw_moms = np.full(n_moms, np.nan, dtype=np.float64)
    cen_moms = np.full(n_moms, np.nan, dtype=np.float64)
    for n in range(n_moms):
        raw_moms[n] = raw_moment_burr12(tau0, beta, delta, n=n + 1)
        cen_moms[n] = mdt.stats.moment_raw2cen(raw_moms[: n + 1])
    skew = skewness(mu2=cen_moms[1], mu3=cen_moms[2])
    kurt = kurtosis(mu2=cen_moms[1], mu4=cen_moms[3])
    # Median = tau_0 * (2^(1/delta) - 1)^(1/beta)
    median = tau0 * (2 ** (1 / delta) - 1) ** (1 / beta)
    lts_bur_characs[i] = np.array(
        [raw_moms[0], np.sqrt(cen_moms[1]), skew, kurt, median]
    )
tau0_bur, beta_bur, d_bur = popt_bur.T
tau0_bur_sd, beta_bur_sd, d_bur_sd = perr_bur.T
delta_bur = d_bur / beta_bur
delta_bur_sd = np.sqrt(  # Propagation of uncertainty.
    delta_bur**2
    * (
        (d_bur_sd / d_bur) ** 2
        + (beta_bur_sd / beta_bur) ** 2
        - 2 * d_bur_sd * beta_bur_sd / (d_bur * beta_bur)
    )
)


print("Creating output file(s)...")
# Read bin edges.
file_suffix = file_suffix_common + "_bins" + ".txt.gz"
infile_bins = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
bins = np.loadtxt(infile_bins)

Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK
box_z = Sim.box[2]

bins_low = bins[states]  # Lower bin edges.
bins_up = bins[states + 1]  # Upper bin edges.
# Distance of the bins to the left/right electrode surface.
bins_low_el = bins_low - elctrd_thk
bins_up_el = box_z - elctrd_thk - bins_up

# Create text output.
data = np.column_stack(
    [
        states,  # 1
        bins_low,  # 2
        bins_up,  # 3
        bins_low - elctrd_thk,  # 4
        box_z - elctrd_thk - bins_low,  # 5
        bins_up - elctrd_thk,  # 6
        box_z - elctrd_thk - bins_up,  # 7
        # Method 1 (censored counting).
        lts_cnt_cen_characs,  # 8-16
        # Method 2 (uncensored counting).
        lts_cnt_unc_characs,  # 17-25
        # Method 3 (inverse transition rate).
        lts_k,  # 26
        # Method 4 (1/e criterion).
        lts_e,  # 27
        # Method 5 (direct integral).
        lts_int_characs,  # 28-32
        # Method 6 (integral of Kohlrausch fit).
        lts_kww_characs,  # 33-37
        tau0_kww,  # 38
        tau0_kww_sd,  # 39
        beta_kww,  # 40
        beta_kww_sd,  # 41
        lts_kww_fit_goodness,  # 42-43
        # Method 7 (integral of Burr fit).
        lts_bur_characs,  # 44-48
        tau0_bur,  # 49
        tau0_bur_sd,  # 50
        beta_bur,  # 51
        beta_bur_sd,  # 52
        delta_bur,  # 53
        delta_bur_sd,  # 54
        lts_bur_fit_goodness,  # 55-56
        # Fit region
        fit_start * time_conv,  # 57
        (fit_stop - 1) * time_conv,  # 58
    ]
)
header = (
    "Bin residence times (hereafter denoted state lifetimes).\n"
    + "Average time that a given compound stays in a given bin calculated\n"
    + "either directly from the discrete trajectory (Method 1-3) or from the\n"
    + "corresponding remain probability function (Method 4-7).\n"
    + "\n"
    + "System:              {:s}\n".format(args.system)
    + "Settings:            {:s}\n".format(args.settings)
    + "Bin edges:           {:s}\n".format(infile_bins)
    + "Discrete trajectory: {:s}\n".format(infile_dtrj)
    + "Remain probability:  {:s}\n".format(infile_rp)
    + "\n"
    + "Compound:                      {:s}\n".format(args.cmp)
)
if surfq is not None:
    header += "Surface charge:                {:.2f} e/nm^2\n".format(surfq)
header += (
    "Lithium-to-ether-oxygen ratio: {:.4f}\n".format(Sim.Li_O_ratio)
    + "Ether oxygens per PEO chain:   {:d}\n".format(Sim.O_per_chain)
    + "\n"
    + "\n"
    + "Lifetimes are calculated using different methods:\n"
    + "\n"
    + "1) The average lifetime <t_cnt_cen> is calculated by counting how\n"
    + "   many frames a given compound stays in a given state including\n"
    + "   truncated states at the trajectory edges -> censored counting.\n"
    + "   Note that lifetimes calculated in this way are usually biased to\n"
    + "   lower values because of the limited length of the trajectory and\n"
    + "   because of truncation/censoring at the trajectory edges.\n"
    + "\n"
    + "2) The average lifetime <t_cnt_unc> is calculated by counting how\n"
    + "   many frames a given compound stays in a given state excluding\n"
    + "   truncated states at the trajectory edges -> uncensored counting.\n"
    + "   Note that lifetimes calculated in this way are usually biased to\n"
    + "   lower values because of the limited length of the trajectory.\n"
    + "   Uncensored counting might waste a significant amount of the\n"
    + "   trajectory.\n"
    + "\n"
    + "3) The average transition rate <k> is calculated as the number of\n"
    + "   transitions leading out of a given state divided by the number of\n"
    + "   frames that compounds have spent in this state.  The average\n"
    + "   lifetime <t_k> is calculated as the inverse transition rate:\n"
    + "     <t_k> = 1 / <k>\n"
    + "\n"
    + "4) The average lifetime <t_e> is set to the lag time at which the\n"
    + "   remain probability function p(t) crosses 1/e.  If this never\n"
    + "   happens, <t_e> is set to NaN.\n"
    + "\n"
    + "5) The remain probability function p(t) is interpreted as the\n"
    + "   survival function of the underlying lifetime distribution.  Thus,\n"
    + "   the lifetime can be calculated according to the alternative\n"
    + "   expectation formula [1]:\n"
    + "     <t_int^n> = n * int_0^inf t^(n-1) p(t) dt\n"
    + "   If p(t) does not decay below the given threshold of\n"
    + "   {:.4f}, <t_int^n> is set to NaN.\n".format(args.int_thresh)
    + "\n"
    + "6) The remain probability function p(t) is fitted by a Kohlrausch\n"
    + "   function (stretched exponential, survival function of the Weibull\n"
    + "   distribution):\n"
    + "     I_kww(t) = exp[-(t/tau0_kww)^beta_kww]\n"
    + "   Thereby, tau0_kww is confined to the interval\n"
    + "   [{:.4f}, {:.4f}] and beta_kww is confined to the interval\n".format(
        bounds_kww[0][0], bounds_kww[1][0]
    )
    + "   [{:.4f}, {:.4f}].\n".format(bounds_kww[0][1], bounds_kww[1][1])
    + "   The average lifetime <t_kww^n> is calculated according to the\n"
    + "   alternative expectation formula [1]:\n"
    + "     <t_kww^n> = n * int_0^inf t^(n-1) I_kww(t) dt\n"
    + "               = tau0_kww^n * Gamma(1 + n/beta_kww)\n"
    + "   where Gamma(z) is the gamma function.\n"
    + "\n"
    + "7) The remain probability function p(t) is fitted by the survival\n"
    + "   function of a Burr Type XII distribution:\n"
    + "     I_bur(t) = 1 / [1 + (t/tau0_bur)^beta_bur]^delta_bur\n"
    + "   Thereby, tau0_bur is confined to the interval\n"
    + "   [{:.4f}, {:.4f}], beta_bur is confined to the interval\n".format(
        bounds_bur[0][0], bounds_bur[1][0]
    )
    + "   [{:.4f}, {:.4f}] and beta_bur * delta_bur is confined to\n".format(
        bounds_bur[0][1], bounds_bur[1][1]
    )
    + "   the interval [{:.4f}, {:.4f}].\n".format(
        bounds_bur[0][2], bounds_bur[1][2]
    )
    + "   The average lifetime <t_bur^n> is calculated according to the\n"
    + "   alternative expectation formula [1]:\n"
    + "     <t_bur^n> = n * int_0^inf t^(n-1) I_bur(t) dt\n"
    + "               = tau0_bur^n * Gamma(delta_bur - n/beta_bur) *\n"
    + "                 Gamma(1 + n/beta_bur) / Gamma(delta_bur)\n"
    + "   where Gamma(z) is the gamma function.\n"
    + "\n"
    + "All fits are done using scipy.optimize.curve_fit with the 'Trust\n"
    + "Region Reflective' method.  The remain probability is always\n"
    + "fitted until it decays below the given threshold or until the\n"
    + "given lag time is reached (whatever happens earlier).\n"
    + "\n"
    + "int_thresh = {:.4f}\n".format(args.int_thresh)
    + "end_fit  = {}\n".format(args.end_fit)
    + "stop_fit = {:.4f}\n".format(args.stop_fit)
    + "Box edges:          {:>16.9e}, {:>16.9e} A\n".format(0, box_z)
    + "Electrode surfaces: {:>16.9e}, {:>16.9e} A\n".format(
        elctrd_thk, box_z - elctrd_thk
    )
    + "\n"
    + "Reference [1]:\n"
    + "  S. Chakraborti, F. Jardim, E. Epprecht,\n"
    + "  Higher-order moments using the survival function: The\n"
    + "  alternative expectation formula,\n"
    + "  The American Statistician, 2019, 73, 2, 191-194."
    + "\n"
    + "\n"
    + "The columns contain:\n"
    + "  1 State/Bin index (zero based)\n"
    + "  2 Lower bin edges / A\n"
    + "  3 Upper bin edges / A\n"
    + "  4 Distance of the lower bin edges to the left electrode surface / A\n"
    + "  5 Distance of the lower bin edges to the right electrode surface / A"
    + "\n"
    + "  6 Distance of the upper bin edges to the left electrode surface / A\n"
    + "  7 Distance of the upper bin edges to the right electrode surface / A"
    + "\n"
    + "\n"
    + "  Lifetime from Method 1 (censored counting)\n"
    + "  8 Sample mean <t_cnt_cen> / ns\n"
    + "  9 Uncertainty of the sample mean (standard error) / ns\n"
    + " 10 Corrected sample standard deviation / ns\n"
    + " 11 Unbiased sample skewness\n"
    + " 12 Unbiased sample excess kurtosis (Fisher)\n"
    + " 13 Sample median / ns\n"
    + " 14 Sample minimum / ns\n"
    + " 15 Sample maximum / ns\n"
    + " 16 Number of observations/samples\n"
    + "\n"
    + "  Lifetime from Method 2 (uncensored counting)\n"
    + " 17-25 As Method 1\n"
    + "\n"
    + "  Lifetime from Method 3 (inverse transition rate)\n"
    + " 26 <t_k> / ns\n"
    + "\n"
    + "  Lifetime from Method 4 (1/e criterion)\n"
    + " 27 <t_e> / ns\n"
    + "\n"
    + "  Lifetime from Method 5 (direct integral)\n"
    + " 28 Mean <t_int> / ns\n"
    + " 29 Standard deviation / ns\n"
    + " 30 Skewness\n"
    + " 31 Excess kurtosis (Fisher)\n"
    + " 32 Median / ns\n"
    + "\n"
    + "  Lifetime from Method 6 (integral of Kohlrausch fit)\n"
    + " 33-37 As Method 5\n"
    + " 38 Fit parameter tau0_kww / ns\n"
    + " 39 Standard deviation of tau0_kww / ns\n"
    + " 40 Fit parameter beta_kww\n"
    + " 41 Standard deviation of beta_kww\n"
    + " 42 Coefficient of determination of the fit (R^2 value)\n"
    + " 43 Root-mean-square error (RMSE) of the fit\n"
    + "\n"
    + "  Lifetime from Method 7 (integral of Burr fit)\n"
    + " 44-52 As Method 6\n"
    + " 53 Fit parameter delta_burr\n"
    + " 54 Standard deviation of delta_burr\n"
    + " 55 Coefficient of determination of the fit (R^2 value)\n"
    + " 56 Root-mean-square error (RMSE) of the fit\n"
    + "\n"
    + "  Fit region for all fitting methods\n"
    + " 57 Start of fit region (inclusive) / ns\n"
    + " 58 End of fit region (inclusive) / ns\n"
    + "\n"
    + "Column number:\n"
)
header += "{:>14d}".format(1)
for i in range(2, data.shape[-1] + 1):
    header += " {:>16d}".format(i)
leap.io_handler.savetxt(outfile_txt, data, header=header)
print("Created {}".format(outfile_txt))


print("Creating plot(s)...")
elctrd_thk /= 10  # A -> nm.
box_z /= 10  # A -> nm.
bins /= 10  # A -> nm.
bin_mids = bins_up - (bins_up - bins_low) / 2
bin_mids /= 10  # A -> nm.

label_cnt_cen = "Cens."
label_cnt_unc = "Uncens."
label_k = "Rate"
# label_e = r"$1/e$"
label_int = "Area"
label_kww = "Kohl."
label_bur = "Burr"

color_cnt_cen = "tab:orange"
color_cnt_unc = "tab:red"
color_k = "tab:brown"
# color_e = "tab:pink"
color_int = "tab:purple"
color_kww = "tab:blue"
color_bur = "tab:cyan"

marker_cnt_cen = "H"
marker_cnt_unc = "h"
marker_k = "p"
# marker_e = "<"
marker_int = ">"
marker_kww = "^"
marker_bur = "v"

xlabel = r"$z$ / nm"
xlim = (0, box_z)
if surfq is None:
    legend_title = ""
else:
    legend_title = r"$\sigma_s = \pm %.2f$ $e$/nm$^2$" % surfq + "\n"
legend_title = (
    legend_title
    + r"$n_{EO} = %d$, " % Sim.O_per_chain
    + r"$r = %.4f$" % Sim.Li_O_ratio
)
height_ratios = (0.2, 1)
cmap = plt.get_cmap()
c_vals = np.arange(n_states)
c_norm = max(1, n_states - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile_pdf)
with PdfPages(outfile_pdf) as pdf:
    # Plot distribution characteristics vs. bins.
    ylabels = (
        "Residence Time / ns",
        "Std. Dev. / ns",
        "Skewness",
        "Excess Kurtosis",
        "Median / ns",
    )
    for i, ylabel in enumerate(ylabels):
        if i == 0:
            offset_i_cnt = 0
        else:
            offset_i_cnt = 1
        fig, axs = plt.subplots(
            clear=True,
            nrows=2,
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        fig.set_figheight(fig.get_figheight() * sum(height_ratios))
        ax_profile, ax = axs
        leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
        if surfq is not None:
            leap.plot.elctrds(
                ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
            )
        if i == 2:
            # Skewness of exponential distribution is 2.
            ax.axhline(
                y=2, color="tab:green", linestyle="dashed", label="Exp. Dist."
            )
        elif i == 3:
            # Excess kurtosis of exponential distribution is 6
            ax.axhline(
                y=6, color="tab:green", linestyle="dashed", label="Exp. Dist."
            )
        # Method 1 (censored counting).
        ax.errorbar(
            bin_mids,
            lts_cnt_cen_characs[:, i + offset_i_cnt],
            yerr=lts_cnt_cen_characs[:, i + 1] if i == 0 else None,
            label=label_cnt_cen,
            color=color_cnt_cen,
            marker=marker_cnt_cen,
            alpha=leap.plot.ALPHA,
        )
        # Method 2 (uncensored counting).
        ax.errorbar(
            bin_mids,
            lts_cnt_unc_characs[:, i + offset_i_cnt],
            yerr=lts_cnt_unc_characs[:, i + 1] if i == 0 else None,
            label=label_cnt_unc,
            color=color_cnt_unc,
            marker=marker_cnt_unc,
            alpha=leap.plot.ALPHA,
        )
        if i == 0:
            # Method 3 (inverse transition rate).
            ax.errorbar(
                bin_mids,
                lts_k,
                yerr=None,
                label=label_k,
                color=color_k,
                marker=marker_k,
                alpha=leap.plot.ALPHA,
            )
            # # Method 4 (1/e criterion).
            # ax.errorbar(
            #     bin_mids,
            #     lts_e,
            #     yerr=None,
            #     label=label_e,
            #     color=color_e,
            #     marker=marker_e,
            #     alpha=leap.plot.ALPHA,
            # )
        # Method 5 (direct integral)
        ax.errorbar(
            bin_mids,
            lts_int_characs[:, i],
            yerr=None,
            label=label_int,
            color=color_int,
            marker=marker_int,
            alpha=leap.plot.ALPHA,
        )
        # Method 6 (integral of Kohlrausch fit).
        ax.errorbar(
            bin_mids,
            lts_kww_characs[:, i],
            yerr=None,
            label=label_kww,
            color=color_kww,
            marker=marker_kww,
            alpha=leap.plot.ALPHA,
        )
        # Method 7 (integral of Burr fit).
        ax.errorbar(
            bin_mids,
            lts_bur_characs[:, i],
            yerr=None,
            label=label_bur,
            color=color_bur,
            marker=marker_bur,
            alpha=leap.plot.ALPHA,
        )
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax.get_ylim()
        if i not in (2, 3) and ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        leap.plot.bins(ax, bins=bins)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend = ax.legend(
            title=legend_title,
            loc="upper center",
            ncol=3,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        yd_min, yd_max = get_ydata_min_max(ax)
        if len(yd_min) > 0:
            # Set y axis to log scale.
            # Round y limits to next lower and higher power of ten.
            ylim = ax.get_ylim()
            ymin = 10 ** np.floor(np.log10(np.min(yd_min)))
            ymax = 10 ** np.ceil(np.log10(np.max(yd_max)))
            ax.set_ylim(
                ymin if np.isfinite(ymin) else None,
                ymax if np.isfinite(ymax) else None,
            )
            ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            pdf.savefig()
        plt.close()

    # Plot number of min, max and number of samples for count methods.
    ylabels = (
        "Min. Lifetime / ns",
        "Max. Lifetime / ns",
        "No. of Samples",
    )
    for i, ylabel in enumerate(ylabels):
        fig, axs = plt.subplots(
            clear=True,
            nrows=2,
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        fig.set_figheight(fig.get_figheight() * sum(height_ratios))
        ax_profile, ax = axs
        leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
        if surfq is not None:
            leap.plot.elctrds(
                ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
            )
        # Method 1 (censored counting).
        ax.plot(
            bin_mids,
            lts_cnt_cen_characs[:, 6 + i],
            label=label_cnt_cen,
            color=color_cnt_cen,
            marker=marker_cnt_cen,
            alpha=leap.plot.ALPHA,
        )
        # Method 2 (uncensored counting).
        ax.plot(
            bin_mids,
            lts_cnt_unc_characs[:, 6 + i],
            label=label_cnt_unc,
            color=color_cnt_unc,
            marker=marker_cnt_unc,
            alpha=leap.plot.ALPHA,
        )
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax.get_ylim()
        if ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        leap.plot.bins(ax, bins=bins)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend = ax.legend(
            title=legend_title,
            loc="lower center" if i == 2 else "upper center",
            ncol=3,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        yd_min, yd_max = get_ydata_min_max(ax)
        if len(yd_min) > 0:
            # Set y axis to log scale.
            # Round y limits to next lower and higher power of ten.
            ylim = ax.get_ylim()
            ymin = 10 ** np.floor(np.log10(np.min(yd_min)))
            ymax = 10 ** np.ceil(np.log10(np.max(yd_max)))
            ax.set_ylim(
                ymin if np.isfinite(ymin) else None,
                ymax if np.isfinite(ymax) else None,
            )
            ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            pdf.savefig()
        plt.close()

    # Plot fit parameters tau0 and beta.
    ylabels = (
        r"Fit Parameter $\tau_0$ / ns",
        r"Fit Parameter $\beta$",
    )
    for i, ylabel in enumerate(ylabels):
        fig, axs = plt.subplots(
            clear=True,
            nrows=2,
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        fig.set_figheight(fig.get_figheight() * sum(height_ratios))
        ax_profile, ax = axs
        leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
        if surfq is not None:
            leap.plot.elctrds(
                ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
            )
        # Method 6 (Kohlrausch fit).
        ax.errorbar(
            bin_mids,
            popt_kww[:, i],
            yerr=perr_kww[:, i],
            label=label_kww,
            color=color_kww,
            marker=marker_kww,
            alpha=leap.plot.ALPHA,
        )
        # Method 7 (Burr fit).
        ax.errorbar(
            bin_mids,
            popt_bur[:, i],
            yerr=perr_bur[:, i],
            label=label_bur,
            color=color_bur,
            marker=marker_bur,
            alpha=leap.plot.ALPHA,
        )
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax.get_ylim()
        if i not in (2, 3) and ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        leap.plot.bins(ax, bins=bins)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend = ax.legend(
            title=legend_title,
            loc="upper center",
            ncol=2,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        yd_min, yd_max = get_ydata_min_max(ax)
        if len(yd_min) > 0:
            # Set y axis to log scale.
            # Round y limits to next lower and higher power of ten.
            ylim = ax.get_ylim()
            ymin = 10 ** np.floor(np.log10(np.min(yd_min)))
            ymax = 10 ** np.ceil(np.log10(np.max(yd_max)))
            ax.set_ylim(
                ymin if np.isfinite(ymin) else None,
                ymax if np.isfinite(ymax) else None,
            )
            ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            pdf.savefig()
        plt.close()

    # Plot fit parameter delta.
    fig, axs = plt.subplots(
        clear=True,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.set_figheight(fig.get_figheight() * sum(height_ratios))
    ax_profile, ax = axs
    leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
    if surfq is not None:
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
    # Method 7 (Burr fit).
    ax.errorbar(
        bin_mids,
        delta_bur,
        yerr=delta_bur_sd,
        label=label_bur,
        color=color_bur,
        marker=marker_bur,
        alpha=leap.plot.ALPHA,
    )
    ax.set(xlabel=xlabel, ylabel=r"Fit Parameter $\delta$", xlim=xlim)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.set_ylim(0, ylim[1])
    leap.plot.bins(ax, bins=bins)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend = ax.legend(title=legend_title, **mdtplt.LEGEND_KWARGS_XSMALL)
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    yd_min, yd_max = get_ydata_min_max(ax)
    if len(yd_min) > 0:
        # Set y axis to log scale.
        # Round y limits to next lower and higher power of ten.
        ylim = ax.get_ylim()
        ymin = 10 ** np.floor(np.log10(np.min(yd_min)))
        ymax = 10 ** np.ceil(np.log10(np.max(yd_max)))
        ax.set_ylim(
            ymin if np.isfinite(ymin) else None,
            ymax if np.isfinite(ymax) else None,
        )
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        pdf.savefig()
    plt.close()

    # Plot goodness of fit quantities.
    ylabels = (r"Coeff. of Determ. $R^2$", "RMSE")
    for i, ylabel in enumerate(ylabels):
        fig, axs = plt.subplots(
            clear=True,
            nrows=2,
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        fig.set_figheight(fig.get_figheight() * sum(height_ratios))
        ax_profile, ax = axs
        leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
        if surfq is not None:
            leap.plot.elctrds(
                ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
            )
        # Method 6 (Kohlrausch fit).
        ax.plot(
            bin_mids,
            lts_kww_fit_goodness[:, i],
            label=label_kww,
            color=color_kww,
            marker=marker_kww,
            alpha=leap.plot.ALPHA,
        )
        # Method 7 (Burr fit).
        ax.plot(
            bin_mids,
            lts_bur_fit_goodness[:, i],
            label=label_bur,
            color=color_bur,
            marker=marker_bur,
            alpha=leap.plot.ALPHA,
        )
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
        ylim = ax.get_ylim()
        if ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        leap.plot.bins(ax, bins=bins)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend = ax.legend(
            title=legend_title, ncol=2, **mdtplt.LEGEND_KWARGS_XSMALL
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        yd_min, yd_max = get_ydata_min_max(ax)
        if len(yd_min) > 0:
            # Set y axis to log scale.
            # Round y limits to next lower and higher power of ten.
            ylim = ax.get_ylim()
            ymin = 10 ** np.floor(np.log10(np.min(yd_min)))
            if i == 0:
                ymax = 2
            else:
                ymax = 10 ** np.ceil(np.log10(np.max(yd_max)))
            ax.set_ylim(
                ymin if np.isfinite(ymin) else None,
                ymax if np.isfinite(ymax) else None,
            )
            ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            pdf.savefig()
        plt.close()

    # Plot end of fit region.
    fig, axs = plt.subplots(
        clear=True,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.set_figheight(fig.get_figheight() * sum(height_ratios))
    ax_profile, ax = axs
    leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
    if surfq is not None:
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
    ax.plot(bin_mids, (fit_stop - 1) * time_conv, marker="v")
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel="End of Fit Region / ns", xlim=xlim)
    leap.plot.bins(ax, bins=bins)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    pdf.savefig()
    plt.close()

    # Plot remain probabilities and Kohlrausch fits for each bin.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, rp in enumerate(remain_props.T):
        times_fit = times[fit_start[i] : fit_stop[i]]
        fit = mdt.func.kww(times_fit, *popt_kww[i])
        lines = ax.plot(
            times,
            rp,
            label=r"$%d$" % (states[i] + 1),
            linewidth=1,
            alpha=leap.plot.ALPHA,
        )
        ax.plot(
            times_fit,
            fit,
            label=label_kww if i == len(remain_props.T) - 1 else None,
            linestyle="dashed",
            color=lines[0].get_color(),
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Time / ns",
        ylabel=r"Autocorrelation $C(t)$",
        xlim=(times[1], times[-1]),
        ylim=(0, 1),
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    legend = ax.legend(
        title=legend_title + "\nBin Number",
        loc="upper right",
        ncol=3,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot Kohlrausch fit residuals for each bin.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, rp in enumerate(remain_props.T):
        times_fit = times[fit_start[i] : fit_stop[i]]
        fit = mdt.func.kww(times_fit, *popt_kww[i])
        res = rp[fit_start[i] : fit_stop[i]] - fit
        ax.plot(
            times_fit,
            res,
            label=r"$%d$" % (states[i] + 1),
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Time / ns",
        ylabel="Kohlrausch Fit Residuals",
        xlim=(times[1], times[-1]),
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    legend = ax.legend(
        title=legend_title + "\nBin Number",
        loc="lower right",
        ncol=3,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot remain probabilities and Burr fits for each bin.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, rp in enumerate(remain_props.T):
        times_fit = times[fit_start[i] : fit_stop[i]]
        fit = mdt.func.burr12_sf_alt(times_fit, *popt_bur[i])
        lines = ax.plot(
            times,
            rp,
            label=r"$%d$" % (states[i] + 1),
            linewidth=1,
            alpha=leap.plot.ALPHA,
        )
        ax.plot(
            times_fit,
            fit,
            label=label_bur if i == len(remain_props.T) - 1 else None,
            linestyle="dashed",
            color=lines[0].get_color(),
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Time / ns",
        ylabel=r"Autocorrelation $C(t)$",
        xlim=(times[1], times[-1]),
        ylim=(0, 1),
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    legend = ax.legend(
        title=legend_title + "\nBin Number",
        loc="upper right",
        ncol=3,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot Burr fit residuals for each bin.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for i, rp in enumerate(remain_props.T):
        times_fit = times[fit_start[i] : fit_stop[i]]
        fit = mdt.func.burr12_sf_alt(times_fit, *popt_bur[i])
        res = rp[fit_start[i] : fit_stop[i]] - fit
        ax.plot(
            times_fit,
            res,
            label=r"$%d$" % (states[i] + 1),
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Time / ns",
        ylabel="Burr Fit Residuals",
        xlim=(times[1], times[-1]),
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    legend = ax.legend(
        title=legend_title + "\nBin Number",
        loc="lower right",
        ncol=3,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile_pdf))
print("Done")
