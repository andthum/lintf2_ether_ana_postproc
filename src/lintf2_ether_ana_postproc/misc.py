"""Miscellaneous functions."""


# Third-party libraries
import mdtools as mdt
import numpy as np


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


def dens2free_energy(x, dens, bulk_region=None):
    r"""
    Calculate free energy profiles from density profiles.

    Parameters
    ----------
    x : array_like
        x values / sample points.
    dens : array_like
        Corresponding density profile values.
    bulk_region : None or 2-tuple of floats, optional
        Start and end of the bulk region in units of `x`.  If provided,
        the free energy profile will be shifted such that the free
        energy in the bulk region is zero.

    Returns
    -------
    free_en : numpy.ndarray
        Free energy profile :math:`F(z)` in units of :math:`k_B T`, i.e.
        :math:`\frac{F(z)}{k_B T}`.

    Notes
    -----
    The free energy profile :math:`F(z)` is calculated from the density
    profile :math:`\rho(z)` according to

    .. math::

        \frac{F(z)}{k_B T} =
        -\ln\left[ \frac{\rho(z)}{\rho^\circ} \right]

    Here, :math:`k_B` is the Boltzmann constant and :math:`T` is the
    temperature.  If `bulk_start` is given, :math:`\rho^\circ` is chosen
    such that the free energy in the bulk region is zero, i.e.
    :math:`\rho^\circ` is set to the average density in the bulk region.
    If `bulk_start` is None, :math:`\rho^\circ` is set to one.
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
        bulk_start, bulk_start_ix = mdt.nph.find_nearest(
            x, bulk_region[0], return_index=True
        )
        if not np.isclose(bulk_start, bulk_region[0], rtol=0, atol=0.1):
            raise ValueError(
                "`bulk_start` ({}) != `bulk_region[0]`"
                " ({})".format(bulk_start, bulk_region[0])
            )
        bulk_stop, bulk_stop_ix = mdt.nph.find_nearest(
            x, bulk_region[1], return_index=True
        )
        if not np.isclose(bulk_stop, bulk_region[1], rtol=0, atol=0.1):
            raise ValueError(
                "`bulk_stop` ({}) != `bulk_region[1]`"
                " ({})".format(bulk_stop, bulk_region[1])
            )
        free_en_bulk = np.mean(free_en[bulk_start_ix : bulk_stop_ix + 1])
        free_en -= free_en_bulk
    return free_en
