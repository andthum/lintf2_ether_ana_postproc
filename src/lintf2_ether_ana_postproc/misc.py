"""Miscellaneous functions."""


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
