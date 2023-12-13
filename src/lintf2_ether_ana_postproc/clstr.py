"""Functions operating on data clusters."""


# Third-party libraries
import mdtools as mdt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist

# First-party libraries
import lintf2_ether_ana_postproc as leap


def peak_pos(
    data, pkp_col_ix, dist_thresh=None, return_dist_thresh=False, **kwargs
):
    """
    Cluster the peak positions of a density/free-energy profile.

    Parameters
    ----------
    data : list
        3-dimensional list of peak data as returned by
        :func:`lintf2_ether_ana_postproc.simulation.read_free_energy_extrema`.
        The first index must address the columns read from the output
        file of
        :file:`scripts/gmx/density-z/get_free-energy_extrema.py`.  The
        second index must address the peak-position type, i.e. whether
        the peak is in the left or right half of the simulation box.
        The third index must address the simulation.  The data will be
        clustered by the peak positions.
    pkp_col_ix : int
        The index for the first dimension of `data` that returns the
        peak positions.
    dist_thresh : float or array_like or None
        Distance threshold for clustering.  Peak positions are assigned
        to clusters such that the distance between peaks in the same
        cluster is not greater than the given threshold.  You can
        provide either one value for each peak-position type or one
        value for all peak-position types.  If ``None``, for each
        peak-position type, `dist_thresh` is set to the half of the
        minimum distance that peaks can have in a single simulation.  If
        all simulations have only one or no peak, `dist_thresh` is set
        to ``np.inf``.
    return_dist_thresh : bool, optional
        If ``True``, return the distance threshold used for clustering
        for each peak-position type.
    kwargs : dict, optional
        Keyword arguments to parse to
        :func:`scipy.cluster.hierarchy.linkage`.  See there for possible
        options.  By default, `method` is set to ``"single"``, `metric`
        is set to ``"euclidean"`` and `optimal_ordering` is set to
        ``True``.

    Returns
    -------
    data_clstr : list
        The input data with the values along the last axis concatenated.
        This means `data_clstr` is a 2-dimensional list.  The first
        index addresses the columns read from the output file of
        :file:`scripts/gmx/density-z/get_free-energy_extrema.py`.  The
        second index addresses the peak-position type.
    clstr_ix : list
        List of NumPy arrays containing the 0-based cluster indices for
        each peak in `data_clstr` for each peak-position type.
    linkage_matrices : list
        List containing a linkage matrix for each peak-position type.
        See :func:`scipy.cluster.hierarchy.linkage` for details about
        the structure of linkage matrices.
    n_clstrs : numpy.ndarray
        Number of clusters for each peak-position type.
    n_pks_per_sim : list
        List of NumPy arrays containing the total number of peaks in
        each simulation for each peak-position type.
    dist_thresh : numpy.ndarray
        The distance threshold used for clustering for each
        peak-position type.  Only returned if `return_dist_thresh` is
        ``True``.
    """
    kwargs.setdefault("method", "single")
    kwargs.setdefault("metric", "euclidean")
    kwargs.setdefault("optimal_ordering", True)

    try:
        n_pkp_types = len(data[0])
    except (TypeError, IndexError):
        raise TypeError("`data` must be a 3-dimensional list")
    try:
        n_sims = len(data[0][0])
    except (TypeError, IndexError):
        raise TypeError("`data` must be a 3-dimensional list")

    try:
        data[pkp_col_ix]
    except IndexError:
        raise IndexError(
            "`pkp_col_ix` ({}) is out of bounds for axis 0 of `data` with size"
            " {}".format(pkp_col_ix, len(data))
        )

    pk_pos_types = ("left", "right")  # Peak at left or right electrode.
    if n_pkp_types != len(pk_pos_types):
        raise TypeError(
            "`data` must contain values for {} peak-position types but"
            " contains values for {} peak-position"
            " type(s)".format(len(pk_pos_types), n_pkp_types)
        )

    if dist_thresh is None:
        dist_thresh = np.full(n_pkp_types, np.nan, dtype=np.float64)
        for pkt_ix, dat_pkt in enumerate(data[pkp_col_ix]):
            pk_dists_min = np.full(n_sims, np.nan, dtype=np.float64)
            for sim_ix, dat_sim in enumerate(dat_pkt):
                if len(dat_sim) > 1:
                    pk_dists = pdist(np.expand_dims(dat_sim, axis=-1))
                    pk_dists_min[sim_ix] = np.nanmin(pk_dists)
                else:
                    pk_dists_min[sim_ix] = np.inf
            if np.all(np.isnan(pk_dists_min)):
                raise ValueError(
                    "No valid peaks found for peak-position type"
                    " '{}'".format(pk_pos_types[pkt_ix])
                )
            dist_thresh[pkt_ix] = 0.5 * np.nanmin(pk_dists_min)
    else:
        try:
            if len(dist_thresh) != len(pk_pos_types):
                raise TypeError(
                    "`dist_thresh` must either be an array-like of shape ({},)"
                    " or a scalar".format(len(pk_pos_types))
                )
        except TypeError:
            dist_thresh = tuple(dist_thresh for pkp_type in pk_pos_types)

    # Total number of peaks in each simulation.
    n_pks_per_sim = [
        np.array([len(dat_sim) for dat_sim in dat_pkt])
        for dat_pkt in data[pkp_col_ix]
    ]

    # Concatenate values along the last axis for clustering.
    data_clstr = [
        [np.concatenate(dat_pkt) for dat_pkt in dat_col] for dat_col in data
    ]

    # Cluster peak positions.
    linkage_matrices = [None for pkp_type in pk_pos_types]
    clstr_ix = [None for pkp_type in pk_pos_types]
    n_clstrs = np.zeros_like(pk_pos_types, dtype=int)
    for pkt_ix, dat_clstr_pkt in enumerate(data_clstr[pkp_col_ix]):
        linkage_matrices[pkt_ix] = linkage(
            np.expand_dims(dat_clstr_pkt, axis=1), **kwargs
        )
        clstr_ix[pkt_ix] = fcluster(
            linkage_matrices[pkt_ix], dist_thresh[pkt_ix], criterion="distance"
        )
        n_clstrs[pkt_ix] = np.max(clstr_ix[pkt_ix])
        clstr_ix[pkt_ix] -= 1  # Convert cluster IDs to 0-based indexing.

    ret = (data_clstr, clstr_ix, linkage_matrices, n_clstrs, n_pks_per_sim)
    if return_dist_thresh:
        ret += (dist_thresh,)
    return ret


def means(data, clstr_ix):
    """
    Calculate the cluster means.

    Calculate the cluster means as the arithmetic average over all data
    that belong to the same cluster.

    Parameters
    ----------
    data : array_like
        Array of data.  If the array is not 1-dimensional, it is
        flattened before calculating the cluster means.
    clstr_ix : array_like
        Array of the same shape as `data` containing the corresponding
        zero-based cluster indices that indicate to which cluster a
        value in `data` belongs.

    Returns
    -------
    clstr_av : numpy.ndarray
        1-dimensional array containing the mean of each cluster.
    """
    data = np.asarray(data)
    clstr_ix = np.asarray(clstr_ix)
    if clstr_ix.shape != data.shape:
        raise ValueError(
            "`clstr_ix` ({}) must have the same shape as `data`"
            " ({})".format(clstr_ix.shape, data.shape)
        )

    clstr_ix_unique = np.unique(clstr_ix)
    clstr_av = np.full_like(clstr_ix_unique, np.nan, dtype=np.float64)
    valid = np.zeros_like(clstr_ix, dtype=bool)
    for cix in clstr_ix_unique:
        valid = np.equal(clstr_ix, cix, out=valid)
        clstr_av[cix] = np.nanmean(data[valid])
    return clstr_av


def sort(data, clstr_ix, return_means=False):
    """
    Sort the clusters by their means in ascending order.

    First, calculate the cluster means as the arithmetic average over
    all data that belong to the same cluster.  Second, sort the cluster
    indices according to the cluster means.

    Parameters
    ----------
    data : array_like
        Array of data.  If the array is not 1-dimensional, it is
        flattened before calculating the cluster means.
    clstr_ix : array_like
        Array of the same shape as `data` containing the corresponding
        zero-based cluster indices that indicate to which cluster a
        value in `data` belongs.
    return_means : bool, optional
        If ``True``, return the sorted cluster means.

    Returns
    -------
    clstr_ix_sorted : numpy.ndarray
        1-dimensional array containing the sorted cluster indices.
    clstr_av_sorted : numpy.ndarray
        1-dimensional array containing the sorted cluster means.  Only
        returned if `return_means` is ``True``.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.clstr.means` :
        Calculate the cluster mean.
    """
    data = np.asarray(data)
    clstr_ix = np.asarray(clstr_ix)
    if clstr_ix.shape != data.shape:
        raise ValueError(
            "`clstr_ix` ({}) must have the same shape as `data`"
            " ({})".format(clstr_ix.shape, data.shape)
        )

    clstr_av = leap.clstr.means(data, clstr_ix)
    sort_ix = np.argsort(clstr_av)
    clstr_ix_sorted = np.copy(clstr_ix)
    valid = np.zeros_like(clstr_ix, dtype=bool)
    for cix, six in enumerate(sort_ix):
        valid = np.equal(clstr_ix, six, out=valid)
        clstr_ix_sorted[valid] = cix

    if return_means:
        return clstr_ix_sorted, clstr_av[sort_ix]
    else:
        return clstr_ix_sorted


def dists_succ(
    data,
    clstr_ix,
    method="single",
    return_ix=False,
    return_means=False,
    return_bounds=False,
):
    r"""
    Calculate the distances between successive clusters.

    First, sort the clusters by their mean in ascending order.  Second,
    calculate the distances between successive clusters.

    Parameters
    ----------
    data : array_like
        Array of data.  If the array is not 1-dimensional, it is
        flattened before calculating the cluster distances.
    clstr_ix : array_like
        Array of the same shape as `data` containing the corresponding
        zero-based cluster indices that indicate to which cluster a
        value in `data` belongs.
    method : {"single", "complete"}
        The method to use to calculate the distances.  If "single", the
        distance between two clusters is calculated as the minimum
        distance between the data points in the two clusters:

        .. math::

            d(u, v) = \min_{|u[i] - v[j]|}

        If "complete", the distance between two clusters is calculated
        as the maximum distance between the data points in the two
        clusters:

        .. math::

            d(u, v) = \max_{|u[i] - v[j]|}

        These definitions are adopted from
        :func:`scipy.cluster.hierarchy.linkage`.
    return_ix : bool, optional
        If ``True``, return the sorted cluster indices.
    return_means : bool, optional
        If ``True``, return the sorted cluster means.
    return_bounds : bool, optional
        If ``True``, return the cluster boundaries, i.e. the mid points
        between two clusters.

    Returns
    -------
    clstr_dists : numpy.ndarray
        1-dimensional array containing the distances between successive
        clusters.
    clstr_ix_sorted : numpy.ndarray
        1-dimensional array containing the sorted cluster indices.  Only
        returned if `return_ix` is ``True``.
    clstr_av_sorted : numpy.ndarray
        1-dimensional array containing the sorted cluster means.  Only
        returned if `return_means` is ``True``.
    clstr_bounds : numpy.ndarray
        1-dimensional array containing the boundaries between successive
        clusters.  Only returned if `return_bounds` is ``True``.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.misc.dists` :
        Calculate the distances between all clusters
    :func:`lintf2_ether_ana_postproc.misc.sort` :
        Sort the clusters by their means in ascending order
    """
    data = np.asarray(data)
    clstr_ix = np.asarray(clstr_ix)
    if clstr_ix.shape != data.shape:
        raise ValueError(
            "`clstr_ix` ({}) must have the same shape as `data`"
            " ({})".format(clstr_ix.shape, data.shape)
        )
    method = method.lower()
    if method not in ("single", "complete"):
        raise ValueError("Unknown `method`: {}".format(method))

    clstr_ix, clstr_av = leap.clstr.sort(data, clstr_ix, return_means=True)
    clstr_ix_unique = np.unique(clstr_ix)
    clstr_dists = np.full(len(clstr_ix_unique) - 1, np.nan, dtype=np.float64)
    clstr_bounds = np.full_like(clstr_dists, np.nan)
    valid_curr = np.zeros_like(clstr_ix, dtype=bool)
    valid_prev = clstr_ix == clstr_ix_unique[0]
    data_prev = np.expand_dims(data[valid_prev], axis=-1)
    for cix in clstr_ix_unique[1:]:
        valid_curr = np.equal(clstr_ix, cix, out=valid_curr)
        data_curr = np.expand_dims(data[valid_curr], axis=-1)
        dists = distance_matrix(data_curr, data_prev)
        if method == "single":
            clstr_dists[cix - 1] = np.min(dists)
            clstr_bounds[cix - 1] = np.max(data_prev)
        elif method == "complete":
            clstr_dists[cix - 1] = np.max(dists)
            clstr_bounds[cix - 1] = np.min(data_prev)
        else:
            raise ValueError("Unknown `method`: {}".format(method))
        clstr_bounds[cix - 1] += clstr_dists[cix - 1] / 2
        np.copyto(valid_prev, valid_curr)
        data_prev = data_curr

    ret = (clstr_dists,)
    if return_ix:
        ret += (clstr_ix,)
    if return_means:
        ret += (clstr_av,)
    if return_bounds:
        ret += (clstr_bounds,)
    if len(ret) == 1:
        ret = ret[0]
    return ret


def dists(data, clstr_ix, method="single"):
    r"""
    Calculate the distances between all clusters.

    Parameters
    ----------
    data : array_like
        Array of data.  If the array is not 1-dimensional, it is
        flattened before calculating the cluster distances.
    clstr_ix : array_like
        Array of the same shape as `data` containing the corresponding
        zero-based cluster indices that indicate to which cluster a
        value in `data` belongs.
    method : {"single", "complete"}
        The method to use to calculate the distances.  If "single", the
        distance between two clusters is calculated as the minimum
        distance between the data points in the two clusters:

        .. math::

            d(u, v) = \min_{|u[i] - v[j]|}

        If "complete", the distance between two clusters is calculated
        as the maximum distance between the data points in the two
        clusters:

        .. math::

            d(u, v) = \max_{|u[i] - v[j]|}

        These definitions are adopted from
        :func:`scipy.cluster.hierarchy.linkage`.

    Returns
    -------
    clstr_dists : numpy.ndarray
        Distance matrix containing the distances between all clusters.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.misc.dists_succ` :
        Calculate the distances between successive clusters
    """
    data = np.asarray(data)
    clstr_ix = np.asarray(clstr_ix)
    if clstr_ix.shape != data.shape:
        raise ValueError(
            "`clstr_ix` ({}) must have the same shape as `data`"
            " ({})".format(clstr_ix.shape, data.shape)
        )
    method = method.lower()
    if method not in ("single", "complete"):
        raise ValueError("Unknown `method`: {}".format(method))

    clstr_ix_unique = np.unique(clstr_ix)
    n_clstrs = len(clstr_ix_unique)
    clstr_dists = np.full((n_clstrs, n_clstrs), np.nan, dtype=np.float64)
    valid_i = np.zeros_like(clstr_ix, dtype=bool)
    valid_j = np.zeros_like(clstr_ix, dtype=bool)
    for cix_i in clstr_ix_unique:
        valid_i = np.equal(clstr_ix, cix_i, out=valid_i)
        data_i = np.expand_dims(data[valid_i], axis=-1)
        for cix_j in clstr_ix_unique:
            valid_j = np.equal(clstr_ix, cix_j, out=valid_j)
            data_j = np.expand_dims(data[valid_j], axis=-1)
            dists = distance_matrix(data_i, data_j)
            if method == "single":
                clstr_dists[cix_i][cix_j] = np.min(dists)
            elif method == "complete":
                clstr_dists[cix_i][cix_j] = np.max(dists)
            else:
                raise ValueError("Unknown `method`: {}".format(method))

    if np.any(np.diagonal(clstr_dists) != 0):
        raise ValueError(
            "At least one diagonal element of `clstr_dists` is not zero.  This"
            " should not have happened.  Diagonal:"
            " {}".format(np.diagonal(clstr_dists))
        )
    return clstr_dists


def bin_indices(
    Sim, use_bin_ix=None, rel_bins=True, bins=None, infile=None, **kwargs
):
    """
    Assign bin indices to layers/free-energy minima and cluster them
    based on the minima positions.

    Parameters
    ----------
    Sim : lintf2_ether_ana_postproc.simulation.Simulation
        The :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        instance for which to assign and cluster the bin indices.
    use_bin_ix : array_like or None
        Only cluster the given bin indices.
    rel_bins : bool, optional
        If ``True``, convert the absolut bin positions to positions
        relative to the electrodes.
    bins : array_like or None, optional
        The bin edges to cluster in Angstrom.  If ``None``, the bin
        edges are read from the given `infile`.
    infile : str or bytes or os.PathLike or None, optional
        Name of the file containing the bin edges in Angstrom.  If
        ``None``, the standard bin file in the analysis directory
        :file:`gmx/density-z/` of the given simulation is read.
    kwargs : dict, optional
        Keyword arguments to parse to :func:`numpy.loadtxt` for reading
        the bin file.  By default, `usecols` is set to ``0``.

    Returns
    -------
    TODO
    """
    Elctrd = leap.simulation.Electrode()
    elctrd_thk = Elctrd.ELCTRD_THK  # [Angstrom].
    bulk_start = Elctrd.BULK_START  # [A].
    box_z = Sim.box[2]  # [A].
    bulk_region = Sim.bulk_region  # [A].

    if bins is not None:
        _bins = mdt.check.bin_edges(bins, amin=0, amax=box_z, verbose=False)
        # Remove the tolerance added to the last bin edge by
        # `mdtools.check.bin_edges`.
        _bins[-1] = bins[-1]
        bins = _bins
        del _bins
    else:
        # Read bin edges from file.
        if infile is None:
            analysis = "density-z"
            tool = "gmx"
            file_suffix = analysis + "_number_Li_binsA.txt.gz"
            infile = leap.simulation.get_ana_file(
                Sim, analysis, tool, file_suffix
            )
        kwargs.setdefault("usecols", 0)
        bins = np.loadtxt(infile, **kwargs)
    bin_edges_lower = bins[:-1]
    bin_edges_upper = bins[1:]
    bin_mids = bins[1:] - np.diff(bins) / 2
    bin_ix = np.arange(len(bin_mids))
    bin_data = np.row_stack(
        [bin_ix, bin_mids, bin_edges_lower, bin_edges_upper]
    )
    del bins, bin_ix, bin_mids, bin_edges_lower, bin_edges_upper

    if use_bin_ix is not None:
        # Select the desired bins.
        use_bin_ix = mdt.check.array(use_bin_ix, dim=1)
        if not np.all(np.isin(use_bin_ix, bin_data[0])):
            raise IndexError(
                "Simulation: '{}'.\n".format(Sim.path)
                + "Bin file: '{}'.\n".format(infile)
                + "Some of the given bin indices are out of bounds.\n"
                + "`bin_ix`     = {}.\n".format(bin_data[0])
                + "`use_bin_ix` = {}.".format(use_bin_ix)
            )
        use_bin_ix = np.flatnonzero(np.isin(bin_data[0], use_bin_ix))
        bin_data = bin_data[:, use_bin_ix]

    bin_is_left = bin_data[1] <= (box_z / 2)  # bin_data[1] = bin_mids.
    if np.any(bin_data[1] <= elctrd_thk):
        raise ValueError(
            "Simulation: '{}'.\n".format(Sim.path)
            + "Bin file: '{}'.\n".format(infile)
            + "At least one bin lies within the left electrode.\n"
            + "Bin mid points: {}.\n".format(bin_data[1])
            + "Left electrode: {}.".format(elctrd_thk)
        )
    if np.any(bin_data[1] >= box_z - elctrd_thk):
        raise ValueError(
            "Simulation: '{}'.\n".format(Sim.path)
            + "Bin file: '{}'.\n".format(infile)
            + "At least one bin lies within the right electrode.\n"
            + "Bin mid points: {}.\n".format(bin_data[1])
            + "Right electrode: {}.".format(box_z - elctrd_thk)
        )

    pk_pos_types = ("left", "right")
    for pkt_ix, pkp_type in enumerate(pk_pos_types):
        if pkp_type == "left":
            valid_bins = bin_is_left
            bin_data_valid = bin_data[:, valid_bins]
            if rel_bins:
                # Convert absolute bin positions to distances to the
                # electrodes.
                bin_data_valid[1:] -= elctrd_thk
        elif pkp_type == "right":
            valid_bins = ~bin_is_left
            bin_data_valid = bin_data[:, valid_bins]
            if rel_bins:
                # Reverse the order of rows to sort bins as function of
                # the distance to the electrodes in ascending order.
                bin_data_valid = bin_data_valid[:, ::-1]
                # Convert absolute bin positions to distances to the
                # electrodes.
                bin_data_valid[1:] += elctrd_thk
                bin_data_valid[1:] -= box_z
                bin_data_valid[1:] *= -1  # Ensure positive distance values.
        else:
            raise ValueError(
                "Unknown peak position type: '{}'".format(pkp_type)
            )
