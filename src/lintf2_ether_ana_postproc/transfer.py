"""
Module containing functions specific to the MD simulations in the
:file:`transfer_Li` directory, i.e. simulations in which a single
lithium ion has been transferred from the negative electrode surface to
the positive electrode surface.  In the following, these simulations are
called 'transfer' simulations.
"""


# Standard libraries
import glob
import os
import re

# First-party libraries
import lintf2_ether_ana_postproc as leap


START_PATTERN = "after_[0-9]*ns"
"""
Glob pattern matching the names of the directories that contain all
'transfer' MD simulations that start from the same snapshot/time of the
underlying equilibrium MD simulation.  Can also be used as regular
expression.

:type: str
"""

LI_PATTERN = "Li[0-9]*_transferred"
"""
Glob pattern matching the names of the directories that contain the
simulations for a specific transferred lithium ion.  Can also be used as
regular expression.

:type: str
"""


def extract_system_from_start_dir(*paths):
    """
    Extract the system name from the paths to the directories that
    contain all 'transfer' MD simulations that start from the same
    snapshot/time of the underlying equilibrium MD simulation.

    Parameters
    ----------
    paths : str or bytes or os.PathLike
        Relative or absolute paths to the directories that contain all
        'transfer' MD simulations that start from the same snapshot/time
        of the underlying equilibrium MD simulation.

    Returns
    -------
    systems : list
        List of system names.  The length of the list is equal to the
        number of given paths.

    Raises
    ------
    FileNotFoundError :
        If a given path is not an existing directory.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.transfer.get_start_dirs` :
        Get the paths to the directories that contain all 'transfer' MD
        simulations that start from the same snapshot/time of the
        underlying equilibrium MD simulation for a given system.
    """
    systems = []
    for path in paths:
        path = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isdir(path):
            raise FileNotFoundError("No such directory: '{}'".format(path))

        tail = os.path.split(path)[1]
        if not re.fullmatch(START_PATTERN, tail):
            raise ValueError(
                "Invalid start directory for 'transfer' simulations:"
                " '{}'".format(path)
            )

        systems.append(os.path.split(os.path.split(path)[0])[1])
    return systems


def extract_start_time_from_start_dir(*paths):
    """
    Extract the start times from the names of the directories that
    contain all 'transfer' MD simulations that start from the same
    snapshot/time of the underlying equilibrium MD simulation.

    The start times are the times at which the snapshots used
    to start the 'transfer' simulations were taken from the underlying
    equilibrium simulation.

    Parameters
    ----------
    paths : str or bytes or os.PathLike
        Relative or absolute paths to the directories that contain all
        'transfer' MD simulations that start from the same snapshot/time
        of the underlying equilibrium MD simulation.

    Returns
    -------
    start_times : list
        List of integers representing the times at which the snapshots
        used to start the 'transfer' simulations were taken from the
        underlying equilibrium simulation.

    Raises
    ------
    FileNotFoundError :
        If a given path is not an existing directory.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.transfer.get_start_dirs` :
        Get the paths to the directories that contain all 'transfer' MD
        simulations that start from the same snapshot/time of the
        underlying equilibrium MD simulation for a given system.
    """
    start_times = []
    for path in paths:
        path = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isdir(path):
            raise FileNotFoundError("No such directory: '{}'".format(path))

        tail = os.path.split(path)[1]
        if not re.fullmatch(START_PATTERN, tail):
            raise ValueError(
                "Invalid start directory for 'transfer' simulations:"
                " '{}'".format(path)
            )

        start_times.append(int("".join(c for c in tail if c.isdigit())))
    return start_times


def get_start_dirs(
    sys_pat, start_pat=START_PATTERN, sort_by_start=True, sort_by_system=True
):
    """
    Get the paths to the directories that contain all 'transfer' MD
    simulations that start from the same snapshot/time of the underlying
    equilibrium MD simulation for a given system.

    Parameters
    ----------
    sys_pat : str
        System name glob pattern.
    start_pat : str, optional
        Start directory glob pattern.
    sort_by_start : bool, optional
        If ``True``, sort the paths by the start times, i.e. the times
        at which the snapshots used to start the 'transfer' simulations
        were taken from the underlying equilibrium simulation.
    sort_by_system : bool, optional
        If ``True``, sort the paths by the system name.  Only relevant
        if `sys_pat` matches more than one system.  The sort by system
        name is performed after the sort by start time.

    Returns
    -------
    start_paths : list
        List of paths to the directories that contain all 'transfer' MD
        simulations that start from the same snapshot/time of the
        underlying equilibrium MD simulation for a given system.

    Raises
    ------
    FileNotFoundError :
        If no directory was found for the given system name.
    """
    SimPaths = leap.simulation.SimPaths()
    sys_pat = os.path.join(SimPaths.PATHS["transfer_Li"], sys_pat)
    start_pat = os.path.join(sys_pat, start_pat)
    start_paths = glob.glob(start_pat)

    if len(start_paths) == 0:
        raise FileNotFoundError(
            "Could not find any file/directory matching the pattern"
            " '{}'".format(start_pat)
        )
    for path in start_paths:
        if not os.path.isdir(path):
            raise FileNotFoundError("No such directory: '{}'".format(path))
        tail = os.path.split(path)[1]
        if not re.fullmatch(START_PATTERN, tail):
            raise ValueError(
                "Invalid start directory for 'transfer' simulations:"
                " '{}'".format(path)
            )

    if sort_by_start:
        # About the sort key:
        # `leap.transfer.extract_start_time_from_start_dir` returns a
        # **list** with one element (the start time of the given start
        # directory).  Returning a list is not an issue, because the
        # list only contains one element and Python compares lists
        # lexicographically.
        # https://docs.python.org/3/tutorial/datastructures.html#comparing-sequences-and-other-types
        start_paths = sorted(
            start_paths, key=leap.transfer.extract_start_time_from_start_dir
        )
    if sort_by_system:
        # `sorted` is guaranteed to be stable, i.e. the relative order
        # of elements that compare equal is not changed => Previous
        # sorts are not messed up.
        start_paths = sorted(
            start_paths, key=leap.transfer.extract_system_from_start_dir
        )

    return start_paths


def get_sims(sys_pat, set_pat, kwargs_start_dirs=None, kwargs_sims=None):
    """
    Create a :class:`~lintf2_ether_ana_postproc.simulation.Simulations`
    instance from glob patterns.

    Parameters
    ----------
    sys_pat : str
        System name glob pattern.
    set_pat : str
        Simulations settings name glob pattern.
    kwargs_start_dirs : dict, optional
        Keyword arguments to parse to
        :func:`lintf2_ether_ana_postproc.transfer.get_start_dirs`.  See
        there for possible options.
    kwargs_sims : dict, optional
        Keyword arguments to parse to
        :func:`lintf2_ether_ana_postproc.simulation.Simulations`.  See
        there for possible options.

    Returns
    -------
    sims_dct : dict of dict
        Dictionary containing for each system and each start time the
        corresponding
        :class:`~lintf2_ether_ana_postproc.simulation.Simulations`
        instance.
    """
    if kwargs_start_dirs is None:
        kwargs_start_dirs = {}
    if kwargs_sims is None:
        kwargs_sims = {}

    sims_dct = {}
    start_paths = leap.transfer.get_start_dirs(sys_pat, **kwargs_start_dirs)
    for start_path in start_paths:
        sim_pat = "[0-9][0-9]_" + set_pat + "_" + sys_pat + "_" + LI_PATTERN
        sim_pat = os.path.join(start_path, LI_PATTERN, sim_pat)
        sim_paths = glob.glob(sim_pat)
        if len(sim_paths) == 0:
            raise FileNotFoundError(
                "Could not find any file/directory matching the pattern"
                " '{}'".format(sim_pat)
            )
        system = leap.transfer.extract_system_from_start_dir(start_path)[0]
        t0 = leap.transfer.extract_start_time_from_start_dir(start_path)[0]
        sims_dct.setdefault(system, {})
        sims_dct[system][t0] = leap.simulation.Simulations(
            *sim_paths, **kwargs_sims
        )

    return sims_dct
