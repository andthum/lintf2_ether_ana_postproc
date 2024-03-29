"""Module to handle Gromacs MD simulations."""


# Standard libraries
import glob
import os
import re
import warnings

# Third-party libraries
import MDAnalysis as mda
import numpy as np
from scipy import constants as const

# First-party libraries
import lintf2_ether_ana_postproc as leap


class SimPaths:
    """
    Class containing the top-level paths to the Gromacs MD simulations.

    All attributes should be treated as read-only attributes.  Don't
    change them after initialization.
    """

    def __init__(self, root=None):
        """
        Initialize a
        :class:`~lintf2_ether_ana_postproc.simulation.SimPaths`
        instance.

        Parameters
        ----------
        root : str or bytes or os.PathLike or None, optional
            Path of the root directory that contains all Gromacs MD
            simulations.  If `root` is ``None``, it is set to
            ``"${HOME}/ownCloud/WWU_Münster/Promotion/Simulationen/results/lintf2_peo"``

        Raises
        ------
        FileNotFoundError
            If the root directory or one of the required sub-directories
            does not exist.
        """  # noqa: E501
        if root is None:
            root = "${HOME}/ownCloud/WWU_Münster/Promotion/Simulationen/results/lintf2_peo"  # noqa: E501
        root = os.path.abspath(os.path.expandvars(os.path.expanduser(root)))

        self.PATHS = {"root": root}
        """
        Top-level directories containing the Gromacs MD simulations.

        :type: dict
        """

        for path in ("bulk", "walls", "transfer_Li", "flux_Li"):
            self.PATHS[path] = os.path.normpath(os.path.join(root, path))
        for path in ("q0", "q0.25", "q0.5", "q0.75", "q1"):
            self.PATHS[path] = os.path.normpath(
                os.path.join(self.PATHS["walls"], path)
            )
        for path in self.PATHS.values():
            if not os.path.isdir(path):
                raise ValueError("No such directory: '{}'".format(path))


class Electrode:
    """
    Class containing the properties of a single graphene electrode used
    in the simulations.

    All attributes should be treated as read-only attributes.  Don't
    change them after initialization.
    """

    def __init__(self):
        self.GRA_SIGMA_LJ = 3.55
        """
        Lennard-Jones radius of graphene atoms in Angstroms.

        :type: float
        """

        self.GRA_LAYERS_N = 3
        """
        Number of graphene layers.

        :type: int
        """

        self.GRA_LAYER_DIST = 3.35
        """
        Distance between two graphene layers in Angstroms.

        :type: float
        """

        self.ELCTRD_THK = (self.GRA_LAYERS_N - 1) * self.GRA_LAYER_DIST
        """
        Electrode thickness in Angstroms.

        :type: float
        """

        self.BULK_START = 40.0
        """
        Distance to the electrodes in Angstroms at which the bulk region
        starts.

        This is just a subjective value I defined from visual inspection
        of the density profiles.  There is no underlying calculation or
        objective criterion behind it.

        :type: float
        """


class Simulation:
    """
    A Class representing a single Gromacs molecular dynamics simulation.

    All attributes should be treated as read-only attributes.  Don't
    change them after initialization.
    """

    sys_prefix = "lintf2_"
    """
    Prefix of all system names.

    :type: str
    """

    bulk_sys_suffix = "_sc80"
    """
    Suffix of the system name of all bulk simulations.

    :type: str
    """

    elctrd_regex = r"_gra_q[0-9][0-9\.]*[0-9]*"
    """
    Regular expression string to match the part describing the electrode
    in a system name.

    :type: str
    """

    dens_file_cmp2col = {
        "elctrlyt": 1,
        "Li": 2,
        "ntf2": 3,
        "sol": 4,
        "NBT": 5,
        "OBT": 6,
        "OE": 7,
        "graB": 8,
        "graT": 9,
        "elctrd": 10,
        "system": 11,
    }
    """
    Dictionary containing for each compound the corresponding column
    number of the density profile files produced by :bash:`gmx density`.

    :type: dict
    """

    def __init__(self, path):
        """
        Initialize an instance of the
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation` class.

        Parameters
        ----------
        path : str or bytes or os.PathLike
            Relative or absolute path to a directory containing the
            input and output files of a Gromacs MD simulation.

            The directory must be named according to the following
            pattern: ``"NN_" + settings + "_" + system``, where
            ``system`` is the name of the simulated system, ``settings``
            can be any string describing the used simulation settings
            and "N" stands for a digit (but can in fact be any
            character).

            The name of the system must follow the pattern
            "lintf2_solvent_EO-Li-ratio_sc80(_additional_description)"
            (e.g. lintf2_g1_20-1_sc80) or
            "lintf2_solvent_EO-Li-ratio_gra_surface-charge_sc80(_additional_description)"
            (e.g. lintf2_peo15_20-1_gra_q0.5_sc80), where
            "_additional_description" is optional.

            The toplevel directory of the given simulation directory
            must be named after the system: ``system``.

            The input files of the simulation must be named
            ``settings + "_" + system + file_extension`` and the output
            files must be named ``settings + "_out_" + system +
            file_extension`` (see :meth:`self.get_sim_files`).
        """
        # Instance attributes.
        self.path = None
        """
        Path to the simulation directory.

        :type: str
        """

        self.path_ana = None
        """
        Path to the corresponding analysis directory.

        :type: str
        """

        self.fname_ana_base = None
        """
        The base name of corresponding analysis files.

        :type: str
        """

        self.system = None
        """
        Name of the simulated system.

        :type: str
        """

        self.settings = None
        """
        String describing the used simulation settings.

        :type: str
        """

        self.is_bulk = None
        """
        ``True`` if the simulation is a bulk simulation, ``False``
        otherwise.

        :type: bool
        """

        self._bulk_sim_path = None
        """
        Path to the corresponding bulk simulation (only relevant for
        surface simulations).

        :type: str
        """

        self._BulkSim = None
        """
        The corresponding bulk
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation` (only
        relevant for surface simulations).

        :type: :class:`lintf2_ether_ana_postproc.simulation.Simulation`
        """

        self.surfq = None
        """
        Surface charge of the electrodes (if present) in e/Angstrom^2 as
        inferred from :attr:`self.system`.

        :type: float
        """

        self._surfq_u = None
        """
        Surface charge of the electrodes (if present) in e/Angstrom^2 as
        inferred from :attr:`self._Universe`.

        See Also
        --------
        :attr:`self.surfq` :
            Surface charge of the electrodes inferred from
            :attr:`self.system`

        :type: float
        """

        self.temp = None
        """
        Temperature of the simulated system in K as inferred from
        :attr:`self.settings`.

        :type: float
        """

        self.sim_files = None
        """
        Dictionary containing the paths of the input and output files of
        the simulation.

        :type: dict
        """

        self.res_names = None
        """
        Dictionary containing the residue names of the cation, the anion
        and the solvent as inferred from :attr:`self.system`.

        :type: dict
        """

        self._res_name_solvent = None
        """
        Full residue name of the solvent as inferred from
        :attr:`self.system`.

        For residue names containing 'peo', the chain length is removed
        in :attr:`self.res_names`.  :attr:`self._res_name_solvent`
        contains the full residue name with the chain length (i.e.
        number of monomers per chain).

        :type: str
        """

        self._Universe = None
        """
        MDAnalysis :class:`~MDAnalysis.core.universe.Universe` created
        from the output .gro file of the simulation.

        :type: :class:`MDAnalysis.core.universe.Universe`
        """

        self.box = None
        """
        The simulation box dimensions in the same format as returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.

        Lengths are given in Angstroms, angles in degrees.

        :type: numpy.ndarray
        """

        self.top_info = None
        """
        Information about the topology of the simulated system.

        Dictionary containing the number of atoms of each atom type and
        name for the entire system and for each individual residue.

        :type: dict
        """

        self.O_per_chain = None
        """
        Number of ether oxygens per PEO chain.

        :type: int
        """

        self.O_Li_ratio = None
        """
        Ratio of ether oxygen atoms to lithium ions.

        See Also
        --------
        :attr:`self.Li_O_ratio` :
            Ratio of lithium ions to ether oxygen atoms

        :type: float
        """

        self._O_Li_ratio_sys = None
        """
        Ratio of ether oxygen atoms to lithium ions as inferred from
        :attr:`self.system`.

        See Also
        --------
        :attr:`self.O_Li_ratio` :
            Ratio of ether oxygen atoms to lithium ions calculated from
            :attr:`self.top_info` and :attr:`self.res_names`

        :type: float
        """

        self.Li_O_ratio = None
        """
        Ratio of lithium ions to ether oxygen atoms.

        See Also
        --------
        :attr:`self.O_Li_ratio` :
            Ratio of ether oxygen atoms to lithium ions

        :type: float
        """

        self.vol = None
        """
        Total and accessible volume of the simulation box in Angstrom
        cubed.

        The simulation box is read from the output .gro file of the
        simulation.  The accessible volume is the total volume minus the
        volume of the electrodes (if present).

        :type: dict
        """

        self.dens = None
        """
        Mass and number densities of all atom types and all electrolyte
        components.

        Mass densities are given in u/A^3, number densities are given in
        1/A^3.

        The densities will be calculated with respect to the accessible
        volume of the simulation box, i.e. the volume of the electrodes
        is subtracted from the total box volume.

        The simulation box is taken from ``self.gro_file``.

        :type: dict
        """

        self.bulk_region = None
        """
        Begin and end of the bulk region of the simulation box along the
        z direction in Angstrom.

        For bulk simulations, this is the entire simulation box from
        zero to :math:`l_z`.  For surface simulations, this is the
        region from
        :attr:`Electrode.ELCTRD_THK` + :attr:`Electrode.BULK_START` to
        :attr:`self.box[2]` - :attr:`Electrode.ELCTRD_THK` -
        :attr:`Electrode.BULK_START`.

        :type: numpy.ndarray of two floats
        """

        path = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isdir(path):
            raise FileNotFoundError("No such directory: '{}'".format(path))
        self.path = os.path.abspath(path)
        self.Elctrd = leap.simulation.Electrode()
        self.get_system()
        self.get_settings()
        self.get_path_ana()
        self.get_fname_ana_base()
        self.get_is_bulk()
        self._get_BulkSim()
        self.get_surfq()
        self.get_temp()
        self.get_sim_files()
        self.get_res_names()
        self._get_Universe()
        self.get_box()
        self.get_top_info()
        self.get_O_per_chain()
        self.get_O_Li_ratio()
        self.get_Li_O_ratio()
        self.get_vol()
        self.get_dens()
        self.get_bulk_region()

        # Release memory.
        del self._Universe, self._BulkSim
        self._Universe = None
        self._BulkSim = None

    def get_system(self):
        """
        Get the name of the simulated system.

        Return the value of :attr:`self.system`.  If :attr:`self.system`
        is ``None``, infer the system name from the name of toplevel
        directory of the simulation directory (:attr:`self.path`).

        Returns
        -------
        self.system : str
            The name of the simulated system.
        """
        if self.system is not None:
            return self.system

        if not os.path.isdir(self.path):
            raise FileNotFoundError(
                "No such directory: '{}'".format(self.path)
            )

        self.system = os.path.basename(self.path)
        if self.sys_prefix not in self.system:
            raise ValueError(
                "Could not infer the system name from the path basename ('{}')"
                " because the path basename does not contain the common system"
                " prefix ('{}')".format(self.system, self.sys_prefix)
            )
        ix = self.system.find(self.sys_prefix)
        self.system = self.system[ix:]
        if len(self.system) == 0:
            raise ValueError(
                "Unexpected error: Could not infer the system name from the"
                " path ('{}')".format(self.path)
            )
        return self.system

    def get_settings(self):
        """
        Get the string describing the used simulation settings.

        Return the value of :attr:`self.settings`.  If
        :attr:`self.settings` is ``None``, infer settings string from
        :attr:`self.path` and :attr:`self.system`.

        Returns
        -------
        self.settings : str
            The string describing the used simulation settings.
        """
        if self.settings is not None:
            return self.settings

        if not os.path.isdir(self.path):
            raise FileNotFoundError(
                "No such directory: '{}'".format(self.path)
            )
        if self.system is None:
            self.get_system()

        if not self.path.endswith(self.system):
            raise ValueError(
                "`path` ({}) must end with the system name"
                " ({})".format(self.path, self.system)
            )
        self.settings = os.path.basename(self.path)
        # Remove system name from settings.
        self.settings = self.settings.split(self.system)[0]
        if self.system in self.settings:
            raise ValueError(
                "The path basename ({}) must contain the system name ({}) at"
                " the end but nowhere"
                " else".format(os.path.basename(self.path), self.system)
            )
        # Remove preceding numbers and underscores.
        while self.settings[0].isdigit() or self.settings.startswith("_"):
            self.settings = self.settings[1:]
        # Remove preceding and trailing underscores.
        self.settings = self.settings.strip("_")
        return self.settings

    def get_path_ana(self):
        """
        Get the path to the corresponding analysis directory.

        Return the value of :attr:`self.path_ana`.  If
        :attr:`self.path_ana` is ``None``, infer the path to the
        analysis from :attr:`self.path`, :attr:`self.system` and
        :attr:`self.settings`.

        Returns
        -------
        self.path_ana : str
            Path to the corresponding analysis directory.

        Raises
        ------
        FileNotFoundError :
            If the corresponding analysis directory cannot be found.
        """
        if self.path_ana is not None:
            return self.path_ana

        if not os.path.isdir(self.path):
            raise FileNotFoundError(
                "No such directory: '{}'".format(self.path)
            )
        if self.system is None:
            self.get_system()
        if self.settings is None:
            self.get_settings()

        self.path_ana = "ana_" + self.settings + "_" + self.system
        self.path_ana = os.path.join(self.path, self.path_ana)
        if not os.path.isdir(self.path_ana):
            raise FileNotFoundError(
                "Could not find the corresponding analysis directory.  No such"
                " directory: '{}'".format(self.path_ana)
            )
        return self.path_ana

    def get_fname_ana_base(self):
        """
        Get the base name of corresponding analysis files.

        Return the value of :attr:`self.fname_ana_base`.  If
        :attr:`self.fname_ana_base` is ``None``, infer the base name of
        analysis files from :attr:`self.system` and
        :attr:`self.settings`.

        Returns
        -------
        self.fname_ana_base : str
            The base name of corresponding analysis files.
        """
        if self.fname_ana_base is not None:
            return self.fname_ana_base

        if self.system is None:
            self.get_system()
        if self.settings is None:
            self.get_settings()

        self.fname_ana_base = self.settings + "_" + self.system + "_"
        return self.fname_ana_base

    def get_is_bulk(self):
        """
        Get the value of :attr:`self.is_bulk`.  If :attr:`self.is_bulk`
        is ``None``, infer the value from :attr:`self.system`.

        Returns
        -------
        self.is_bulk : bool
            ``True`` if the simulation is a bulk simulation, ``False``
            otherwise.
        """
        if self.is_bulk is not None:
            return self.is_bulk

        if self.system is None:
            self.get_system()

        if re.search(self.elctrd_regex, self.system) is None:
            self.is_bulk = True
        else:
            self.is_bulk = False
        return self.is_bulk

    def _get_BulkSim_path(self, root=None):
        """
        Get the path to the corresponding bulk simulation (only relevant
        for surface simulations).

        Return the value of :attr:`self._bulk_sim_path`.  If
        :attr:`self._bulk_sim_path` is ``None``, assemble the path from
        :attr:`self.path`, :attr:`self.system`, and
        :attr:`self.settings`.

        Parameters
        ----------
        root : str or bytes or os.PathLike or None, optional
            Path of the root directory that contains all Gromacs MD
            simulations.  See :meth:`SimPaths.__init__`.

        Returns
        -------
        self._bulk_sim_path : str
            The path to the corresponding bulk simulation.  If the
            current simulation is already a bulk simulation, this is
            the same as :attr:`self.path`.
        """
        if self._bulk_sim_path is not None:
            return self._bulk_sim_path

        if not os.path.isdir(self.path):
            raise FileNotFoundError(
                "No such directory: '{}'".format(self.path)
            )
        if self.system is None:
            self.get_system()
        if self.settings is None:
            self.get_settings()
        if self.is_bulk is None:
            self.get_is_bulk()

        if self.is_bulk:
            # The simulation is already a bulk simulation.
            self._bulk_sim_path = self.path
            return self._bulk_sim_path

        SimPaths = leap.simulation.SimPaths(root=root)
        bulk_sys = re.sub(self.elctrd_regex, "", self.system)
        if self.bulk_sys_suffix not in bulk_sys:
            warnings.warn(
                "Could not find the corresponding bulk simulation.  The system"
                " name ('{}') does not contain the bulk system suffix"
                " ('{}')".format(self.system, self.bulk_sys_suffix),
                UserWarning,
                stacklevel=2,
            )
            self._bulk_sim_path = None
            return self._bulk_sim_path
        elif not bulk_sys.endswith(self.bulk_sys_suffix):
            # Allow "_additional_description" (like "Li104_transferred"
            # or "flux") at the end of the system name which is not
            # present in the bulk simulation.
            ix = bulk_sys.rfind(self.bulk_sys_suffix)
            bulk_sys = bulk_sys[: ix + len(self.bulk_sys_suffix)]
        bulk_sys_path = os.path.join(SimPaths.PATHS["bulk"], bulk_sys)
        if not os.path.isdir(bulk_sys_path):
            warnings.warn(
                "Could not find the corresponding bulk simulation.  No such"
                " directory: '{}'".format(bulk_sys_path),
                UserWarning,
                stacklevel=2,
            )
            self._bulk_sim_path = None
            return self._bulk_sim_path

        glob_pattern = os.path.join(
            bulk_sys_path, "[0-9][0-9]_" + self.settings + "_" + bulk_sys
        )
        bulk_sim_path = glob.glob(glob_pattern)
        if len(bulk_sim_path) == 0:
            warnings.warn(
                "Could not find the corresponding bulk simulation.  The glob"
                " pattern '{}' does not match anything".format(glob_pattern),
                UserWarning,
                stacklevel=2,
            )
            self._bulk_sim_path = None
            return self._bulk_sim_path
        elif len(bulk_sim_path) > 1:
            raise ValueError(
                "Found multiple bulk simulations matching the glob pattern"
                " '{}'".format(glob_pattern)
            )
        bulk_sim_path = bulk_sim_path[0]
        if not os.path.isdir(bulk_sim_path):
            warnings.warn(
                "Could not find the corresponding bulk simulation.  No such"
                " directory: '{}'".format(bulk_sim_path),
                UserWarning,
                stacklevel=2,
            )
            self._bulk_sim_path = None
            return self._bulk_sim_path

        self._bulk_sim_path = bulk_sim_path
        return self._bulk_sim_path

    def _get_BulkSim(self):
        """
        Get the corresponding bulk
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation` (only
        relevant for surface simulations).

        Return the value of :attr:`self._BulkSim`.  If
        :attr:`self._BulkSim` is ``None``, create the bulk
        :class:`Simulation` from :attr:`self._bulk_sim_path`.

        Returns
        -------
        self._BulkSim :
            :class:`lintf2_ether_ana_postproc.simulation.Simulation`

            The corresponding bulk
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.
            If the current
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation` is
            already a bulk simulation, this is ``None``.
        """
        if self._BulkSim is not None:
            return self._BulkSim

        if self._bulk_sim_path is None:
            self._get_BulkSim_path()

        if self.is_bulk:
            # This simulation is already a bulk simulation.
            self._BulkSim = None
        elif self._bulk_sim_path is None:
            # This simulation has no corresponding bulk simulation.
            self._BulkSim = None
        else:
            self._BulkSim = leap.simulation.Simulation(self._bulk_sim_path)
        return self._BulkSim

    def get_surfq(self):
        """
        Get the surface charge of the electrodes (if present) in
        e/Angstrom^2 as inferred from :attr:`self.settings`.

        Return the value of :attr:`self.surfq`.  If :attr:`self.surfq`
        is ``None``, infer the surface charge from
        :attr:`self.settings`.

        Returns
        -------
        self.surfq : float
            The surface charge of the electrodes in e/Angstrom^2.  If
            the system contains no electrodes, this is ``None``.
        """
        if self.surfq is not None:
            return self.surfq

        if self.system is None:
            self.get_system()
        if self.is_bulk is None:
            self.get_is_bulk()

        if self.is_bulk:
            self.surfq = None
            return self.surfq

        self.surfq = leap.simulation.get_surfq(self.system)
        self.surfq /= 100  # e/nm^2 -> e/Angstrom^2

        if self._surfq_u is not None and not np.isclose(
            self._surfq_u, self.surfq, rtol=0, atol=1e-3
        ):
            raise ValueError(
                "`self._surfq_u` ({}) != `self.surfq`"
                " ({})".format(self._surfq_u, self.surfq)
            )

        return self.surfq

    def _get_surfq_u(self):
        """
        Get the surface charge of the electrodes (if present) in
        e/Angstrom^2 as inferred from :attr:`self._Universe`.

        Return the value of :attr:`self._surfq_u`.  If
        :attr:`self._surfq_u` is ``None``, infer the surface charge from
        :attr:`self._Universe`.

        Returns
        -------
        self._surfq_u : float
            The surface charge of the electrodes in e/Angstrom^2.  If
            the system contains no electrodes, this is ``None``.

        See Also
        --------
        :meth:`self.get_surfq` :
            Get the surface charge of the electrodes as inferred from
            :attr:`self.settings`.
        """
        if self._surfq_u is not None:
            return self._surfq_u

        if self.system is None:
            self.get_system()
        if self.is_bulk is None:
            self.get_is_bulk()
        if self.res_names is None:
            # `self.res_names` contains all non-electrode residue names.
            self.get_res_names()
        if self._Universe is None:
            self._get_Universe()
        if self.box is None:
            self.get_box()

        if self.is_bulk:
            self._surfq_u = None
            return self._surfq_u

        elctrds = "not resname "
        elctrds += " and not resname ".join(self.res_names.values())
        elctrds = self._Universe.select_atoms(elctrds)
        if elctrds.n_atoms == 0:
            raise ValueError(
                "This is a surface simulation but the system contains no"
                " electrodes"
            )
        self._surfq_u = np.sum(np.abs(elctrds.charges)) / 2
        self._surfq_u /= np.prod(self.box[:2])

        if self.surfq is not None and not np.isclose(
            self._surfq_u, self.surfq, rtol=0, atol=1e-3
        ):
            raise ValueError(
                "`self._surfq_u` ({}) != `self.surfq`"
                " ({})".format(self._surfq_u, self.surfq)
            )

        return self._surfq_u

    def get_temp(self):
        """
        Get the temperature of the simulated system in K as inferred
        from :attr:`self.settings`.

        Return the value of :attr:`self.temp`.  If :attr:`self.temp` is
        ``None``, infer the temperature from :attr:`self.settings`.

        Returns
        -------
        self.temp : float
            The temperature of the simulated system in K.
        """
        if self.temp is not None:
            return self.temp

        if self.settings is None:
            self.get_settings()

        settings = self.settings.split("_")
        for setting in settings:
            if setting.startswith("nvt"):
                self.temp = "".join(char for char in setting if char.isdigit())
                break
            elif setting.startswith("npt"):
                self.temp = "".join(char for char in setting if char.isdigit())
                break
        if self.temp is None:
            raise ValueError(
                "Could not infer the temperature from `self.settings`"
                " ({})".format(self.settings)
            )
        self.temp = float(self.temp)

        if self._BulkSim is not None and not np.isclose(
            self.temp, self._BulkSim.temp, rtol=0
        ):
            raise ValueError(
                "`self.temp` ({}) != `self._BulkSim.temp`"
                " ({})".format(self.temp, self._BulkSim.temp)
            )

        return self.temp

    def get_sim_files(self):
        """
        Get the paths of the input and output files of the simulation.

        Return the value of :attr:`self.sim_files`.  If
        :attr:`self.sim_files` is ``None``, assemble the file names
        from :attr:`self.system` and :attr:`self.settings`:

            * ``settings + "_" + system + ".mdp"``.
            * ``settings + "_" + system + ".tpr"``.
            * ``settings + "_out_" + system + ".edr.gz"``.
            * ``settings + "_out_" + system + ".gro"``.
            * ``settings + "_out_" + system + ".log.gz"``.

        Returns
        -------
        self.sim_files : dict
            The paths of the input and output files of the simulation.

        Raises
        ------
        FileNotFoundError :
            If a file cannot be found.
        """
        if self.sim_files is not None:
            return self.sim_files

        if self.system is None:
            self.get_system()
        if self.settings is None:
            self.get_settings()

        files_in = ("mdp", "tpr")
        files_out = ("edr.gz", "gro", "log.gz")

        self.sim_files = {}
        for ext in files_in:
            f = self.settings + "_" + self.system + "." + ext
            self.sim_files[ext] = os.path.join(self.path, f)
        for ext in files_out:
            f = self.settings + "_out_" + self.system + "." + ext
            self.sim_files[ext] = os.path.join(self.path, f)

        for f in self.sim_files.values():
            if not os.path.isfile(f):
                raise FileNotFoundError("No such file: '{}'".format(f))

        return self.sim_files

    def get_res_names(self):
        """
        Get the the residue names of the cation, the anion and the
        solvent as inferred from :attr:`self.system`.

        Return the value of :attr:`self.res_names`.  If
        :attr:`self.res_names` is ``None``, infer the residue names from
        the system name (:attr:`self.system`).

        Returns
        -------
        self.res_names : dict
            The residue names of the cation, the anion and the solvent.
        """
        if self.res_names is not None:
            return self.res_names

        if self.system is None:
            self.get_system()

        if (
            not self.system.startswith(self.sys_prefix)
            or self.system.count("_") < 2
        ):
            raise ValueError(
                "The system name ('{}') must start with"
                " '{}<solvent>_'".format(self.system, self.sys_prefix)
            )
        salt = self.system.split("_")[0]
        solvent = self.system.split("_")[1]
        self.res_names = {}
        self.res_names["cation"] = salt[:2]
        self.res_names["anion"] = salt[2:]
        self._res_name_solvent = solvent
        if "peo" in solvent:
            # Remove digits from the solvent name, because in my
            # simulations with PEO, the residue name of PEO is always
            # "peo" regardless of the chain length.
            solvent = "".join(char for char in solvent if not char.isdigit())
        self.res_names["solvent"] = solvent

        if self.top_info is not None and any(
            rn not in self.top_info["res"] for rn in self.res_names.values()
        ):
            raise ValueError(
                "At least one of the residue names in `self.res_names` ({}) is"
                " not contained in `self.top_info['res']` ({})".format(
                    self.res_names.values(), self.top_info["res"].keys()
                )
            )
        if (
            self._BulkSim is not None
            and self.res_names != self._BulkSim.res_names
        ):
            raise ValueError(
                "`self.res_names` ({}) != `self._BulkSim.res_names`"
                " ({})".format(self.res_names, self._BulkSim.res_names)
            )

        return self.res_names

    def _get_Universe(self):
        """
        Get an MDAnalysis :class:`~MDAnalysis.core.universe.Universe`
        created from the output .gro file of the simulation.

        Return the value of :attr:`self._Universe`.  If
        :attr:`self._Universe` is ``None``, create an MDAnalysis
        :class:`~MDAnalysis.core.universe.Universe` from the output .gro
        file of the simulation as given by :attr:`self.gro_file`.

        Returns
        -------
        self._Universe : MDAnalysis.core.universe.Universe
            The created :class:`~MDAnalysis.core.universe.Universe`.
        """
        if self._Universe is not None:
            return self._Universe

        if self.is_bulk is None:
            self.get_is_bulk()
        if self.sim_files is None:
            self.get_sim_files()

        self._Universe = mda.Universe(
            self.sim_files["tpr"], self.sim_files["gro"]
        )

        if not self.is_bulk:
            if self.surfq is None:
                self.get_surfq()
            if self._surfq_u is None:
                self._get_surfq_u()
            if not np.isclose(self._surfq_u, self.surfq, rtol=0, atol=1e-3):
                raise ValueError(
                    "`self._surfq_u` ({}) != `self.surfq`"
                    " ({})".format(self._surfq_u, self.surfq)
                )

            if "B1" not in self._Universe.residues.resnames:
                raise ValueError(
                    "The simulation is a surface simulation but it does not"
                    " contain the residue 'B1'"
                )
            elctrd_bot = self._Universe.select_atoms("resname B1")
            elctrd_bot_pos_z = elctrd_bot.positions[:, 2]
            if not np.allclose(
                elctrd_bot_pos_z, self.Elctrd.ELCTRD_THK, rtol=0, atol=1e-6
            ):
                raise ValueError(
                    "`elctrd_bot_pos_z` ({}) != `self.Elctrd.ELCTRD_THK`"
                    " ({})".format(
                        (np.min(elctrd_bot_pos_z), np.max(elctrd_bot_pos_z)),
                        self.Elctrd.ELCTRD_THK,
                    )
                )

            if "T1" not in self._Universe.residues.resnames:
                raise ValueError(
                    "The simulation is a surface simulation but it does not"
                    " contain the residue 'T1'"
                )
            elctrd_top = self._Universe.select_atoms("resname T1")
            elctrd_top_pos_z = elctrd_top.positions[:, 2]
            box_z = self._Universe.dimensions[2]
            if not np.allclose(
                elctrd_top_pos_z,
                box_z - self.Elctrd.ELCTRD_THK,
                rtol=0,
                atol=1e-2,
            ):
                raise ValueError(
                    "`elctrd_top_pos_z` ({}) !="
                    " `box_z - self.Elctrd.ELCTRD_THK` ({})".format(
                        (np.min(elctrd_top_pos_z), np.max(elctrd_top_pos_z)),
                        box_z - self.Elctrd.ELCTRD_THK,
                    )
                )

        return self._Universe

    def get_box(self):
        """
        Get the simulation box dimensions of the simulated system.

        Return the value of :attr:`self.box`.  If :attr:`self.box` is
        ``None``, get the box dimensions from :attr:`self._Universe`.

        Lengths are given in Angstroms, angles in degrees.

        Returns
        -------
        self.box : numpy.ndarray
            The simulation box dimensions in the same format as returned
            by :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
            ``[lx, ly, lz, alpha, beta, gamma]``.
        """
        if self.box is not None:
            return self.box

        if self._Universe is None:
            self._get_Universe()

        self.box = self._Universe.dimensions
        return self.box

    def get_top_info(self):
        """
        Get information about the topology of the simulated system.

        Return the value of :attr:`self.top_info`.  If
        :attr:`self.top_info` is ``None``, create it by extracting the
        relevant information from :attr:`self._Universe`.

        Returns
        -------
        self.top_info : dict
            Dictionary containing the number of atoms of each atom type
            and name for the entire system and for each individual
            residue.
        """
        if self.top_info is not None:
            return self.top_info

        if self.res_names is None:
            self.get_res_names()
        if self._Universe is None:
            self._get_Universe()

        ag = self._Universe.atoms
        sys_dct = {
            "atm_type": leap.simulation.num_atoms_per_type(ag, attr="types"),
            "atm_name": leap.simulation.num_atoms_per_type(ag, attr="names"),
        }
        n_res_per_name = leap.simulation.num_res_per_name(ag)
        res_dct = {}
        for rn, n_res in n_res_per_name.items():
            res_dct[rn] = {"n_res": n_res}
            ag_res = ag.select_atoms("resname {}".format(rn))
            n_atoms_per_type = leap.simulation.num_atoms_per_type(
                ag_res, attr="types"
            )
            res_dct[rn].update({"atm_type": n_atoms_per_type})
            n_atoms_per_name = leap.simulation.num_atoms_per_type(
                ag_res, attr="names"
            )
            res_dct[rn].update({"atm_name": n_atoms_per_name})
        self.top_info = {"sys": sys_dct, "res": res_dct}

        if any(
            rn not in self.top_info["res"] for rn in self.res_names.values()
        ):
            raise ValueError(
                "At least one of the residue names in `self.res_names` ({}) is"
                " not contained in `self.top_info['res']` ({})".format(
                    self.res_names.values(), self.top_info["res"].keys()
                )
            )
        if self._BulkSim is not None and any(
            self.top_info["res"][rn] != self._BulkSim.top_info["res"][rn]
            for rn in self.res_names.values()
        ):
            raise ValueError(
                "self.top_info['res'] ({}) != self._BulkSim.top_info['res']"
                " ({})".format(
                    self.top_info["res"], self._BulkSim.top_info["res"]
                )
            )

        return self.top_info

    def get_O_per_chain(self):
        """
        Get the number of ether oxygens per PEO chain.

        Return the value of :attr:`self.O_per_chain`.  If
        :attr:`self.O_per_chain` is ``None``, infer the value from
        :attr:`self.top_info` and :attr:`self.res_names`.

        Returns
        -------
        self.O_per_chain : float
            Number of ether oxygens per PEO chain.
        """
        if self.O_per_chain is not None:
            return self.O_per_chain

        if self.res_names is None:
            self.get_res_names()
        if self.top_info is None:
            self.get_top_info()

        solvent = self.res_names["solvent"]
        n_chains = self.top_info["res"][solvent]["n_res"]
        n_O = self.top_info["res"][solvent]["atm_type"]["OE"]
        O_per_chain = n_O / n_chains
        if not O_per_chain.is_integer():
            raise ValueError(
                "The number of ether oxygens per PEO chain ({}) is not an"
                " integer.  This should not have happened.".format(O_per_chain)
            )
        self.O_per_chain = int(O_per_chain)

        O_per_chain_check = "".join(
            char for char in self._res_name_solvent if char.isdigit()
        )
        O_per_chain_check = int(O_per_chain_check)
        # Number of oxygens per chain = Number of monomers plus one.
        O_per_chain_check += 1
        if self.O_per_chain != O_per_chain_check:
            raise ValueError(
                "`self.O_per_chain` ({}) != `O_per_chain_check`"
                " ({})".format(self.O_per_chain, O_per_chain_check)
            )
        if (
            self._BulkSim is not None
            and self.O_per_chain != self._BulkSim.O_per_chain
        ):
            raise ValueError(
                "`self.O_per_chain` ({}) != `self._BulkSim.O_per_chain`"
                " ({})".format(self.O_per_chain, self._BulkSim.O_per_chain)
            )

        return self.O_per_chain

    def get_O_Li_ratio(self):
        """
        Get the ratio of ether oxygen atoms to lithium ions.

        Return the value of :attr:`self.O_Li_ratio`.  If
        :attr:`self.O_Li_ratio` is ``None``, calculate the EO-Li ratio
        from :attr:`self.top_info` and :attr:`self.res_names`.

        Returns
        -------
        self.O_Li_ratio : float
            Ratio of ether oxygen atoms to lithium ions.

        See Also
        --------
        :func:`self.get_Li_O_ratio` :
            Get the ratio of lithium ions to ether oxygen atoms
        """
        if self.O_Li_ratio is not None:
            return self.O_Li_ratio

        if self.res_names is None:
            self.get_res_names()
        if self.top_info is None:
            self.get_top_info()

        cation = self.res_names["cation"]
        solvent = self.res_names["solvent"]
        n_O = self.top_info["res"][solvent]["atm_type"]["OE"]
        n_Li = self.top_info["res"][cation]["atm_type"]["Li"]
        self.O_Li_ratio = n_O / n_Li

        if self._O_Li_ratio_sys is None:
            self._get_O_Li_ratio_sys()
        if not np.isclose(self.O_Li_ratio, self._O_Li_ratio_sys, rtol=0):
            raise ValueError(
                "`self.O_Li_ratio` ({}) != `self._O_Li_ratio_sys`"
                " ({})".format(self.O_Li_ratio, self._O_Li_ratio_sys)
            )
        if self._BulkSim is not None and not np.isclose(
            self.O_Li_ratio, self._BulkSim.O_Li_ratio, rtol=0
        ):
            raise ValueError(
                "`self.O_Li_ratio` ({}) != `self._BulkSim.O_Li_ratio`"
                " ({})".format(self.O_Li_ratio, self._BulkSim.O_Li_ratio)
            )

        return self.O_Li_ratio

    def _get_O_Li_ratio_sys(self):
        """
        Get the ratio of ether oxygen atoms to lithium ions as inferred
        from :attr:`self.system`.

        Return the value of :attr:`self._O_Li_ratio_sys`.  If
        :attr:`self._O_Li_ratio_sys` is ``None``, infer the ratio of
        ether oxygen atoms to lithium ions from :attr:`self.system`.

        Returns
        -------
        self._O_Li_ratio_sys : float
            Ratio of ether oxygen atoms to lithium ions as inferred from
            :attr:`self.system`.

        See Also
        --------
        :func:`self.get_O_Li_ratio` :
            Get the ratio of ether oxygen atoms to lithium ions from
            :attr:`self.top_info` and :attr:`self.res_names`.
        """
        if self._O_Li_ratio_sys is not None:
            return self._O_Li_ratio_sys

        if self.system is None:
            self.get_system()

        if self.system.count("_") < 2:
            raise ValueError(
                "The system name ('{}') must start with"
                " '{}<solvent>_<EO-Li-ratio>'".format(
                    self.system, self.sys_prefix
                )
            )
        O_Li_ratio = self.system.split("_")[2]
        n_O, n_Li = O_Li_ratio.split("-")
        self._O_Li_ratio_sys = float(n_O) / float(n_Li)

        if self.O_Li_ratio is not None and not np.isclose(
            self.O_Li_ratio, self._O_Li_ratio_sys, rtol=0
        ):
            raise ValueError(
                "`self.O_Li_ratio` ({}) != `self._O_Li_ratio_sys`"
                " ({})".format(self.O_Li_ratio, self._O_Li_ratio_sys)
            )
        if self._BulkSim is not None and not np.isclose(
            self._O_Li_ratio_sys, self._BulkSim._O_Li_ratio_sys, rtol=0
        ):
            raise ValueError(
                "`self._O_Li_ratio_sys` ({}) !="
                " `self._BulkSim._O_Li_ratio_sys` ({})".format(
                    self._O_Li_ratio_sys, self._BulkSim._O_Li_ratio_sys
                )
            )

        return self._O_Li_ratio_sys

    def get_Li_O_ratio(self):
        """
        Get the ratio of lithium ions to ether oxygen atoms.

        Return the value of :attr:`self.Li_O_ratio`.  If
        :attr:`self.Li_O_ratio` is ``None``, calculate the Li-EO ratio
        from :attr:`self.O_Li_ratio`.

        Returns
        -------
        self.Li_O_ratio : float
            Ratio of lithium ions to ether oxygen atoms.

        See Also
        --------
        :func:`self.get_O_Li_ratio` :
            Get the ratio of ether oxygen atoms to lithium ions
        """
        if self.Li_O_ratio is not None:
            return self.Li_O_ratio

        if self.O_Li_ratio is None:
            self.get_O_Li_ratio()

        self.Li_O_ratio = 1 / self.O_Li_ratio

        if self._BulkSim is not None and not np.isclose(
            self.Li_O_ratio, self._BulkSim.Li_O_ratio, rtol=0
        ):
            raise ValueError(
                "`self.Li_O_ratio` ({}) != `self._BulkSim.Li_O_ratio`"
                " ({})".format(self.Li_O_ratio, self._BulkSim.Li_O_ratio)
            )

        return self.Li_O_ratio

    def get_vol(self):
        """
        Get the total and accessible volume of the simulation box in
        Angstrom cubed.

        Return the value of :attr:`self.vol`.  If :attr:`self.vol` is
        ``None``, it is calculated from the box information contained in
        :attr:`self._Universe`.  It is assumed that the box is
        orthorhombic.

        The accessible volume is the total volume minus the volume of
        the electrodes (if present).

        Returns
        -------
        self.vol : dict
            Total and accessible volume of the simulation box in
            Angstrom cubed.
        """
        if self.vol is not None:
            return self.vol

        if self.system is None:
            self.get_system()
        if self.box is None:
            self.get_box()

        vol_tot = np.prod(self.box[:3])
        if vol_tot <= 0:
            raise ValueError("`vol_tot` ({}) <= 0".format(vol_tot))

        if self.is_bulk:
            vol_elctrd = 0.0
        else:
            elctrd_thk = 2 * self.Elctrd.ELCTRD_THK + self.Elctrd.GRA_SIGMA_LJ
            vol_elctrd = elctrd_thk * np.prod(self.box[:2])
        vol_access = vol_tot - vol_elctrd
        if vol_access <= 0:
            raise ValueError("`vol_access` ({}) <= 0".format(vol_access))

        self.vol = {
            "tot": vol_tot,
            "access": vol_access,
            "elctrd": vol_elctrd,
        }

        if self._BulkSim is not None and not np.isclose(
            self.vol["access"], self._BulkSim.vol["access"], rtol=0, atol=5
        ):
            raise ValueError(
                "`self.vol['access']` ({}) != `self._BulkSim.vol['access']`"
                " ({})".format(self.vol["access"], self._BulkSim.vol["access"])
            )

        return self.vol

    def get_dens(self):
        """
        Get the mass and number densities of all atom types and all
        electrolyte components.

        Return the value of :attr:`self.n_dens`.  If :attr:`self.n_dens`
        is ``None``, calculate the densities from :attr:`self._Universe`
        and :attr:`self.vol`.

        Mass densities are given in u/A^3, number densities are given in
        1/A^3.

        Returns
        -------
        self.n_dens : dict
            Mass and number densities of all atom types and all
            electrolyte components.

        Notes
        -----
        The densities are calculated with respect to the accessible
        volume of the simulation box, i.e. the volume of the electrodes
        is subtracted from the total box volume.  See
        :meth:`self.get_vol`.
        """
        if self.dens is not None:
            return self.dens

        if self.res_names is None:
            self.get_res_names()
        if self._Universe is None:
            self._get_Universe()
        if self.top_info is None:
            self.get_top_info()
        if self.vol is None:
            self.get_vol()

        vol = self.vol["access"]
        self.dens = {}

        # Densities of all atoms by their type.
        self.dens["atm_type"] = {}
        for at in self.top_info["sys"]["atm_type"].keys():
            ag = self._Universe.select_atoms("type {}".format(at))
            n_atm = ag.n_atoms
            m_atm = np.sum(ag.masses)
            atm_dct = {"num": n_atm / vol, "mass": m_atm / vol}
            self.dens["atm_type"][at] = atm_dct

        # # Commented out, because not all my simulations share the same
        # # atom names.
        # # Densities of all atoms by their name.
        # self.dens["atm_name"] = {}
        # for an in self.top_info["sys"]["atm_name"].keys():
        #     ag = self._Universe.select_atoms("name {}".format(an))
        #     n_atm = ag.n_atoms
        #     m_atm = np.sum(ag.masses)
        #     atm_dct = {"num": n_atm / vol, "mass": m_atm / vol}
        #     self.dens["atm_name"][an] = atm_dct

        # # Commented out, because not all my simulations share the same
        # # residue names.
        # # Densities of all residues.
        # self.dens["res"] = {}
        # for rn in self.top_info["res"].keys():
        #     ag = self._Universe.select_atoms("resname {}".format(rn))
        #     n_res = ag.n_residues
        #     m_res = np.sum(ag.residues.masses)
        #     res_dct = {"num": n_res / vol, "mass": m_res / vol}
        #     self.dens["res"][rn] = res_dct

        # Densities of the electrolyte: cation, anion, solvent and total
        self.dens["elctrlyt"] = {}
        n_elctrlyt, m_elctrlyt = 0, 0
        for res_type, rn in self.res_names.items():
            ag = self._Universe.select_atoms("resname {}".format(rn))
            n_res = ag.n_residues
            n_elctrlyt += n_res
            m_res = np.sum(ag.residues.masses)
            m_elctrlyt += m_res
            res_dct = {"num": n_res / vol, "mass": m_res / vol}
            self.dens["elctrlyt"][res_type] = res_dct
        self.dens["elctrlyt"]["tot"] = {
            "num": n_elctrlyt / vol,
            "mass": m_elctrlyt / vol,
        }

        if self._BulkSim is not None:
            for cmp, cmp_dens_dct in self.dens.items():
                for cn, dens_dct in cmp_dens_dct.items():
                    for dens_type, density in dens_dct.items():
                        if cn in self._BulkSim.dens[cmp] and not np.isclose(
                            density,
                            self._BulkSim.dens[cmp][cn][dens_type],
                            rtol=0,
                            atol=1e-5,
                        ):
                            raise ValueError(
                                "`self.dens` ({}) != `self._BulkSim.dens`"
                                " ({})".format(self.dens, self.dens)
                            )

        return self.dens

    def get_bulk_region(self):
        """
        Get the begin and end of the bulk region of the simulation box
        along the z direction in Angstrom.

        Return the value of :attr:`self.bulk_region`.  If
        :attr:`self.bulk_region` is ``None``, calculate the bulk region
        from :attr:`self.box`, :attr:`Electrode.ELCTRD_THK` and
        :attr:`Electrode.BULK_START`.

        For bulk simulations, the bulk region spans the entire
        simulation box.

        Returns
        -------
        self.bulk_region : numpy.ndarray of two floats
            The begin and end of the bulk region of the simulation box
            along the z direction in Angstrom.
        """
        if self.bulk_region is not None:
            return self.bulk_region

        if self.is_bulk is None:
            self.get_is_bulk()
        if self.box is None:
            self.get_box()

        bulk_begin = 0
        bulk_end = self.box[2]
        if not self.is_bulk:
            Elctrd = leap.simulation.Electrode()
            bulk_begin += Elctrd.ELCTRD_THK + Elctrd.BULK_START
            bulk_end -= Elctrd.ELCTRD_THK + Elctrd.BULK_START
        self.bulk_region = np.array([bulk_begin, bulk_end])
        return self.bulk_region


class Simulations:
    """
    Container class holding multiple
    :class:`~lintf2_ether_ana_postproc.simulation.Simulation` instances.

    All attributes should be treated as read-only attributes.  Don't
    change them after initialization.
    """

    # Class attributes (shared by all instances).

    NDENS2SI = 1e30 / const.Avogadro
    """
    Conversion factor to convert the number density from 1/A^3 to
    mol/m^3.

    :type: float
    """

    MDENS2SI = const.atomic_mass * 1e30
    """
    Conversion factor to convert the mass density from u/A^3 to kg/m^3.

    :type: float
    """

    def __init__(self, *paths, sort_key=None):
        """
        Initialize an instance of the
        :class:`~lintf2_ether_ana_postproc.simulation.Simulations`
        class.

        Parameters
        ----------
        paths : str or bytes or os.PathLike
            Relative or absolute paths to the directories containing the
            input and output files of the Gromacs MD simulations you
            want to use.  See
            :class:`lintf2_ether_ana_postproc.simulation.Simulation` for
            details.
        sort_key : callable, str or None, optional
            Sort key for sorting the simulations with the built-in
            function :func:`sorted`.  If `sort_key` is ``None``, the
            simulations in this container class are sorted in the order
            of the input paths.  If `sort_key` is a string, it must be
            a common attribute of all
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
            instances that can be used for sorting with :func:`sorted`.
            Duplicate simulations are always kept.
        """
        # Instance attributes.
        try:
            self.sims
            # Variable does already exist, because class instance is
            # re-initialized by `self.sort_sims` -> Don't set
            # `self.sims` to ``None`` to avoid unnecessary
            # re-initialization of all stored Simulation instances.
        except AttributeError:
            # Variable does not exist -> Class instance is initialized
            # for the first time.
            self.sims = None
            """
            List containing all stored
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
            instances.

            :type: list
            """

        self.n_sims = None
        """
        Number of simulations stored in this container class.

        :type: int
        """

        self.paths = None
        """
        List containing the path to each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        directory.

        :type: list
        """

        self.paths_ana = None
        """
        List containing the path to each corresponding analysis
        directory.

        :type: list
        """

        self.fnames_ana_base = None
        """
        List containing the base name of corresponding analysis files
        for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        :type: list
        """

        self.res_names = None
        """
        Dictionary containing the residue names of the cation, the anion
        and the solvent for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        :type: dict
        """

        self.res_nums = None
        """
        Dictionary containing the number of cation, anion and solvent
        residues for each :class:`Simulation`.

        :type: dict
        """

        self.surfqs = None
        """
        Array containing the surface charge for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in
        e/nm^2.

        :type: numpy.ndarray
        """

        self.temps = None
        """
        Array containing the temperature for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in K.

        :type: numpy.ndarray
        """

        self.O_per_chain = None
        """
        Array containing the number of ether oxygens per PEO chain for
        each :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        :type: numpy.ndarray
        """

        self.O_Li_ratios = None
        """
        Array containing the ratio of ether oxygen atoms to lithium ions
        for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        :type: numpy.ndarray
        """

        self.Li_O_ratios = None
        """
        Array containing the ratio of lithium ions to ether oxygen atoms
        for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        :type: numpy.ndarray
        """

        self.boxes = None
        """
        Array containing the simulation box dimensions for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in the
        same format as returned by
        :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
        ``[lx, ly, lz, alpha, beta, gamma]``.

        Lengths are given in nanometers, angles in degrees.

        :type: numpy.ndarray
        """

        self.boxes_z = None
        """
        Array containing the length of the simulation box along the z
        direction for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in
        nanometers.

        Note that :attr:`self.boxes_z` is just a view of
        :attr:`self.boxes`.

        :type: numpy.ndarray
        """

        self.vols = None
        """
        Simulation box volume for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in
        nanometers cubed.

        :type: dict
        """

        self.dens = None
        """
        Dictionary containing the mass and number density of all atom
        types and all electrolyte components for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        Mass densities are given in kg/m^3, number densities are given
        in 1/nm^3.

        :type: dict
        """

        self.bulk_regions = None
        """
        Array of shape ``(n_sims, 2)`` containing the begin and end of
        the bulk region of the simulation box along the z direction in
        nanometers for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        :type: numpy.ndarray
        """

        self.paths = []
        if len(paths) == 0:
            raise ValueError(
                "No paths provided (`paths` = '{}')".format(paths)
            )
        for path in paths:
            path = os.path.expandvars(os.path.expanduser(path))
            if not os.path.isdir(path):
                raise FileNotFoundError("No such directory: '{}'".format(path))
            self.paths.append(os.path.abspath(path))
        self.get_sims(sort_key)
        self.get_n_sims()
        self.get_paths_ana()
        self.get_fnames_ana_base()
        self.get_res_names()
        self.get_res_nums()
        self.get_surfqs()
        self.get_temps()
        self.get_O_per_chain()
        self.get_O_Li_ratios()
        self.get_Li_O_ratios()
        self.get_boxes()
        self.get_boxes_z()
        self.get_vols()
        self.get_dens()
        self.get_bulk_regions()

    def get_sims(self, sort_key=None):
        """
        Get the stored
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        instances.

        Return the value of :attr:`self.sims`.  If :attr:`self.sims` is
        ``None``, create
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        instances from :attr:`self.paths` and store them in
        :attr:`self.sims`.

        Parameters
        ----------
        sort_key : callable, str or None, optional
            Sort key for sorting the simulations with the built-in
            function :func:`sorted`.  If `sort_key` is ``None``, the
            simulations are sorted in the order of :attr:`self.paths`.
            If `sort_key` is a string, it must be a common attribute of
            all
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
            instances that can be used for sorting with :func:`sorted`.
            Duplicate simulations are always kept.

        Returns
        -------
        self.sims : list
            The list of stored (and sorted)
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
            instances.
        """
        if self.sims is not None:
            return self.sims

        if any(not os.path.isdir(path) for path in self.paths):
            raise FileNotFoundError(
                "At least one of the paths is not a directory:"
                " {}".format(self.paths)
            )

        self.sims = [leap.simulation.Simulation(path) for path in self.paths]
        if sort_key is not None:
            self.sort_sims(sort_key)
        return self.sims

    def sort_sims(self, sort_key):
        """
        Sort the simulations contained in this container class.

        This re-initialized the class instance to sort all instance
        attributes accordingly.

        Parameters
        ----------
        sort_key : callable or str, optional
            Sort key for sorting the simulations with the built-in
            function :func:`sorted`.  If `sort_key` is a string, it must
            be a common attribute of all
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
            instances that can be used for sorting with :func:`sorted`.
            Duplicate simulations are always kept.
        """
        if self.sims is None:
            self.get_sims()

        if self.sims is None:
            raise ValueError("No simulations to sort")
        if isinstance(sort_key, str):
            self.sims = sorted(
                self.sims, key=lambda sim: getattr(sim, sort_key)
            )
        else:
            self.sims = sorted(self.sims, key=sort_key)

        # Sort all other attributes accordingly by re-initializing the
        # class instance.
        paths = [sim.path for sim in self.sims]
        self.__init__(*paths)

        return self.sims

    def get_n_sims(self):
        """
        Get the number of simulations stored in this container class.

        Return the value of :attr:`self.n_sims`.  If :attr:`self.n_sims`
        is ``None``, infer the number of simulations from
        :attr:`self.sims`.

        Returns
        -------
        self.n_sims : int
            The number of simulations stored in this container class.
        """
        if self.n_sims is not None:
            return self.n_sims

        if self.sims is None:
            self.get_sims()

        self.n_sims = len(self.sims)
        return self.n_sims

    def get_paths(self):
        """
        Get the path to each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        directory.

        Return the value of :attr:`self.paths`.  If :attr:`self.paths`
        is ``None``, create from :attr:`self.sims`.

        Returns
        -------
        self.paths : list
            List containing the path to each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
            directory.
        """
        if self.paths is not None:
            return self.paths

        if self.sims is None:
            self.get_sims()

        self.paths = [sim.path for sim in self.sims]
        return self.paths

    def get_paths_ana(self):
        """
        Get the path to each corresponding analysis directory.

        Return the value of :attr:`self.paths_ana`.  If
        :attr:`self.paths_ana` is ``None``, create from
        :attr:`self.sims`.

        Returns
        -------
        self.paths_ana : list
            List containing the path to each corresponding analysis
            directory.
        """
        if self.paths_ana is not None:
            return self.paths_ana

        if self.sims is None:
            self.get_sims()

        self.paths_ana = [sim.path_ana for sim in self.sims]
        return self.paths_ana

    def get_fnames_ana_base(self):
        """
        Get the base name of corresponding analysis files for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        Return the value of :attr:`self.fnames_ana_base`.  If
        :attr:`self.fnames_ana_base` is ``None``, create it from
        :attr:`self.sims`.

        Returns
        -------
        self.fnames_ana_base : list
            List containing the base name of corresponding analysis
            files for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.
        """
        if self.fnames_ana_base is not None:
            return self.fnames_ana_base

        if self.sims is None:
            self.get_sims()

        self.fnames_ana_base = [sim.fname_ana_base for sim in self.sims]
        return self.fnames_ana_base

    def get_res_names(self):
        """
        Get the residue names of the cation, the anion and the solvent
        for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        Return the value of :attr:`self.res_names`.  If
        :attr:`self.res_names` is ``None``, create it from
        :attr:`self.sims`.

        Returns
        -------
        self.res_names : dict
            Dictionary containing the residue names of the cation, the
            anion and the solvent for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.
        """
        if self.res_names is not None:
            return self.res_names

        if self.sims is None:
            self.get_sims()

        self.res_names = {
            res_type: [] for res_type in self.sims[0].res_names.keys()
        }
        for sim in self.sims:
            for res_type, rn in sim.res_names.items():
                self.res_names[res_type].append(rn)
        return self.res_names

    def get_res_nums(self):
        """
        Get the number of cation, anion and solvent residues for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        Return the value of :attr:`self.res_nums`.  If
        :attr:`self.res_nums` is ``None``, create it from
        :attr:`self.sims`.

        Returns
        -------
        self.res_nums : dict
            Dictionary containing the number of cation, anion and
            solvent residues for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.
        """
        if self.res_nums is not None:
            return self.res_nums

        if self.sims is None:
            self.get_sims()

        res_nums = {res_type: [] for res_type in self.sims[0].res_names.keys()}
        for sim in self.sims:
            for res_type, rn in sim.res_names.items():
                res_nums[res_type].append(sim.top_info["res"][rn]["n_res"])
        self.res_nums = {}
        for res_type, res_num in res_nums.items():
            self.res_nums[res_type] = np.array(res_num)
        return self.res_nums

    def get_surfqs(self):
        """
        Get the surfaces charge for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in
        e/nm^2.

        Return the value of :attr:`self.surfqs`.  If :attr:`self.surfqs`
        is ``None``, create it from :attr:`self.sims`.

        Returns
        -------
        self.surfqs : numpy.ndarray
            Array containing the surface charge for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in
            e/nm^2.
        """
        if self.surfqs is not None:
            return self.surfqs

        if self.sims is None:
            self.get_sims()

        self.surfqs = np.array(
            [
                sim.surfq if sim.surfq is not None else np.nan
                for sim in self.sims
            ]
        )
        self.surfqs *= 100  # e/Angstrom^2 -> e/nm^2
        return self.surfqs

    def get_temps(self):
        """
        Get the temperature for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in K.

        Return the value of :attr:`self.temps`.  If :attr:`self.temps`
        is ``None``, create it from :attr:`self.sims`.

        Returns
        -------
        self.temps : numpy.ndarray
            Array containing the temperature for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in
            K.
        """
        if self.temps is not None:
            return self.temps

        if self.sims is None:
            self.get_sims()

        self.temps = np.array([sim.temp for sim in self.sims])
        return self.temps

    def get_O_per_chain(self):
        """
        Get the number of ether oxygens per PEO chain for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        Return the value of :attr:`self.O_per_chain`.  If
        :attr:`self.O_per_chain` is ``None``, create it from
        :attr:`self.sims`.

        Returns
        -------
        self.O_per_chain : numpy.ndarray
            Array containing the number of ether oxygens per PEO chain
            for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.
        """
        if self.O_per_chain is not None:
            return self.O_per_chain

        if self.sims is None:
            self.get_sims()

        self.O_per_chain = np.array([sim.O_per_chain for sim in self.sims])
        return self.O_per_chain

    def get_O_Li_ratios(self):
        """
        Get the ratio of ether oxygen atoms to lithium ions for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        Return the value of :attr:`self.O_Li_ratios`.  If
        :attr:`self.O_Li_ratios` is ``None``, create it from
        :attr:`self.sims`.

        Returns
        -------
        self.O_Li_ratios : numpy.ndarray
            Array containing the ratio of ether oxygen atoms to lithium
            ions for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.
        """
        if self.O_Li_ratios is not None:
            return self.O_Li_ratios

        if self.sims is None:
            self.get_sims()

        self.O_Li_ratios = np.array([sim.O_Li_ratio for sim in self.sims])
        return self.O_Li_ratios

    def get_Li_O_ratios(self):
        """
        Get the ratio of lithium ions to ether oxygen atoms for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        Return the value of :attr:`self.Li_O_ratios`.  If
        :attr:`self.Li_O_ratios` is ``None``, create it from
        :attr:`self.sims`.

        Returns
        -------
        self.Li_O_ratios : numpy.ndarray
            Array containing the ratio of lithium ions to ether oxygen
            atoms for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.
        """
        if self.Li_O_ratios is not None:
            return self.Li_O_ratios

        if self.sims is None:
            self.get_sims()

        self.Li_O_ratios = np.array([sim.Li_O_ratio for sim in self.sims])
        return self.Li_O_ratios

    def get_boxes(self):
        """
        Get the simulation box dimensions for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        Return the value of :attr:`self.boxes`.  If :attr:`self.boxes`
        is ``None``, create it from :attr:`self.sims`.

        Lengths are given in nanometers, angles in degrees.

        Returns
        -------
        self.boxes : numpy.ndarray
            Array containing the simulation box dimensions for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in
            the same format as returned by
            :attr:`MDAnalysis.coordinates.base.Timestep.dimensions`:
            ``[lx, ly, lz, alpha, beta, gamma]``.

        See Also
        --------
        :meth:`self.get_boxes_z` :
            Get the length of the simulation box along the z direction
            for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in
            nanometers.
        """
        if self.boxes is not None:
            return self.boxes

        if self.sims is None:
            self.get_sims()

        self.boxes = np.array([sim.box for sim in self.sims])
        self.boxes[:, :3] /= 10  # Angstrom -> nm
        return self.boxes

    def get_boxes_z(self):
        """
        Get the length of the simulation box along the z direction for
        each :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        in nanometers.

        Return the value of :attr:`self.boxes_z`.  If
        :attr:`self.boxes_z` is ``None``, create it from
        :attr:`self.boxes`.

        Returns
        -------
        self.boxes_z : numpy.ndarray
            Array containing the length of the simulation box along the
            z direction for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in
            nanometers.

        See Also
        --------
        :meth:`self.get_boxes` :
            Get the simulation box dimensions for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`

        Notes
        -----
        :attr:`self.boxes_z` is just a view of :attr:`self.boxes`.
        """
        if self.boxes_z is not None:
            return self.boxes_z

        if self.boxes is None:
            self.boxes()

        self.boxes_z = self.boxes[:, 2]
        return self.boxes_z

    def get_vols(self):
        """
        Get the simulation box volumes for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in
        nanometers cubed.

        Return the value of :attr:`self.vols`.  If :attr:`self.vols` is
        ``None``, create it from :attr:`self.sims`.

        Returns
        -------
        self.vols : dict
            Simulation box volume for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in
            nanometers cubed.
        """
        if self.vols is not None:
            return self.vols

        if self.sims is None:
            self.get_sims()

        vols = {vol_type: [] for vol_type in self.sims[0].vol.keys()}
        for sim in self.sims:
            for vol_type, vl in sim.vol.items():
                vols[vol_type].append(vl)
        self.vols = {}
        for vol_type, vls in vols.items():
            self.vols[vol_type] = np.array(vls) * 1e-3  # A^3 -> nm^3
        return self.vols

    def get_dens(self):
        """
        Get the mass and number densities of all atom types and all
        electrolyte components for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        Return the value of :attr:`self.dens`.  If :attr:`self.dens` is
        ``None``, create it from :attr:`self.sims`.

        Mass densities are given in kg/m^3, number densities are given
        in 1/nm^3.

        Returns
        -------
        self.dens : dict
            Dictionary containing the mass and number density of all
            atom types and all electrolyte components for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.
        """
        if self.dens is not None:
            return self.dens

        if self.sims is None:
            self.get_sims()

        dens = {
            cmp: {
                cn: {dens_type: [] for dens_type in dens_dct.keys()}
                for cn, dens_dct in cmp_dens_dct.items()
            }
            for cmp, cmp_dens_dct in self.sims[0].dens.items()
        }

        for sim in self.sims:
            for cmp, cmp_dens_dct in sim.dens.items():
                for cn, dens_dct in cmp_dens_dct.items():
                    for dens_type, density in dens_dct.items():
                        dens[cmp][cn][dens_type].append(density)
        self.dens = {
            cmp: {cn: {} for cn in cmp_dens_dct.keys()}
            for cmp, cmp_dens_dct in dens.items()
        }
        for cmp, cmp_dens_dct in dens.items():
            for cn, dens_dct in cmp_dens_dct.items():
                for dens_type, density in dens_dct.items():
                    dens_array = np.array(density)
                    if dens_type == "num":
                        dens_array *= 1e3  # 1/A^3 -> 1/nm^3
                    elif dens_type == "mass":
                        dens_array *= self.MDENS2SI  # u/A^3 -> kg/m^3
                    self.dens[cmp][cn][dens_type] = dens_array
        return self.dens

    def get_bulk_regions(self):
        """
        Get the begin and end of the bulk region of the simulation box
        along the z direction in nanometers for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        Return the value of :attr:`self.bulk_regions`.  If
        :attr:`self.bulk_regions` is ``None``, create it from
        :attr:`self.sims`.

        Returns
        -------
        self.bulk_regions : numpy.ndarray
            Array of shape ``(n_sims, 2)`` containing the begin and end
            of the bulk region of the simulation box along the z
            direction in nanometers for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.
        """
        if self.bulk_regions is not None:
            return self.bulk_regions

        if self.sims is None:
            self.get_sims()

        self.bulk_regions = np.array([sim.bulk_region for sim in self.sims])
        self.bulk_regions /= 10  # Angstrom -> nm
        return self.bulk_regions


def get_surfq(system):
    """
    Extract the surface charge of the electrodes in a surface simulation
    from the system name.

    Parameters
    ----------
    system : str
        Name of the simulated system.  The surface charge of the
        electrodes must be contained in the system name as "*_qX_*",
        where X can be any integer or floating point number.

    Returns
    -------
    surfq : float
        The surface charge of the electrodes inferred from the system
        name.

    Raises
    ------
    ValueError :
        If `system` does not contain the pattern "*_qX_*".
    """
    for s in system.split("_"):
        if s.startswith("q"):
            surfq = s[1:]
            break
    if surfq is None:
        raise ValueError(
            "Could not infer the surface charge from `system`"
            " ({})".format(system)
        )
    return float(surfq)


def num_res_per_name(ag):
    """
    Get the number of residues of each residue name in an MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        Input :class:`~MDAnalysis.core.groups.AtomGroup`.

    Returns
    -------
    n_res_per_name : dict
        Dictionary with the residue names as keys and the corresponding
        number of residues as values.
    """
    unique_resnames = np.unique(ag.resnames)
    n_res_per_name = [
        ag.select_atoms("resname " + rn).n_residues for rn in unique_resnames
    ]
    return dict(zip(unique_resnames, n_res_per_name))


def num_atoms_per_type(ag, attr="types"):
    """
    Get the number of atoms of each atom type/name in an MDAnalysis
    :class:`~MDAnalysis.core.groups.AtomGroup`.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        Input :class:`~MDAnalysis.core.groups.AtomGroup`.
    attr : {"types", "names"}, optional
        Whether to return the number of atoms of each atom type or of
        each atom name.

    Returns
    -------
    n_atoms_per_type : dict
        Dictionary with the atom types/names as keys and the
        corresponding number of atoms as values.
    """
    unique_atom_types = np.unique(getattr(ag, attr))
    n_atoms_per_type = [
        ag.select_atoms(attr[:-1] + " " + at).n_atoms
        for at in unique_atom_types
    ]
    return dict(zip(unique_atom_types, n_atoms_per_type))


def get_sim(sys_pat, set_pat, path_key, exclude_pat=None):
    """
    Create a :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
    instance from glob patterns.

    Parameters
    ----------
    sys_pat : str
        System name pattern.
    set_pat : str
        Simulations settings name pattern.
    path_key : str
        Path key to fetch the top-level path that contains the desired
        simulation from
        :attr:`~lintf2_ether_ana_postproc.simulation.SimPaths.PATHS`.
        See there for possible keys.
    exclude_pat : str or None, optional
        System name pattern to exclude.  Any simulation whose system
        system name matches this glob pattern will be excluded from the
        created
        :class:`~lintf2_ether_ana_postproc.simulation.Simulations`
        instance.

    Returns
    -------
    Sim : lintf2_ether_ana_postproc.simulation.Simulation
        A :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        instance.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.simulation.get_sims` :
        Create a
        :class:`~lintf2_ether_ana_postproc.simulation.Simulations`
        instance from glob patterns.
    """
    SimPaths = leap.simulation.SimPaths()
    pattern = os.path.join(SimPaths.PATHS[path_key], sys_pat, set_pat)
    paths = glob.glob(pattern)

    if exclude_pat is not None:
        pattern_exclude = os.path.join(
            SimPaths.PATHS[path_key], exclude_pat, set_pat
        )
        paths_exclude = glob.glob(pattern_exclude)
        paths = list(set(paths) - set(paths_exclude))
        err_msg_suffix = " and excluding the pattern '{}'".format(
            pattern_exclude
        )
    else:
        err_msg_suffix = ""

    if len(paths) == 0:
        err_msg = (
            "Could not find any file/directory matching the pattern"
            " '{}'".format(pattern)
        )
        raise ValueError(err_msg + err_msg_suffix)
    elif len(paths) > 1:
        err_msg = (
            "Found more than one file/directory matching the pattern"
            " '{}'".format(pattern)
        )
        raise ValueError(err_msg + err_msg_suffix)
    return leap.simulation.Simulation(paths[0])


def get_sims(sys_pat, set_pat, path_key, exclude_pat=None, **kwargs):
    """
    Create a :class:`~lintf2_ether_ana_postproc.simulation.Simulations`
    instance from glob patterns.

    Parameters
    ----------
    sys_pat : str
        System name pattern.
    set_pat : str
        Simulations settings name pattern.
    path_key : str
        Path key to fetch the top-level path that contains the desired
        simulations from
        :attr:`~lintf2_ether_ana_postproc.simulation.SimPaths.PATHS`.
        See there for possible keys.
    exclude_pat : str or None, optional
        System name pattern to exclude.  Any simulation whose system
        system name matches this glob pattern will be excluded from the
        created
        :class:`~lintf2_ether_ana_postproc.simulation.Simulations`
        instance.
    kwargs : dict, optional
        Additional keyword arguments (besides `paths`) to parse to the
        constructor of
        :class:`~lintf2_ether_ana_postproc.simulation.Simulations`

    Returns
    -------
    Sims : lintf2_ether_ana_postproc.simulation.Simulations
        A :class:`~lintf2_ether_ana_postproc.simulation.Simulations`
        instance.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.simulation.get_sim` :
        Create a
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        instance from glob patterns.
    """
    SimPaths = leap.simulation.SimPaths()
    pattern = os.path.join(SimPaths.PATHS[path_key], sys_pat, set_pat)
    paths = glob.glob(pattern)

    if exclude_pat is not None:
        pattern_exclude = os.path.join(
            SimPaths.PATHS[path_key], exclude_pat, set_pat
        )
        paths_exclude = glob.glob(pattern_exclude)
        paths = list(set(paths) - set(paths_exclude))
        err_msg_suffix = " and excluding the pattern '{}'".format(
            pattern_exclude
        )
    else:
        err_msg_suffix = ""

    if len(paths) == 0:
        err_msg = (
            "Could not find any file/directory matching the pattern"
            " '{}'".format(pattern)
        )
        raise ValueError(err_msg + err_msg_suffix)
    return leap.simulation.Simulations(*paths, **kwargs)


def get_ana_file(Sim, ana_name, ana_tool, file_suffix):
    """
    Get the path to the given analysis file for a given
    :class:`~lintf2_ether_ana_postproc.simulation.Simulation` instance.

    Parameters
    ----------
    Sim : lintf2_ether_ana_postproc.simulation.Simulation
        The :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        instance.
    ana_name : str
        The analysis name.
    ana_tool : {"gmx", "mdt"}
        The software/tool used to generate the analysis file.
    file_suffix : str
        The suffix of the analysis file without the simulation settings
        and the system name, i.e. everything after
        :attr:`lintf2_ether_ana_postproc.simulation.Simulation.fname_ana_base`.

    Returns
    -------
    ana_file : str
        Path to the analysis files.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.get_ana_files` :
        Get the paths to the given analysis files for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in a
        given :class:`~lintf2_ether_ana_postproc.simulation.Simulations`
        instance.
    """
    if ana_tool not in ("gmx", "mdt"):
        raise ValueError("Unknown `ana_tool`: '{}'".format(ana_tool))

    fname = Sim.fname_ana_base + file_suffix
    ana_file = os.path.join(Sim.path_ana, ana_tool, ana_name, fname)
    if not os.path.isfile(ana_file):
        raise FileNotFoundError("No such file: '{}'".format(ana_file))
    return ana_file


def get_ana_files(Sims, ana_name, ana_tool, file_suffix):
    """
    Get the paths to the given analysis files for each
    :class:`~lintf2_ether_ana_postproc.simulation.Simulation` in a given
    :class:`~lintf2_ether_ana_postproc.simulation.Simulations` instance.

    Parameters
    ----------
    Sims : lintf2_ether_ana_postproc.simulation.Simulations
        The :class:`~lintf2_ether_ana_postproc.simulation.Simulations`
        instance.
    ana_name : str
        The analysis name.
    ana_tool : {"gmx", "mdt"}
        The software/tool used to generate the analysis file.
    file_suffix : str
        The suffix of the analysis file without the simulation settings
        and the system name, i.e. everything after
        :attr:`lintf2_ether_ana_postproc.simulation.Simulations.fnames_ana_base`.

    Returns
    -------
    ana_files : list
        List of paths to the analysis files.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.get_ana_file` :
        Get the path to the given analysis file for a given
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        instance.
    """
    if ana_tool not in ("gmx", "mdt"):
        raise ValueError("Unknown `ana_tool`: '{}'".format(ana_tool))

    ana_files = []
    for sim_ix, path in enumerate(Sims.paths_ana):
        fname = Sims.fnames_ana_base[sim_ix] + file_suffix
        fpath = os.path.join(path, ana_tool, ana_name, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError("No such file: '{}'".format(fpath))
        ana_files.append(fpath)
    return ana_files


def read_free_energy_extrema_single(Sim, cmp, peak_type, cols, prom_min=None):
    """
    Read the free-energy extrema from file for a single simulation.

    Parameters
    ----------
    Sim : lintf2_ether_ana_postproc.simulation.Simulation
        The :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        for which to read the free-energy extrema from file.
    cmp : {"Li", "NBT", "OBT", "OE"}
        The compound to consider.
    peak_type : {"minima", "maxima"}
        The peak/extremum type to consider.
    cols : array_like
        The columns to read from the output files of
        :file:`scripts/gmx/density-z/get_free-energy_extrema.py`.
        `cols` must contain the column that holds the peak positions in
        nm.
    prom_min : float or None, optional
        If provided, only return peaks with a prominence of at least
        `prom_min`.

    Returns
    -------
    data : list
        2-dimensional list of read data.  The first index addresses the
        read columns.  The second index addresses the peak-position
        type, i.e. whether the peak is in the left or right half of the
        simulation box.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.simulation.read_free_energy_extrema` :
        Read free-energy extrema from file for multiple simulations
    """  # noqa: W505
    if cmp not in ("Li", "NBT", "OBT", "OE"):
        raise ValueError("Unknown `cmp`: {}".format(cmp))
    peak_type = peak_type.lower()
    if peak_type not in ("minima", "maxima"):
        raise ValueError("Unknown `peak_type`: {}".format(peak_type))
    cols = np.asarray(cols)
    if cols.ndim != 1:
        raise ValueError(
            "`cols` has {} dimension(s) but must have 1"
            " dimension".format(cols.ndim)
        )

    # Assemble input file names
    file_suffix = "free_energy_" + peak_type + "_" + cmp + ".txt.gz"
    analysis = "density-z"  # Analysis name.
    tool = "gmx"  # Analysis software.
    infile = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)

    # Column in the output file of
    # "scripts/gmx/density-z/get_free-energy_extrema.py" that contains
    # the peak positions in nm.  Column numbering starts at zero.
    pkp_col = 1
    pkp_col_ix = np.where(cols == pkp_col)[0]
    if pkp_col_ix.shape != (1,):
        raise ValueError(
            "`cols` ({}) must contain the column index {} exactly"
            " ones".format(cols, pkp_col)
        )
    pkp_col_ix = pkp_col_ix[0]

    if prom_min is not None:
        prom_col = 3  # Column containing the peak prominences.
        if prom_col not in cols:
            cols = np.append(cols, prom_col)
        prom_col_ix = np.flatnonzero(cols == prom_col)[0]

    Elctrd = leap.simulation.Electrode()
    elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm
    bulk_start = Elctrd.BULK_START / 10  # A -> nm
    box_z = Sim.box[2] / 10  # A -> nm
    bulk_region = Sim.bulk_region / 10  # A -> nm

    # Due to `unpack=True`, columns in the input file become rows in the
    # created array and rows become columns.
    data_raw = np.loadtxt(infile, usecols=cols, unpack=True, ndmin=2)
    if prom_min is not None:
        valid_pks = data_raw[prom_col_ix] >= prom_min
        data_raw = data_raw[:, valid_pks]
        if not np.any(valid_pks):
            raise ValueError(
                "No peaks with a prominence of at least {} contained in the"
                " input file".format(prom_min)
            )
    pk_pos = data_raw[pkp_col_ix]
    pk_is_left = pk_pos <= (box_z / 2)

    if np.any(pk_pos <= elctrd_thk):
        raise ValueError(
            "At least one peak lies within the left electrode.  Peak"
            " positions: {}.  Left electrode:"
            " {}".format(pk_pos, elctrd_thk)
        )
    if np.any(pk_pos >= box_z - elctrd_thk):
        raise ValueError(
            "At least one peak lies within the right electrode.  Peak"
            " positions: {}.  Right electrode:"
            " {}".format(pk_pos, box_z - elctrd_thk)
        )
    if np.any((pk_pos >= bulk_region[0]) & (pk_pos <= bulk_region[1])):
        raise ValueError(
            "At least one peak lies within the bulk region.  Peak"
            " positions: {}.  Bulk region:"
            " {}".format(pk_pos, bulk_region)
        )
    if Sim.surfq == 0:
        n_pks_left = np.count_nonzero(pk_is_left)
        n_pks_right = len(pk_is_left) - n_pks_left
        if n_pks_left != n_pks_right:
            raise ValueError(
                "The surface charge is {} e/nm^2 but the number of left"
                " ({}) and right free-energy {} ({}) do not"
                " match.".format(Sim.surfq, n_pks_left, peak_type, n_pks_right)
            )

    pk_pos_types = ("left", "right")
    data = [[None for pkp_type in pk_pos_types] for col in cols]
    for pkt_ix, pkp_type in enumerate(pk_pos_types):
        if pkp_type == "left":
            valid_pks = pk_is_left
            data_raw_valid = data_raw[:, valid_pks]
            pk_pos = data_raw_valid[pkp_col_ix]
            # Convert absolute peak positions to distances to the
            # electrodes.
            pk_pos -= elctrd_thk
        elif pkp_type == "right":
            valid_pks = ~pk_is_left
            data_raw_valid = data_raw[:, valid_pks]
            # Reverse the order of rows to sort peaks as function of
            # the distance to the electrodes in ascending order.
            data_raw_valid = data_raw_valid[:, ::-1]
            pk_pos = data_raw_valid[pkp_col_ix]
            # Convert absolute peak positions to distances to the
            # electrodes.
            pk_pos += elctrd_thk
            pk_pos -= box_z
            pk_pos *= -1  # Ensure positive distance values.
        else:
            raise ValueError(
                "Unknown peak position type: '{}'".format(pkp_type)
            )
        if np.any(pk_pos <= 0):
            raise ValueError(
                "Peak position type: '{}'.\n"
                "At least one peak lies within the electrode.  This should"
                " not have happened.  Peak positions: {}.  Electrode:"
                " 0".format(pkp_type, pk_pos)
            )
        if np.any(pk_pos >= bulk_start):
            raise ValueError(
                "Peak position type: '{}'.\n"
                "At least one peak lies within the bulk region or near the"
                " opposite electrode.  Peak positions: {}.  Bulk start:"
                " {}.  Opposite electrode: {}".format(
                    pkp_type, pk_pos, bulk_start, box_z - 2 * elctrd_thk
                )
            )
        data_raw_valid[pkp_col_ix] = pk_pos

        for col_ix, dat_col_raw_valid in enumerate(data_raw_valid):
            data[col_ix][pkt_ix] = dat_col_raw_valid

    return data


def read_free_energy_extrema(Sims, cmp, peak_type, cols, prom_min=None):
    """
    Read free-energy extrema from file for multiple simulations.

    Read the output file of
    :file:`scripts/gmx/density-z/get_free-energy_extrema.py` for
    multiple simulations.

    Parameters
    ----------
    Sims : lintf2_ether_ana_postproc.simulation.Simulations
        A :class:`~lintf2_ether_ana_postproc.simulation.Simulations`
        instance holding the simulations for which to read the
        free-energy extrema from file.
    cmp : {"Li", "NBT", "OBT", "OE"}
        The compound to consider.
    peak_type : {"minima", "maxima"}
        The peak/extremum type to consider.
    cols : array_like
        The columns to read from the output files of
        :file:`scripts/gmx/density-z/get_free-energy_extrema.py`.
        `cols` must contain the column that holds the peak positions in
        nm.
    prom_min : float or None, optional
        If provided, only return peaks with a prominence of at least
        `prom_min`.

    Returns
    -------
    data : list
        3-dimensional list of read data.  The first index addresses the
        read columns.  The second index addresses the peak-position
        type, i.e. whether the peak is in the left or right half of the
        simulation box.  The third index addresses the simulation.  This
        structure of `data` is useful for plotting the values in
        specific columns as function of the simulation.
    n_pks_max : numpy.ndarray
        Maximum number of peaks in a single simulation for each
        peak-position type.

    See Also
    --------
    :func:`lintf2_ether_ana_postproc.simulation.read_free_energy_extrema_single` :
        Read the free-energy extrema from file for a single simulation
    """  # noqa: E501, W505
    if cmp not in ("Li", "NBT", "OBT", "OE"):
        raise ValueError("Unknown `cmp`: {}".format(cmp))
    peak_type = peak_type.lower()
    if peak_type not in ("minima", "maxima"):
        raise ValueError("Unknown `peak_type`: {}".format(peak_type))
    cols = np.asarray(cols)
    if cols.ndim != 1:
        raise ValueError(
            "`cols` has {} dimension(s) but must have 1"
            " dimension".format(cols.ndim)
        )

    # Assemble input file names
    file_suffix = "free_energy_" + peak_type + "_" + cmp + ".txt.gz"
    analysis = "density-z"  # Analysis name.
    tool = "gmx"  # Analysis software.
    infiles = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)

    # Column in the output file of
    # "scripts/gmx/density-z/get_free-energy_extrema.py" that contains
    # the peak positions in nm.  Column numbering starts at zero.
    pkp_col = 1
    pkp_col_ix = np.where(cols == pkp_col)[0]
    if pkp_col_ix.shape != (1,):
        raise ValueError(
            "`cols` ({}) must contain the column index {} exactly"
            " ones".format(cols, pkp_col)
        )
    pkp_col_ix = pkp_col_ix[0]

    if prom_min is not None:
        prom_col = 3  # Column containing the peak prominences.
        if prom_col not in cols:
            cols = np.append(cols, prom_col)
        prom_col_ix = np.flatnonzero(cols == prom_col)[0]

    Elctrd = leap.simulation.Electrode()
    elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm
    bulk_start = Elctrd.BULK_START / 10  # A -> nm

    pk_pos_types = ("left", "right")
    n_pks_max = np.zeros_like(pk_pos_types, dtype=int)
    data = [
        [[None for sim in Sims.sims] for pkp_type in pk_pos_types]
        for col in cols
    ]
    for sim_ix, Sim in enumerate(Sims.sims):
        box_z = Sim.box[2] / 10  # A -> nm
        bulk_region = Sim.bulk_region / 10  # A -> nm

        # Due to `unpack=True`, columns in the input file become rows in
        # the created array and rows become columns.
        data_sim = np.loadtxt(
            infiles[sim_ix], usecols=cols, unpack=True, ndmin=2
        )
        if prom_min is not None:
            valid_pks = data_sim[prom_col_ix] >= prom_min
            data_sim = data_sim[:, valid_pks]
            if not np.any(valid_pks):
                raise ValueError(
                    "Simulation: '{}'.\n"
                    "No peaks with a prominence of at least {} contained in"
                    " the input file".format(Sim.path, prom_min)
                )
        pk_pos = data_sim[pkp_col_ix]
        pk_is_left = pk_pos <= (box_z / 2)

        if np.any(pk_pos <= elctrd_thk):
            raise ValueError(
                "Simulation: '{}'.\n"
                "At least one peak lies within the left electrode.  Peak"
                " positions: {}.  Left electrode: {}".format(
                    Sim.path, pk_pos, elctrd_thk
                )
            )
        if np.any(pk_pos >= box_z - elctrd_thk):
            raise ValueError(
                "Simulation: '{}'.\n"
                "At least one peak lies within the right electrode.  Peak"
                " positions: {}.  Right electrode:"
                " {}".format(Sim.path, pk_pos, box_z - elctrd_thk)
            )
        if np.any((pk_pos >= bulk_region[0]) & (pk_pos <= bulk_region[1])):
            raise ValueError(
                "Simulation: '{}'.\n"
                "At least one peak lies within the bulk region.  Peak"
                " positions: {}.  Bulk region: {}".format(
                    Sim.path, pk_pos, bulk_region
                )
            )
        if Sim.surfq == 0:
            n_pks_left = np.count_nonzero(pk_is_left)
            n_pks_right = len(pk_is_left) - n_pks_left
            if n_pks_left != n_pks_right:
                raise ValueError(
                    "Simulation: '{}'.\n"
                    "The surface charge is {} e/nm^2 but the number of left"
                    " ({}) and right free-energy {} ({}) do not"
                    " match.".format(
                        Sim.path, Sim.surfq, n_pks_left, peak_type, n_pks_right
                    )
                )

        for pkt_ix, pkp_type in enumerate(pk_pos_types):
            if pkp_type == "left":
                valid_pks = pk_is_left
                data_sim_valid = data_sim[:, valid_pks]
                pk_pos = data_sim_valid[pkp_col_ix]
                # Convert absolute peak positions to distances to the
                # electrodes.
                pk_pos -= elctrd_thk
            elif pkp_type == "right":
                valid_pks = ~pk_is_left
                data_sim_valid = data_sim[:, valid_pks]
                # Reverse the order of rows to sort peaks as function of
                # the distance to the electrodes in ascending order.
                data_sim_valid = data_sim_valid[:, ::-1]
                pk_pos = data_sim_valid[pkp_col_ix]
                # Convert absolute peak positions to distances to the
                # electrodes.
                pk_pos += elctrd_thk
                pk_pos -= box_z
                pk_pos *= -1  # Ensure positive distance values.
            else:
                raise ValueError(
                    "Unknown peak position type: '{}'".format(pkp_type)
                )
            if np.any(pk_pos <= 0):
                raise ValueError(
                    "Simulation: '{}'.\n"
                    "Peak-position type: '{}'.\n"
                    "At least one peak lies within the electrode.  This should"
                    " not have happened.  Peak positions: {}.  Electrode:"
                    " 0".format(Sim.path, pkp_type, pk_pos)
                )
            if np.any(pk_pos >= bulk_start):
                raise ValueError(
                    "Simulation: '{}'.\n"
                    "Peak-position type: '{}'.\n"
                    "At least one peak lies within the bulk region or near the"
                    " opposite electrode.  Peak positions: {}.  Bulk start:"
                    " {}.  Opposite electrode: {}".format(
                        Sim.path,
                        pkp_type,
                        pk_pos,
                        bulk_start,
                        box_z - 2 * elctrd_thk,
                    )
                )
            data_sim_valid[pkp_col_ix] = pk_pos
            n_pks_max[pkt_ix] = max(n_pks_max[pkt_ix], len(pk_pos))

            for col_ix, dat_col_sim_valid in enumerate(data_sim_valid):
                data[col_ix][pkt_ix][sim_ix] = dat_col_sim_valid

    return data, n_pks_max


def read_displvar_single(
    Sim, cmp, dim, time_conv=1e-3, length_conv=0.1, displvar=True
):
    """
    Read the displacement variance per bin from file for a single
    simulation.

    Parameters
    ----------
    Sim : lintf2_ether_ana_postproc.simulation.Simulation
        The :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        for which to read the displacement variance from file.
    cmp : {"Li", "NTf2", "ether", "NBT", "OBT", "OE"}
        The compound to consider.
    dim : {"x", "y", "z"}
        The dimension of the displacement to read from file.
    time_conv : scalar, optional
        Factor for converting the time scale.  The time scale of the
        input files is usually ps.
    length_conv : scalar, optional
        Factor for converting the length scale.  The length scale of the
        input files is usually Angstrom.
    displvar : bool, optional
        If ``True``, subtract the squared mean displacement from the
        mean squared displacement to calculate the displacement
        variance.

    Returns
    -------
    times : numpy.ndarray
        1-dimensional array containing the lag times.
    bins : numpy.ndarray
        1-dimensional array containing the bin edges.
    md_data : numpy.ndarray
        2-dimensional array containing the mean displacements for each
        lag time and bin.
    msd_data : numpy.ndarray
        2-dimensional array containing the mean squared displacements
        (if `displvar` is ``False``) or the displacement variance (if
        `displvar` is ``True``) for lag time and bin.
    """
    if cmp not in ("Li", "NTf2", "ether", "NBT", "OBT", "OE"):
        raise ValueError("Unknown `cmp`: {}".format(cmp))
    if dim not in ("x", "y", "z"):
        raise ValueError("Unknown `cmp`: {}".format(dim))

    analysis = "msd_layer"  # Analysis name.
    analysis_suffix = "_" + cmp  # Analysis name specification.
    ana_path = os.path.join(analysis, analysis + analysis_suffix)
    tool = "mdt"  # Analysis software.

    # Read mean squared displacement from file.
    file_suffix_msd = cmp + "_msd" + dim + "_layer.txt.gz"
    infile_msd = leap.simulation.get_ana_file(
        Sim, ana_path, tool, file_suffix_msd
    )
    msd_data = np.loadtxt(infile_msd)
    bins = msd_data[0]
    times = msd_data[1:, 0]
    msd_data = msd_data[1:, 1:]

    # Read mean displacement from file.
    file_suffix_md = cmp + "_md" + dim + "_layer.txt.gz"
    infile_md = leap.simulation.get_ana_file(
        Sim, ana_path, tool, file_suffix_md
    )
    md_data = np.loadtxt(infile_md)
    bins_md = md_data[0]
    times_md = md_data[1:, 0]
    md_data = md_data[1:, 1:]

    if bins_md.shape != bins.shape:
        raise ValueError(
            "The input files do not contain the same number of bins"
        )
    if not np.allclose(bins_md, bins, atol=0):
        raise ValueError("The bin edges are not the same in all input files")
    if times_md.shape != times.shape:
        raise ValueError(
            "The input files do not contain the same number of lag times"
        )
    if not np.allclose(times_md, times, atol=0):
        raise ValueError("The lag times are not the same in all input files")
    del bins_md, times_md

    bins *= length_conv
    times *= time_conv
    msd_data *= length_conv**2
    md_data *= length_conv

    if displvar:
        # Calculate displacement variance.
        msd_data -= md_data**2

    return times, bins, md_data, msd_data


def read_time_state_matrix(
    fname,
    fname_var=None,
    time_conv=1,
    amin=None,
    amax=None,
    n_rows_check=None,
    n_cols_check=None,
    states_check=None,
):
    """
    Read a time-state matrix from file.

    Read a data matrix from file where the first column contains the
    times, the first row contains the discrete states and the remaining
    matrix elements contain the corresponding data.

    Parameters
    ----------
    fname : str or bytes or os.PathLike
        Name of the file containing the data matrix.
    fname_var : str or bytes or os.PathLike or None, optional
        Optional name of the file containing the variance of the data.
        The file must have the same format as the first one.
    time_conv : scalar, optional
        Time conversion factor.  All times read from the input file are
        multiplied by this factor.
    amin : scalar, optional
        A minimum value that the data in the matrix must not undermine.
    amax : scalar, optional
        A maximum value that the data in the matrix must not exceed.
    n_rows_check, n_cols_check : int or None, optional
        Expected number of rows/columns of the matrix.  If provided, the
        number of rows/columns of the matrix differs from the given
        number, an exception will be raised.
    states_check : array_like or None, optional
        Expected state indices.  If provided, the state indices that
        were read from the input file are checked against the provided
        state indices.  If they differ, an exception will be raised.
        The length of `states_check` must be equal to `n_cols_check` if
        both are provided.

    Returns
    -------
    data : numpy.ndarray
        Array of shape ``(t, s)`` where ``t`` is the number of times and
        ``s`` is the number of different states.  The ij-th element of
        `data` is the data value for state j at time i.
    data_var : numpy.ndarray
        Array of the same shape as `data` containing the variance of the
        data.  Only returned if `fname_var` was provided.
    times : numpy.ndarray
        Array of shape ``(t,)`` containing the corresponding times.
    states : numpy.ndarray
        Array of shape ``(s,)`` containing the corresponding state
        indices.
    """
    data = np.loadtxt(fname)
    states = data[0, 1:]  # State indices.
    times = data[1:, 0]
    times *= time_conv
    data = data[1:, 1:]
    if amin is not None and np.any(data < amin):
        raise ValueError(
            "At least one value of the data is less than {}.  Input file:"
            " {}".format(amin, fname)
        )
    if amax is not None and np.any(data > amax):
        raise ValueError(
            "At least one value of the data is greater than {}.  Input file:"
            " {}".format(amax, fname)
        )
    if n_rows_check is not None and data.shape[0] != n_rows_check:
        raise ValueError(
            "The number of rows in the data matrix ({}) does not match the"
            " expected number of rows ({}).  Input file:"
            " {}".format(n_rows_check, data.shape[0], fname)
        )
    if n_cols_check is not None and data.shape[1] != n_cols_check:
        raise ValueError(
            "The number of columns in the data matrix ({}) does not match the"
            " expected number of columns ({}).  Input file:"
            " {}".format(n_cols_check, data.shape[1], fname)
        )
    if np.any(np.modf(states)[0] != 0):
        raise ValueError(
            "Some state indices are not integers but floats.  `states` ="
            " {}.  Input file: {}".format(states, fname)
        )
    states = states.astype(np.int64)
    if states_check is not None and not np.array_equal(states, states_check):
        raise ValueError(
            "The state indices from the input file ({}) do not match with the"
            " provided state indices ({}).  Input file:"
            " {}".format(states, states_check, fname)
        )

    if fname_var is not None:
        data_var, times_var, states_var = read_time_state_matrix(
            fname=fname_var,
            fname_var=None,
            time_conv=time_conv,
            amin=0,
            amax=None,
            n_rows_check=n_rows_check,
            n_cols_check=n_cols_check,
            states_check=states_check,
        )
        if data_var.shape != data.shape:
            raise ValueError(
                "`data_var.shape` ({}) != `data.shape` ({}).  Input file:"
                " ({})".format(data_var.shape, data.shape, fname_var)
            )
        if not np.allclose(times_var, times, rtol=0):
            raise ValueError(
                "`times_var` != `times`.  Input file: {}".format(fname_var)
            )
        if not np.array_equal(states_var, states):
            raise ValueError(
                "`states_var` != `states`.  Input file: {}".format(fname_var)
            )
        return data, data_var, times, states
    else:
        return data, times, states
