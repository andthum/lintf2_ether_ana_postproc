"""Module to handle Gromacs MD simulations."""


# Standard libraries
import glob
import os
import re

# Third-party libraries
import MDAnalysis as mda
import numpy as np
from scipy import constants as const

# First-party libraries
import lintf2_ether_ana_postproc as leap


#: Lennard-Jones radius of graphene atoms in Angstrom.
GRA_SIGMA_LJ = 3.55

#: Number of graphene layers.
GRA_LAYERS_N = 3

#: Distance between two graphene layers in Angstrom.
GRA_LAYER_DIST = 3.35

#: Electrode thickness in Angstrom.
ELCTRD_THK = (GRA_LAYERS_N - 1) * GRA_LAYER_DIST


class Simulation:
    """
    A Class representing a single Gromacs molecular dynamics simulation.

    All attributes should be treated as read-only attributes.  Don't
    change them after initialization.
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
            "lintf2_solvent_OE-Li-ratio(_charge-scaling)" (e.g.
            lintf2_g1_20-1_sc80) or
            "lintf2_solvent_OE-Li-ratio_electrode_surface-charge(_charge-scaling)"
            (e.g. lintf2_peo15_20-1_gra_q0.5_sc80), where
            "_charge-scaling" can be omitted.

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

        self._bulk_sim = None
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
        inferred from :attr:`self.universe`.

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

        self.universe = None
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
        Ratio of ether oxygen atoms to lithium ions  as inferred
        from :attr:`self.system`.

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
        Mass and number densities of the cation, the anion and the
        solvent as given by :attr:`self.res_names`.

        Mass densities are given in u/A^3, number densities are given in
        1/A^3.

        The densities will be calculated with respect to the accessible
        volume of the simulation box, i.e. the volume of the electrodes
        is subtracted from the total box volume.

        The simulation box is taken from ``self.gro_file``.

        :type: dict
        """

        path = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isdir(path):
            raise FileNotFoundError("No such directory: '{}'".format(path))
        self.path = os.path.abspath(path)
        self.get_system()
        self.get_settings()
        self.get_is_bulk()
        self._get_bulk_sim()
        self.get_surfq()
        self.get_temp()
        self.get_sim_files()
        self.get_res_names()
        self.get_universe()
        self.get_box()
        self.get_top_info()
        self.get_O_per_chain()
        self.get_O_Li_ratio()
        self.get_Li_O_ratio()
        self.get_vol()
        self.get_dens()

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

        self.system = os.path.basename(os.path.dirname(self.path))
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
                "The last part of `path` ({}) must contain the system name"
                " ({}) at the end but nowhere"
                " else".format(os.path.basename(self.path), self.system)
            )
        # Remove preceding number and trailing underscore.
        self.settings = self.settings[3:-1]
        return self.settings

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

        if re.search("_gra_q[0-9].*_", self.system) is None:
            self.is_bulk = True
        else:
            self.is_bulk = False
        return self.is_bulk

    def _get_bulk_sim_path(self):
        """
        Get the path to the corresponding bulk simulation (only relevant
        for surface simulations).

        Return the value of :attr:`self._bulk_sim_path`.  If
        :attr:`self._bulk_sim_path` is ``None``, assemble the path from
        :attr:`self.path`, :attr:`self.system`, and
        :attr:`self.settings`.

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

        bulk_sys = re.sub("_gra_q[0-9].*_", "_", self.system)
        bulk_sys_path = "../../../../bulk/" + bulk_sys
        bulk_sys_path = os.path.join(self.path, bulk_sys_path)
        bulk_sys_path = os.path.normpath(bulk_sys_path)
        if not os.path.isdir(bulk_sys_path):
            raise FileNotFoundError(
                "Could not find the corresponding bulk simulation.  No such"
                " directory: '{}'".format(bulk_sys_path)
            )

        glob_pattern = bulk_sys_path + "/*" + self.settings + "*"
        bulk_sim_path = glob.glob(glob_pattern)
        if len(bulk_sim_path) == 0:
            raise FileNotFoundError(
                "Could not find the corresponding bulk simulation.  The glob"
                " pattern '{}' does not match anything".format(glob_pattern)
            )
        bulk_sim_path = bulk_sim_path[0]
        if not os.path.isdir(bulk_sim_path):
            raise FileNotFoundError(
                "Could not find the corresponding bulk simulation.  No such"
                " directory: '{}'".format(bulk_sim_path)
            )

        self._bulk_sim_path = bulk_sim_path
        return self._bulk_sim_path

    def _get_bulk_sim(self):
        """
        Get the corresponding bulk
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation` (only
        relevant for surface simulations).

        Return the value of :attr:`self._bulk_sim`.  If
        :attr:`self._bulk_sim` is ``None``, create the bulk
        :class:`Simulation` from :attr:`self._bulk_sim_path`.

        Returns
        -------
        self._bulk_sim :
            :class:`lintf2_ether_ana_postproc.simulation.Simulation`

            The corresponding bulk
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.
            If the current
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation` is
            already a bulk simulation, this is ``None``.
        """
        if self._bulk_sim is not None:
            return self._bulk_sim

        if self._bulk_sim_path is None:
            self._get_bulk_sim_path()

        if self.is_bulk:
            # This simulation is already a bulk simulation.
            self._bulk_sim = None
        else:
            self._bulk_sim = leap.simulation.Simulation(self._bulk_sim_path)
        return self._bulk_sim

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

        system = self.system.split("_")
        for s in system:
            if s.startswith("q"):
                self.surfq = s[1:]
                break
        if self.surfq is None:
            raise ValueError(
                "Could not infer the surface charge from `self.system`"
                " ({})".format(self.system)
            )
        self.surfq = float(self.surfq)
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
        e/Angstrom^2 as inferred from :attr:`self.universe`.

        Return the value of :attr:`self._surfq_u`.  If
        :attr:`self._surfq_u` is ``None``, infer the surface charge from
        :attr:`self.universe`.

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
        if self.universe is None:
            self.get_universe()
        if self.box is None:
            self.get_box()

        if self.is_bulk:
            self._surfq_u = None
            return self._surfq_u

        elctrds = "not resname "
        elctrds += " and not resname ".join(self.res_names.values())
        elctrds = self.universe.select_atoms(elctrds)
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

        if self._bulk_sim is not None and not np.isclose(
            self.temp, self._bulk_sim.temp, rtol=0
        ):
            raise ValueError(
                "`self.temp` ({}) != `self._bulk_sim.temp`"
                " ({})".format(self.temp, self._bulk_sim.temp)
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

        _counts = list(range(2, 6))
        if self.system.count("_") not in _counts or not self.system.startswith(
            "lintf2"
        ):
            raise ValueError(
                "The system name ({}) must follow the pattern"
                " 'lintf2_solvent_OE-Li-ratio_charge-scaling' or"
                " 'lintf2_solvent_OE-Li-ratio_electrode_surface-charge"
                "_charge-scaling' (`_charge-scaling' can be"
                " omitted)".format(self.system)
            )
        salt = self.system.split("_")[0]
        self.res_names = {}
        self.res_names["cation"] = salt[:2]
        self.res_names["anion"] = salt[2:]
        solvent = self.system.split("_")[1]
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
            self._bulk_sim is not None
            and self.res_names != self._bulk_sim.res_names
        ):
            raise ValueError(
                "`self.res_names` ({}) != `self._bulk_sim.res_names`"
                " ({})".format(self.res_names, self._bulk_sim.res_names)
            )

        return self.res_names

    def get_universe(self):
        """
        Get an MDAnalysis :class:`~MDAnalysis.core.universe.Universe`
        created from the output .gro file of the simulation.

        Return the value of :attr:`self.universe`.  If
        :attr:`self.universe` is ``None``, create an MDAnalysis
        :class:`~MDAnalysis.core.universe.Universe` from the output .gro
        file of the simulation as given by :attr:`self.gro_file`.

        Returns
        -------
        self.universe : MDAnalysis.core.universe.Universe
            The created :class:`~MDAnalysis.core.universe.Universe`.
        """
        if self.universe is not None:
            return self.universe

        if self.is_bulk is None:
            self.get_is_bulk()
        if self.sim_files is None:
            self.get_sim_files()

        self.universe = mda.Universe(
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

            if "B1" not in self.universe.residues.resnames:
                raise ValueError(
                    "The simulation is a surface simulation but it does not"
                    " contain the residue 'B1'"
                )
            elctrd_bot = self.universe.select_atoms("resname B1")
            elctrd_bot_pos_z = elctrd_bot.positions[:, 2]
            if not np.allclose(
                elctrd_bot_pos_z, ELCTRD_THK, rtol=0, atol=1e-6
            ):
                raise ValueError(
                    "`elctrd_bot_pos_z` ({}) != `ELCTRD_THK` ({})".format(
                        (np.min(elctrd_bot_pos_z), np.max(elctrd_bot_pos_z)),
                        ELCTRD_THK,
                    )
                )

            if "T1" not in self.universe.residues.resnames:
                raise ValueError(
                    "The simulation is a surface simulation but it does not"
                    " contain the residue 'T1'"
                )
            elctrd_top = self.universe.select_atoms("resname T1")
            elctrd_top_pos_z = elctrd_top.positions[:, 2]
            box_z = self.universe.dimensions[2]
            if not np.allclose(
                elctrd_top_pos_z, box_z - ELCTRD_THK, rtol=0, atol=1e-2
            ):
                raise ValueError(
                    "`elctrd_top_pos_z` ({}) != `box_z - ELCTRD_THK`"
                    " ({})".format(
                        (np.min(elctrd_top_pos_z), np.max(elctrd_top_pos_z)),
                        box_z - ELCTRD_THK,
                    )
                )

        return self.universe

    def get_box(self):
        """
        Get the simulation box dimensions of the simulated system.

        Return the value of :attr:`self.box`.  If :attr:`self.box` is
        ``None``, get the box dimensions from :attr:`self.universe`.

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

        if self.universe is None:
            self.get_universe()

        self.box = self.universe.dimensions
        return self.box

    def get_top_info(self):
        """
        Get information about the topology of the simulated system.

        Return the value of :attr:`self.top_info`.  If
        :attr:`self.top_info` is ``None``, create it by extracting the
        relevant information from :attr:`self.universe`.

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
        if self.universe is None:
            self.get_universe()

        ag = self.universe.atoms
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
        if self._bulk_sim is not None and any(
            self.top_info["res"][rn] != self._bulk_sim.top_info["res"][rn]
            for rn in self.res_names.values()
        ):
            raise ValueError(
                "self.top_info['res'] ({}) != self._bulk_sim.top_info['res']"
                " ({})".format(
                    self.top_info["res"], self._bulk_sim.top_info["res"]
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
            self._bulk_sim is not None
            and self.O_per_chain != self._bulk_sim.O_per_chain
        ):
            raise ValueError(
                "`self.O_per_chain` ({}) != `self._bulk_sim.O_per_chain`"
                " ({})".format(self.O_per_chain, self._bulk_sim.O_per_chain)
            )

        return self.O_per_chain

    def get_O_Li_ratio(self):
        """
        Get the ratio of ether oxygen atoms to lithium ions.

        Return the value of :attr:`self.O_Li_ratio`.  If
        :attr:`self.O_Li_ratio` is ``None``, calculate the OE-Li ratio
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
        if self._bulk_sim is not None and not np.isclose(
            self.O_Li_ratio, self._bulk_sim.O_Li_ratio, rtol=0
        ):
            raise ValueError(
                "`self.O_Li_ratio` ({}) != `self._bulk_sim.O_Li_ratio`"
                " ({})".format(self.O_Li_ratio, self._bulk_sim.O_Li_ratio)
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

        _counts = list(range(2, 6))
        if self.system.count("_") not in _counts:
            raise ValueError(
                "The system name ({}) must follow the pattern"
                " 'lintf2_solvent_OE-Li-ratio_charge-scaling' or"
                " 'lintf2_solvent_OE-Li-ratio_electrode_surface-charge"
                "_charge-scaling' (`_charge-scaling' can be"
                " omitted)".format(self.system)
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
        if self._bulk_sim is not None and not np.isclose(
            self._O_Li_ratio_sys, self._bulk_sim._O_Li_ratio_sys, rtol=0
        ):
            raise ValueError(
                "`self._O_Li_ratio_sys` ({}) !="
                " `self._bulk_sim._O_Li_ratio_sys` ({})".format(
                    self._O_Li_ratio_sys, self._bulk_sim._O_Li_ratio_sys
                )
            )

        return self._O_Li_ratio_sys

    def get_Li_O_ratio(self):
        """
        Get the ratio of lithium ions to ether oxygen atoms.

        Return the value of :attr:`self.Li_O_ratio`.  If
        :attr:`self.Li_O_ratio` is ``None``, calculate the Li-OE ratio
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

        if self._bulk_sim is not None and not np.isclose(
            self.Li_O_ratio, self._bulk_sim.Li_O_ratio, rtol=0
        ):
            raise ValueError(
                "`self.Li_O_ratio` ({}) != `self._bulk_sim.Li_O_ratio`"
                " ({})".format(self.Li_O_ratio, self._bulk_sim.Li_O_ratio)
            )

        return self.Li_O_ratio

    def get_vol(self):
        """
        Get the total and accessible volume of the simulation box in
        Angstrom cubed.

        Return the value of :attr:`self.vol`.  If :attr:`self.vol` is
        ``None``, it is calculated from the box information contained in
        :attr:`self.universe`.  It is assumed that the box is
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
            elctrd_thk = 2 * ELCTRD_THK + GRA_SIGMA_LJ
            vol_elctrd = elctrd_thk * np.prod(self.box[:2])
        vol_access = vol_tot - vol_elctrd
        if vol_access <= 0:
            raise ValueError("`vol_access` ({}) <= 0".format(vol_access))

        self.vol = {
            "tot": vol_tot,
            "access": vol_access,
            "elctrd": vol_elctrd,
        }

        if self._bulk_sim is not None and not np.isclose(
            self.vol["access"], self._bulk_sim.vol["access"], rtol=0, atol=5
        ):
            raise ValueError(
                "`self.vol['access']` ({}) != `self._bulk_sim.vol['access']`"
                " ({})".format(
                    self.vol["access"], self._bulk_sim.vol["access"]
                )
            )

        return self.vol

    def get_dens(self):
        """
        Get the mass and number densities of the cation, the anion and
        the solvent as given by :attr:`self.res_names`.

        Return the value of :attr:`self.n_dens`.  If :attr:`self.n_dens`
        is ``None``, calculate the densities from :attr:`self.universe`
        and :attr:`self.vol`.

        Mass densities are given in u/A^3, number densities are given in
        1/A^3.

        Returns
        -------
        self.n_dens : dict
            Mass and number densities of the cation, the anion and the
            solvent as given by :attr:`self.res_names`.

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
        if self.universe is None:
            self.get_universe()
        if self.vol is None:
            self.get_vol()

        vol = self.vol["access"]
        self.dens = {}
        n_tot, m_tot = 0, 0
        for res_type, rn in self.res_names.items():
            ag = self.universe.select_atoms("resname {}".format(rn))
            n_res = ag.n_residues
            n_tot += n_res
            m_res = ag.residues[0].mass
            m_tot += m_res * n_res
            res_dct = {"num": n_res / vol, "mass": m_res * n_res / vol}
            self.dens[res_type] = res_dct
        self.dens["tot"] = {"num": n_tot / vol, "mass": m_tot / vol}

        if self._bulk_sim is not None:
            for res_type, dens_dct in self.dens.items():
                for dens_type, density in dens_dct.items():
                    if not np.isclose(
                        density,
                        self._bulk_sim.dens[res_type][dens_type],
                        rtol=0,
                        atol=1e-5,
                    ):
                        raise ValueError(
                            "`self.dens` ({}) != `self._bulk_sim.dens`"
                            " ({})".format(self.dens, self.dens)
                        )

        return self.dens


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

    def __init__(self, *paths):
        """
        Initialize an instance of the
        :class:`~lintf2_ether_ana_postproc.simulation.Simulations`
        class.

        Parameters
        ----------
        path : str or bytes or os.PathLike
            Relative or absolute paths to the directories containing the
            input and output files of the Gromacs MD simulations you
            want to use.  See
            :class:`lintf2_ether_ana_postproc.simulation.Simulation` for
            details.

            The simulations in this container class are sorted in the
            order of the input paths.  Duplicate simulations are kept.
        """
        # Instance attributes.
        self.sims = None
        """
        List containing all stored
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        instances.

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
        Dictionary containing the mass and number density of the cation,
        the anion and the solvent for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        Mass densities are given in kg/m^3, number densities are given
        in 1/nm^3.

        :type: dict
        """

        self.paths = []
        for path in paths:
            path = os.path.expandvars(os.path.expanduser(path))
            if not os.path.isdir(path):
                raise FileNotFoundError("No such directory: '{}'".format(path))
            self.paths.append(os.path.abspath(path))
        self.get_sims()
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

    def get_sims(self):
        """
        Get the stored
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        instances.

        Return the value of :attr:`self.sims`.  If :attr:`self.sims` is
        ``None``, create
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`
        instances from :attr:`self.paths` and store them in
        :attr:`self.sims`.

        Returns
        -------
        self.sims : list
            The list of stored
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
        Get the mass and number densities of the cation, the anion and
        the solvent for each
        :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.

        Return the value of :attr:`self.dens`.  If :attr:`self.dens` is
        ``None``, create it from :attr:`self.sims`.

        Mass densities are given in kg/m^3, number densities are given
        in 1/nm^3.

        Returns
        -------
        self.dens : dict
            Dictionary containing the mass and number density of the
            cation, the anion and the solvent for each
            :class:`~lintf2_ether_ana_postproc.simulation.Simulation`.
        """
        if self.dens is not None:
            return self.dens

        if self.sims is None:
            self.get_sims()

        dens = {
            res_type: {dens_type: [] for dens_type in dens_dct.keys()}
            for res_type, dens_dct in self.sims[0].dens.items()
        }
        for sim in self.sims:
            for res_type, dens_dct in sim.dens.items():
                for dens_type, density in dens_dct.items():
                    dens[res_type][dens_type].append(density)
        self.dens = {res_type: {} for res_type in dens.keys()}
        for res_type, dens_dct in dens.items():
            for dens_type, density in dens_dct.items():
                dens_array = np.array(density)
                if dens_type == "num":
                    dens_array *= 1e3  # 1/A^3 -> 1/nm^3
                elif dens_type == "mass":
                    dens_array *= self.MDENS2SI  # u/A^3 -> kg/m^3
                self.dens[res_type][dens_type] = dens_array
        return self.dens


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
