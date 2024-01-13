#!/usr/bin/env python3


"""
Extract energy terms from `Gromacs .edr files
<https://manual.gromacs.org/current/reference-manual/file-formats.html#edr>`_,
calculate their time averages and write them to a text file as function
of the PEO chain length.
"""


# Standard libraries
import argparse
import os

# Third-party libraries
import numpy as np

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Extract energy terms from Gromacs .edr files, calculate their time"
        " averages and write them to a text file as function of the PEO chain"
        " length."
    ),
)
parser.add_argument(
    "--settings",
    type=str,
    choices=("pr_nvt423_nh", "eq_npt423_pr_nh", "eq_npt303_pr_nh"),
    required=False,
    default="pr_nvt423_nh",
    help="Simulation settings.  Default: %(default)s",
)
args = parser.parse_args()

outfile = args.settings + "_lintf2_peoN_20-1_sc80_energy.txt.gz"

observables = (  # Energy terms to read from the .edr files.
    "Potential",
    "Kinetic En.",
    "Total Energy",
    "Temperature",
    "Pressure",
)
if "npt" in args.settings:
    observables += ("Volume", "Density")
begin = 50000  # First frame to read from the .edr files.
end = -1  # Last frame to read from the .edr files (exclusive).
every = 1  # Read ever n-th frame from the .edr files.


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_sc80"
set_pat = "[0-9][0-9]_"
if args.settings == "eq_npt423_pr_nh":
    set_pat += "[3-9]_"
set_pat += args.settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="O_per_chain"
)


print("Reading data...")
energies = np.full(
    (len(observables), 2, Sims.n_sims), np.nan, dtype=np.float64
)
for sim_ix, Sim in enumerate(Sims.sims):
    infile = Sim.settings + "_out_" + Sim.system + ".edr.gz"
    infile = os.path.join(Sim.path, infile)
    data, units = leap.io_handler.read_edr(
        infile, observables=observables, begin=begin, end=end, every=every
    )
    for obs_ix, obs in enumerate(observables):
        energies[obs_ix, 0, sim_ix] = np.mean(data[obs])
        energies[obs_ix, 1, sim_ix] = np.std(data[obs], ddof=1)
del data


print("Creating output...")
header = (
    "Time averaged energy terms obtained from Gromacs .edr files.\n"
    + "\n\n"
    + "Simulation settings: {:s}\n".format(args.settings)
    + "Li-to-EO ratio r:    {:.2f}\n".format(Sims.Li_O_ratios[0])
    + "\n"
    + "Frames read from the .edr files:\n"
    + "begin = {}\n".format(begin)
    + "end   = {}\n".format(end)
    + "every = {}\n".format(every)
    + "\n\n"
    + "The columns contain:\n"
    + "  1 n_EO: Number of ether oxygens per PEO chain\n"
    + "  2 N_Chain: Total number of PEO chains in the system\n"
    + "  3 N_Salt: Total number of LiTFSI ion pairs in the system\n"
)
data = np.column_stack(
    [Sims.O_per_chain, Sims.res_nums["solvent"], Sims.res_nums["cation"]]
)

col_num = 4
for obs_ix, obs in enumerate(observables):
    header += "{:>3d} {} / {}\n".format(col_num, obs, units[obs])
    header += "{:>3d} {} standard deviation / {}\n".format(
        col_num + 1, obs, units[obs]
    )
    col_num += 2
    data = np.column_stack([data, energies[obs_ix].T])

header += "\n"
header += "Column number:\n"
header += "{:>14d}".format(1)
for i in range(2, data.shape[-1] + 1):
    header += " {:>16d}".format(i)

leap.io_handler.savetxt(outfile, data, header=header)
print("Created {}".format(outfile))
print("Done")
