#!/usr/bin/env python3


"""
Bin the simulation box of a given (bulk) simulation in z direction in
equidistant bins.

This script was created for bulk simulations but can also be used for
surface simulations.
"""


# Standard libraries
import argparse

# Third-party libraries
import numpy as np

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Bin the simulation box of a given (bulk) simulation in z direction in"
        " equidistant bins."
    )
)
parser.add_argument(
    "--system",
    type=str,
    required=True,
    help="Name of the simulated system, e.g. lintf2_g1_20-1_sc80.",
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
args = parser.parse_args()

analysis = "density-z"  # Analysis name.
analysis_suffix = "_number"  # Analysis name specification.
tool = "gmx"  # Analysis software.
outfile = (
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis
    + analysis_suffix
    + "_Li_binsA.txt.gz"
)

# Desired bin width for the equidistant bins.
bw_desired = 10  # Angstrom


print("Creating Simulation instance(s)...")
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    top_path = "q%g" % surfq
else:
    surfq = None
    top_path = "bulk"
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, top_path)


print("Creating equidistant bins...")
box_z = Sim.box[2]
bins, bw_actual = leap.misc.gen_equidist_bins(0, box_z, bw_desired)
bin_widths = np.diff(bins, prepend=bins[0])


print("Creating output file(s)...")
data = np.column_stack([bins, bin_widths])
header = (
    "Equidistant bin edges for binning the z direction of the simulation box."
    + "\n\n"
    + "System:             {:s}\n".format(args.system)
    + "Settings:           {:s}\n".format(args.settings)
    + "\n"
    + "Lithium-to-ether-oxygen ratio: {:.4f}\n".format(Sim.Li_O_ratio)
    + "Ether oxygens per PEO chain:   {:d}\n".format(Sim.O_per_chain)
    + "\n"
    + "\n"
    + "Box length in z direction: {:>16.9e} A\n".format(box_z)
    + "Desired bin width:         {:>16.9e} A\n".format(bw_desired)
    + "Actual bin width:          {:>16.9e} A\n".format(bw_actual)
    + "\n"
    + "\n"
    + "The columns contain:\n"
    + "  1 Bin edges / A\n"
    + "  2 Bin widths / A\n"
    + "\n"
    + "Column number:\n"
    + "{:>14d}".format(1)
)
for col_num in range(2, data.shape[-1] + 1):
    header += " {:>16d}".format(col_num)
leap.io_handler.savetxt(outfile, data, header=header)

print("Created {}".format(outfile))
print("Done")
