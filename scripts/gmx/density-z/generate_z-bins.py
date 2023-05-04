#!/usr/bin/env python3


"""
Bin the simulation box of a given simulation in z direction based on the
maxima of the free-energy profile of the given compound.
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
        "Bin the simulation box of a given simulation in z direction based on"
        " the maxima of the free-energy profile of the given compound."
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
    choices=("Li", "NBT", "OBT", "OE"),
    help="Compound.  Default: %(default)s",
)
args = parser.parse_args()

analysis = "density-z"  # Analysis name.
analysis_suffix = "_number"  # Analysis name specification.
tool = "gmx"  # Analysis software.
# `outfile`: "_density-z_number_" should better be "free_energy", but to
# be compatible with my old simulations I kept it as is.
outfile = (
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis
    + analysis_suffix
    + "_"
    + args.cmp
    + "_binsA.txt.gz"
)

# Columns to read from the file containing the density profile.
cols_dens = (0,)  # bin edges [nm]
if args.cmp == "Li":
    cols_dens += (2,)  # Li number density [nm^-3]
elif args.cmp == "NBT":
    cols_dens += (5,)  # NBT number density [nm^-3]
elif args.cmp == "OBT":
    cols_dens += (6,)  # OBT number density [nm^-3]
elif args.cmp == "OE":
    cols_dens += (7,)  # OE number density [nm^-3]
else:
    raise ValueError("Unknown --cmp ({})".format(args.cmp))

# Column to read from the file containing the free-energy maxima.
col_fe_max = 1  # Peak positions [nm]

# Desired bin width for the equidistant bins in the "bulk region", i.e.
# between the last free-energy maximum at the left electrode and the
# first free-energy maximum at the right electrode.
bw_desired = 10  # Angstrom

# Minimum distance of the last bin edge within the electrolyte from the
# electrode surface.
min_dist_elctrd = 2  # Angstrom


print("Creating Simulation instance(s)...")
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    top_path = "q%g" % surfq
else:
    surfq = None
    top_path = "bulk"
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, top_path)


print("Reading data...")
Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK
box_z = Sim.box[2]
bulk_dens = Sim.dens["atm_type"][args.cmp]["num"]

# Read density profile.
file_suffix = analysis + analysis_suffix + ".xvg.gz"
infile_dens = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
xdata, ydata = np.loadtxt(
    infile_dens, comments=["#", "@"], usecols=cols_dens, unpack=True
)
xdata *= 10  # nm -> A
ydata *= 1e-3  # 1/nm^3 -> 1/A^3
ydata /= bulk_dens

# Read free-energy maxima.
file_suffix = "free_energy_maxima_" + args.cmp + ".txt.gz"
infile_fe_max = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
pk_pos = np.loadtxt(infile_fe_max, usecols=col_fe_max)
pk_pos *= 10  # nm -> A
pk_is_left = pk_pos <= (box_z / 2)
if np.any(pk_pos <= elctrd_thk):
    raise ValueError(
        "At least one peak lies within the left electrode.  Peak positions:"
        " {}.  Left electrode: {}".format(pk_pos, elctrd_thk)
    )
if np.any(pk_pos >= box_z - elctrd_thk):
    raise ValueError(
        "At least one peak lies within the right electrode.  Peak positions:"
        " {}.  Right electrode: {}".format(pk_pos, box_z - elctrd_thk)
    )
if np.any((pk_pos >= Sim.bulk_region[0]) & (pk_pos <= Sim.bulk_region[1])):
    raise ValueError(
        "At least one peak lies within the bulk region.  Peak positions: {}."
        "  Bulk region: {}".format(pk_pos, Sim.bulk_region)
    )
if surfq == 0:
    n_pks_left = np.count_nonzero(pk_is_left)
    n_pks_right = len(pk_is_left) - n_pks_left
    if n_pks_left != n_pks_right:
        raise ValueError(
            "The surface charge is {} e/nm^2 but the number of left ({}) and"
            " right free-energy maxima ({}) do not"
            " match.".format(surfq, n_pks_left, n_pks_right)
        )


print("Creating bins...")
# Equidistant bins in the "bulk region", i.e. between the last
# free-energy maximum at the left electrode and the first free-energy
# maximum at the right electrode.
start = pk_pos[pk_is_left][-1]
stop = pk_pos[~pk_is_left][0]
bins, bw_actual = leap.misc.gen_equidist_bins(start, stop, bw_desired)
# Discard `start` and `stop`, because they are already contained in
# `pk_pos`.
bins = bins[1:-1]

# Bins in the "layering region" = free-energy-maxima positions.
bins = np.insert(bins, 0, pk_pos[pk_is_left])
bins = np.append(bins, pk_pos[~pk_is_left])

# Third and third-last bin = onset and end of the density profile
# (`ydata`), except if they are closer to the electrode surface than
# `min_dist_elctrd`.
if Sim.surfq is None or Sim.surfq == 0:
    # Use symmetrized density profile for bulk simulations and surface
    # simulations with zero surface charge.
    xdata_3rd_bin, ydata_3rd_bin = leap.misc.symmetrize_data(
        xdata, ydata, x2shift=box_z, reassemble=True, tol=1e-3
    )
else:
    xdata_3rd_bin, ydata_3rd_bin = xdata, ydata
non_zero_first, non_zero_last = np.flatnonzero(ydata_3rd_bin)[[0, -1]]
non_zero_first = max(non_zero_first - 1, 0)
non_zero_last = min(non_zero_last + 1, len(xdata_3rd_bin) - 1)
if xdata_3rd_bin[non_zero_first] - elctrd_thk < min_dist_elctrd:
    prepend = None
else:
    prepend = xdata_3rd_bin[non_zero_first]
if box_z - elctrd_thk - xdata_3rd_bin[non_zero_last] < min_dist_elctrd:
    append = None
else:
    append = xdata_3rd_bin[non_zero_last]
if Sim.surfq is None or Sim.surfq == 0:
    if (prepend is None and append is not None) or (
        prepend is not None and append is None
    ):
        raise ValueError(
            "The simulation is a bulk simulation or a surface simulation with"
            " a surface charge of zero, but the third ({}) and third-last ({})"
            " bin edge do not match".format(prepend, append)
        )
elif Sim.surfq > 0.025:
    msg = (
        "The simulation is a surface simulation with a surface charge of"
        + " {:.4f} e/A^2".format(Sim.surfq)
    )
    if args.cmp == "Li":
        msg += " and the selected compound is {}".format(args.cmp)
        if prepend is None:
            msg += " but the third ({}) bin edge is None".format(prepend)
            raise ValueError(msg)
        elif append is not None:
            msg += " but the third-last ({}) bin edge is not None".format(
                append
            )
            raise ValueError(msg)
    elif args.cmp in ("NBT", "OBT"):
        msg += " and the selected compound is {}".format(args.cmp)
        if prepend is not None:
            msg += " but the third ({}) bin edge is not None".format(prepend)
            raise ValueError(msg)
        if append is None:
            msg += " but the third-last ({}) bin edge is None".format(append)
            raise ValueError(msg)
bins = leap.misc.extend_bins(bins, prepend, append)
del xdata_3rd_bin, ydata_3rd_bin

# Second and second-last bin = electrode surface.
bins = leap.misc.extend_bins(bins, elctrd_thk, box_z - elctrd_thk)

# First and last bin = box edges.
bins = leap.misc.extend_bins(bins, 0, box_z)


print("Creating output file(s)...")
tol = 0.05

# Bin widths.
bin_widths = np.diff(bins, prepend=bins[0])

# Distance of the bins to the left/right electrode surface.
bins_dist_left = bins - elctrd_thk
bins_dist_right = box_z - elctrd_thk - bins
if Sim.surfq is None or Sim.surfq == 0:
    first_3_bins = bins_dist_left[:3]
    last_3_bins = bins_dist_right[len(bins_dist_right) - 3 :][::-1]
    if not np.allclose(first_3_bins, last_3_bins, rtol=0, atol=tol):
        raise ValueError(
            "The simulation is a bulk simulation or a surface simulation with"
            " a surface charge of zero, but three first ({}) and the three"
            "  last ({}) bin edges do not"
            " match".format(first_3_bins, last_3_bins)
        )

# Density and free-energy values at the bin edges.
ix = leap.misc.find_nearest(xdata, bins, tol=tol)
free_en = leap.misc.dens2free_energy(
    xdata, ydata, bulk_region=Sim.bulk_region, tol=tol
)
free_en = free_en[ix]
ydata = ydata[ix]

# Write output.
data = np.column_stack(
    [bins, ydata, free_en, bin_widths, bins_dist_left, bins_dist_right]
)
header = (
    "Bin edges for binning the z direction of the simulation box.\n"
    + "\n"
    + "System:             {:s}\n".format(args.system)
    + "Settings:           {:s}\n".format(args.settings)
    + "Density profile:    {:s}\n".format(infile_dens)
    + "Read Column(s):     {}\n".format(np.array(cols_dens) + 1)
    + "Free-energy maxima: {:s}\n".format(infile_fe_max)
    + "Read Column(s):     {}\n".format(np.array(col_fe_max) + 1)
    + "\n"
    + "Compound:                      {:s}\n".format(args.cmp)
    + "Surface charge:                {:.2f} e/nm^2\n".format(surfq)
    + "Lithium-to-ether-oxygen ratio: {:.4f}\n".format(Sim.Li_O_ratio)
    + "Ether oxygens per PEO chain:   {:d}\n".format(Sim.O_per_chain)
    + "\n"
    + "\n"
    + "Bin edge generation procedure:\n"
    + "  * The first and last bin edges are the box edges.\n"
    + "  * The second and second-last bin edges are the electrode surfaces.\n"
    + "  * The third and third-last bin edges are the onset and end of the\n"
    + "    density profile, except if they are closer to the electrode\n"
    + "    surface than {:.2f} A.\n".format(min_dist_elctrd)
    + "  * The bin edges in the layering regime are inferred from the\n"
    + "    free-energy maxima which are read from the corresponding input\n"
    + "    file.\n"
    + "  * The bin edges in the bulk regime are chosen equidistant with a\n"
    + "    distance of roughly {:.2f} A.\n".format(bw_desired)
    + "\n"
    + "Box edges:          {:>16.9e}, {:>16.9e} A\n".format(0, box_z)
    + "Electrode surfaces: {:>16.9e}, {:>16.9e} A\n".format(
        elctrd_thk, box_z - elctrd_thk
    )
    + "Min. dist. to electrode surf.:        {:>16.9e} A\n".format(
        min_dist_elctrd
    )
    + "Desired bin width in the bulk regime: {:>16.9e} A\n".format(bw_desired)
    + "Actual bin width in the bulk regime:  {:>16.9e} A\n".format(bw_actual)
    + "\n"
    + "\n"
    + "The columns contain:\n"
    + "  1 Bin edges / A\n"
    + "  2 Density rho(z) / rho(bulk)\n"
    + "  3 Free energy F(z) / kT\n"
    + "  4 Bin widths / A\n"
    + "  5 Distance of the bin edges to the left electrode surface / A\n"
    + "  6 Distance of the bin edges to the right electrode surface / A\n"
    + "\n"
    + "Column number:\n"
    + "{:>14d}".format(1)
)
for col_num in range(2, data.shape[-1] + 1):
    header += " {:>16d}".format(col_num)
leap.io_handler.savetxt(outfile, data, header=header)

print("Created {}".format(outfile))
print("Done")
