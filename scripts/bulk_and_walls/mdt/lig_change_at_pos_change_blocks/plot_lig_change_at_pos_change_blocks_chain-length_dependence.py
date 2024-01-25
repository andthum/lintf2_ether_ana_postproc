#!/usr/bin/env python3

"""
Plot the number lithium-ion ligands that associate, dissociate or remain
coordinated during the crossing of a free-energy barrier as function of
the PEO chain length.

Free-energy barriers are clustered based on their distance to the
electrode.
"""


# Standard libraries
import argparse
import glob
import os
import warnings

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.cluster.hierarchy import dendrogram

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the number lithium-ion ligands that associate, dissociate or"
        " remain coordinated during the crossing of a free-energy barrier as"
        " function of the PEO chain length."
    )
)
parser.add_argument(
    "--surfq",
    type=str,
    required=True,
    choices=("q0", "q0.25", "q0.5", "q0.75", "q1"),
    help="Surface charge in e/nm^2.",
)
parser.add_argument(
    "--cmp",
    type=str,
    required=True,
    choices=("Li-OE", "Li-OBT", "Li-PEO", "Li-NTf2"),
    help=(
        "Compounds for which to plot the coordination change upon barrier"
        " crossing."
    ),
)
parser.add_argument(
    "--prob-thresh",
    type=float,
    required=False,
    default=0.5,
    help=(
        "Only consider free-energy barriers whose prominence is at least such"
        " high that only 100*PROB_THRESH percent of the particles have a"
        " higher 1-dimensional kinetic energy.  Default: %(default)s"
    ),
)
parser.add_argument(
    "--common-ylim",
    required=False,
    default=False,
    action="store_true",
    help="Use common y limits for all plots.",
)
args = parser.parse_args()
cmp1, cmp2 = args.cmp.split("-")
if args.prob_thresh < 0 or args.prob_thresh > 1:
    raise ValueError(
        "--prob-thresh ({}) must be between 0 and 1".format(args.prob_thresh)
    )

analysis_suffix = "_" + cmp1  # Analysis name specification.
if cmp2 in ("OE", "PEO"):
    analysis_suffix += "-OE"
elif cmp2 in ("OBT", "NTf2"):
    analysis_suffix += "-OBT"
else:
    raise ValueError("Unknown `cmp2`: '{}'".format(cmp2))

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "lig_change_at_pos_change_blocks"  # Analysis name.
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    settings
    + "_lintf2_peoN_20-1_gra_"
    + args.surfq
    + "_sc80_"
    + analysis
    + "_"
    + args.cmp
    + "_cluster_pthresh_%.2f" % args.prob_thresh
)
if args.common_ylim:
    outfile += "_common_ylim.pdf"
else:
    outfile += ".pdf"

# Rows and columns to read from the file that contains the ligand
# exchange information.
rows = (
    4,  # No. dissociated ligands.
    5,  # No. associated ligands.
    6,  # No. remained/stayed/persisted ligands.
    1,  # No. of valid barrier crossings.
    3,  # Average crossover time.
)
if cmp2 in ("OE", "OBT"):
    cols = list(range(2, 10))
elif cmp2 in ("PEO", "NTf2"):
    cols = list(range(10, 18))
else:
    raise ValueError("Unknown `cmp2`: '{}'".format(cmp2))

# Tolerance for matching the barrier position used in the ligand
# exchange analysis to free-energy maxima calculated from the density
# profile.
tol = 1e-4
# Required minimum prominence of the free-energy barriers/maxima.
prom_min = leap.misc.e_kin(args.prob_thresh)

# Index of the column that contains the free-energy barrier positions.
pkp_col_ix = 0

# The method to use for calculating the distance between clusters.  See
# `scipy.cluster.hierarchy.linkage`.
clstr_dist_method = "single"


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_gra_" + args.surfq + "_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, args.surfq, sort_key="O_per_chain"
)


print("Reading data...")
Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm.
bulk_start = Elctrd.BULK_START / 10  # A -> nm.

pk_pos_types = ("left", "right")
lig_change_types = ("Dissociated", "Associated", "Remained")
trans_drctn_types = ("Toward", "Away")  # Movement relative to electrode
trans_types = ("successful", "unsuccessful")
# Data:
# Plot type 1: (one plot for all clusters)
#   * Free-energy barrier/maxima positions (distance to electrode).
# Plot type 2: Number of Dissociated ligands (one plot per cluster)
#   * Toward, Successful (+/- Std).
#   * Toward, Unsuccessful (+/- Std).
#   * Away, Successful (+/- Std).
#   * Away, Unsuccessful (+/- Std).
# Plot type 3: Number of Associated ligands (one plot per cluster)
#   * Toward, Successful (+/- Std).
#   * Toward, Unsuccessful (+/- Std).
#   * Away, Successful (+/- Std).
#   * Away, Unsuccessful (+/- Std).
# Plot type 4: Number of Remained ligands (one plot per cluster)
#   * Toward, Successful (+/- Std).
#   * Toward, Unsuccessful (+/- Std).
#   * Away, Successful (+/- Std).
#   * Away, Unsuccessful (+/- Std).
# Plot type 5: Number of valid barrier crossings (one plot per cluster)
#   * Toward, Successful.
#   * Toward, Unsuccessful.
#   * Away, Successful.
#   * Away, Unsuccessful.
# Plot type 6: Fraction of valid barrier crossings that were successful
#   (one plot per cluster)
#   * Toward.
#   * Away.
# Plot type 7: Average crossover time (one plot per cluster)
#   * Toward, Successful (+/- Std).
#   * Toward, Unsuccessful (+/- Std).
#   * Away, Successful (+/- Std).
#   * Away, Unsuccessful (+/- Std).
n_data = 1  # Barrier positions.
n_data += len(rows) * len(trans_drctn_types) * len(trans_types) * 2
n_data -= len(trans_drctn_types) * len(trans_types)  # No Std for Type 5
n_data += 2  # Plot type 6.
ydata = [
    [[[] for sim in Sims.sims] for pkp_type in pk_pos_types]
    for dat_ix in range(n_data)
]

ana_parent_dir = os.path.join(tool, analysis + analysis_suffix)
for sim_ix, Sim in enumerate(Sims.sims):
    # Read free-energy barriers/maxima.
    fe_file_suffix = "free_energy_maxima_" + cmp1 + ".txt.gz"
    fe_analysis = "density-z"  # Analysis name.
    fe_tool = "gmx"  # Analysis software.
    infile = leap.simulation.get_ana_file(
        Sim, fe_analysis, fe_tool, fe_file_suffix
    )
    # Barrier positions in nm and prominences in kT.
    pk_pos, pk_prom = np.loadtxt(infile, usecols=(1, 3), unpack=True)

    # Read ligand exchange data.
    dir_pattern = analysis + analysis_suffix + "_[0-9]*"
    dir_pattern = os.path.join(Sim.path_ana, ana_parent_dir, dir_pattern)
    dirs = glob.glob(dir_pattern)
    if len(dirs) == 0:
        raise ValueError(
            "Could not find any directory matching the pattern"
            " '{}'".format(dir_pattern)
        )

    for directory in dirs:
        # File containing the position of the crossed free-energy
        # barrier.
        infile = (
            Sim.fname_ana_base + os.path.basename(directory) + "_bins.txt.gz"
        )
        infile = os.path.join(directory, infile)
        bins = np.loadtxt(infile)
        bins /= 10  # Angstrom -> nm.
        if len(bins) != 3:
            raise ValueError(
                "The bin file must contain exactly three bin edges."
            )
        barrier = bins[1]
        if (
            barrier < np.min(pk_pos) - 2 * tol
            or barrier > np.max(pk_pos) + 2 * tol
        ):
            # The crossing point does not match an actual free-energy
            # barrier, because it is too close to the electrodes.
            continue
        sort_ix = np.flatnonzero(np.isclose(pk_pos - barrier, 0, atol=1e-4))
        if len(sort_ix) != 1:
            raise ValueError(
                "The crossing point ({}) match with exactly one free-energy"
                " barrier ({})".format(barrier, pk_pos)
            )
        if pk_prom[sort_ix] < prom_min:
            # The prominence of the crossed free-energy barrier is lower
            # than the given threshold.
            continue

        # Check whether the free-energy barrier lies in the left or
        # right half of the simulation box.
        box_z = Sim.box[2] / 10  # Angstrom -> nm.
        if barrier <= box_z / 2:
            pkp_type = "left"
            # Convert absolute barrier position to distance to the
            # electrodes.
            barrier -= elctrd_thk
        else:
            pkp_type = "right"
            # Convert absolute barrier position to distance to the
            # electrodes.
            barrier += elctrd_thk
            barrier -= box_z
            barrier *= -1  # Ensure positive distance values.
        pkt_ix = pk_pos_types.index(pkp_type)
        ydata[pkp_col_ix][pkt_ix][sim_ix].append(barrier)

        # File containing the ligand exchange information.
        infile = Sim.fname_ana_base + os.path.basename(directory) + ".txt.gz"
        infile = os.path.join(directory, infile)
        lig_exchange_data = np.loadtxt(infile, usecols=cols)
        # Assignment of columns indices.
        # to/aw = toward/away; succ/unsc = successful, unsuccessful.
        if pkp_type == "left":
            to_succ_ix, to_succ_sd_ix = 2, 3
            to_unsc_ix, to_unsc_sd_ix = 6, 7
            aw_succ_ix, aw_succ_sd_ix = 0, 1
            aw_unsc_ix, aw_unsc_sd_ix = 4, 5
        elif pkp_type == "right":
            to_succ_ix, to_succ_sd_ix = 0, 1
            to_unsc_ix, to_unsc_sd_ix = 4, 5
            aw_succ_ix, aw_succ_sd_ix = 2, 3
            aw_unsc_ix, aw_unsc_sd_ix = 6, 7
        else:
            raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))
        # fmt: off
        col_ndx = (
            to_succ_ix, to_succ_sd_ix, to_unsc_ix, to_unsc_sd_ix,
            aw_succ_ix, aw_succ_sd_ix, aw_unsc_ix, aw_unsc_sd_ix,
        )
        # fmt: on
        data_ix = 1
        for row in rows:
            for col_ix in col_ndx:
                if row == 1 and col_ix % 2 != 0:
                    # No. of valid barrier crossings: Skip columns that
                    # contain standard deviations.
                    continue
                ydata[data_ix][pkt_ix][sim_ix].append(
                    lig_exchange_data[row][col_ix]
                )
                data_ix += 1
            if row == 1:
                # Fraction of valid barrier crossings that were
                # successful.
                to_succ_frac = lig_exchange_data[row][to_succ_ix] / (
                    lig_exchange_data[row][to_succ_ix]
                    + lig_exchange_data[row][to_unsc_ix]
                )
                ydata[data_ix][pkt_ix][sim_ix].append(to_succ_frac)
                data_ix += 1
                aw_succ_frac = lig_exchange_data[row][aw_succ_ix] / (
                    lig_exchange_data[row][aw_succ_ix]
                    + lig_exchange_data[row][aw_unsc_ix]
                )
                ydata[data_ix][pkt_ix][sim_ix].append(aw_succ_frac)
                data_ix += 1

# Convert lists to NumPy arrays and sort data by their distance to the
# electrode.
n_pks_max = np.zeros(len(pk_pos_types), np.uint32)
for col_ix, yd_col in enumerate(ydata):
    for pkt_ix, yd_pkt in enumerate(yd_col):
        for sim_ix, yd_sim in enumerate(yd_pkt):
            sort_ix = np.argsort(ydata[pkp_col_ix][pkt_ix][sim_ix])
            ydata[col_ix][pkt_ix][sim_ix] = np.asarray(yd_sim)[sort_ix]
            n_pks_max[pkt_ix] = max(n_pks_max[pkt_ix], len(yd_sim))


print("Clustering peak positions...")
(
    ydata,
    clstr_ix,
    linkage_matrices,
    n_clstrs,
    n_pks_per_sim,
    clstr_dist_thresh,
) = leap.clstr.peak_pos(
    ydata, pkp_col_ix, return_dist_thresh=True, method=clstr_dist_method
)
clstr_ix_unq = [np.unique(clstr_ix_pkt) for clstr_ix_pkt in clstr_ix]

# Sort clusters by ascending average peak position and get cluster
# boundaries.
clstr_bounds = [None for clstr_ix_pkt in clstr_ix]
for pkt_ix, clstr_ix_pkt in enumerate(clstr_ix):
    _clstr_dists, clstr_ix[pkt_ix], bounds = leap.clstr.dists_succ(
        ydata[pkp_col_ix][pkt_ix],
        clstr_ix_pkt,
        method=clstr_dist_method,
        return_ix=True,
        return_bounds=True,
    )
    clstr_bounds[pkt_ix] = np.append(bounds, bulk_start)

if np.any(n_clstrs < n_pks_max):
    warnings.warn(
        "Any `n_clstrs` ({}) < `n_pks_max` ({}).  This means different"
        " peaks of the same simulation were assigned to the same cluster."
        "  Try to decrease the threshold distance"
        " ({})".format(n_clstrs, n_pks_max, clstr_dist_thresh),
        RuntimeWarning,
        stacklevel=2,
    )

xdata = [None for n_pks_per_sim_pkt in n_pks_per_sim]
for pkt_ix, n_pks_per_sim_pkt in enumerate(n_pks_per_sim):
    if np.max(n_pks_per_sim_pkt) != n_pks_max[pkt_ix]:
        raise ValueError(
            "`np.max(n_pks_per_sim[{}])` ({}) != `n_pks_max[{}]` ({}).  This"
            " should not have happened".format(
                pkt_ix, np.max(n_pks_per_sim_pkt), pkt_ix, n_pks_max[pkt_ix]
            )
        )
    xdata[pkt_ix] = [
        Sims.O_per_chain[sim_ix]
        for sim_ix, n_pks_sim in enumerate(n_pks_per_sim_pkt)
        for _ in range(n_pks_sim)
    ]
    xdata[pkt_ix] = np.array(xdata[pkt_ix])


print("Creating plot(s)...")
