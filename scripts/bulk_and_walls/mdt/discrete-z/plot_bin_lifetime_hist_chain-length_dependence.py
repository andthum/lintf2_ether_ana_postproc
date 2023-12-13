#!/usr/bin/env python3


"""
Plot the lifetime histogram obtained from the count method for selected
bins for various chain lengths.
"""


# Standard libraries
import argparse

# Third-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# First-party libraries
import lintf2_ether_ana_postproc as leap


def histograms(dtrj_file, uncensored=False, intermittency=0):
    """TODO"""
    # Read discrete trajectory.
    dtrj = mdt.fh.load_dtrj(dtrj_file)
    n_frames = dtrj.shape[-1]

    if intermittency > 0:
        print("Correcting for intermittency...")
        dtrj = mdt.dyn.correct_intermittency(
            dtrj.T, args.intermittency, inplace=True, verbose=True
        )
        dtrj = dtrj.T

    # Get list of all lifetimes for each state.
    lts_per_state, states = mdt.dtrj.lifetimes_per_state(
        dtrj, uncensored=uncensored, return_states=True
    )
    states = states.astype(np.uint16)
    n_states = len(states)
    del dtrj

    # Calculate lifetime histogram for each state.
    # Binning is done in trajectory steps.
    # Linear bins.
    # step = 1
    # bins = np.arange(1, n_frames, step, dtype=np.float64)
    # Logarithmic bins.
    stop = int(np.ceil(np.log2(n_frames))) + 1
    bins = np.logspace(0, stop, stop + 1, base=2, dtype=np.float64)
    bins -= 0.5
    hists = np.full((n_states, len(bins) - 1), np.nan, dtype=np.float32)
    for state_ix, lts_state in enumerate(lts_per_state):
        if np.any(lts_state < bins[0]) or np.any(lts_state > bins[-1]):
            raise ValueError(
                "At least one lifetime lies outside the binned region"
            )
        hists[state_ix], _bins = np.histogram(
            lts_state, bins=bins, density=True
        )
        if not np.allclose(_bins, bins, rtol=0):
            raise ValueError(
                "`_bins` != `bins`.  This should not have happened"
            )
        if not np.isclose(np.sum(hists[state_ix] * np.diff(bins)), 1):
            raise ValueError(
                "The integral of the histogram ({}) is not close to"
                " one".format(np.sum(hists[state_ix] * np.diff(bins)))
            )
    del lts_per_state, lts_state, _bins
    return hists, bins.astype(np.float32), states


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the lifetime histogram obtained from the count method for"
        " selected bins for various chain lengths."
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
    required=False,
    default="Li",
    choices=("Li",),
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--prob-thresh",
    type=float,
    required=False,
    default=0.5,
    help=(
        "Only calculate the lifetime histogram for layers/free-energy minima"
        " whose prominence is at least such high that only 100*PROB_THRESH"
        " percent of the particles have a higher 1-dimensional kinetic energy."
        "  Default: %(default)s"
    ),
)
parser.add_argument(
    "--uncensored",
    required=False,
    default=False,
    action="store_true",
    help=(
        "Use the 'uncensored' counting method, i.e. discard truncated"
        " lifetimes at the trajectory edges."
    ),
)
parser.add_argument(
    "--intermittency",
    type=int,
    required=False,
    default=0,
    help=(
        "Maximum number of frames a compound is allowed to leave its state"
        " while still being considered to be in this state provided that it"
        " returns to this state after the given number of frames."
    ),
)
args = parser.parse_args()
if args.prob_thresh < 0 or args.prob_thresh > 1:
    raise ValueError(
        "--prob-thresh ({}) must be between 0 and 1".format(args.prob_thresh)
    )

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "discrete-z"  # Analysis name.
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    settings
    + "_lintf2_peoN_20-1_gra_"
    + args.surfq
    + "_sc80_"
    + analysis
    + "_"
    + args.cmp
    + "_lifetime_hist"
)
if args.uncensored:
    outfile += "_uncensored"
if args.intermittency > 0:
    outfile += "_intermittency_%d" % args.intermittency
outfile += "_pthresh_%.2f.pdf" % args.prob_thresh

# Time conversion factor to convert from trajectory steps to ns.
time_conv = 2e-3
# The method to use for calculating the distance between clusters.  See
# `scipy.cluster.hierarchy.linkage`.
clstr_dist_method = "single"

# Columns to read from the file containing the free-energy extrema
# (output file of `scripts/gmx/density-z/get_free-energy_extrema.py`).
pkp_col = 1  # Column that contains the peak positions in nm.
cols_fe = (pkp_col,)  # Peak positions [nm].
pkp_col_ix = cols_fe.index(pkp_col)


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_gra_" + args.surfq + "_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, args.surfq, sort_key="O_per_chain"
)


print("Reading data...")
# Read free-energy minima positions.
prom_min = leap.misc.e_kin(args.prob_thresh)
peak_pos, n_pks_max = leap.simulation.read_free_energy_extrema(
    Sims, args.cmp, peak_type="minima", cols=cols_fe, prom_min=prom_min
)

# Get filenames of the discrete trajectories.
file_suffix = analysis + "_" + args.cmp + "_dtrj.npz"
infiles_dtrj = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
n_infiles_dtrj = len(infiles_dtrj)

# Get filenames of the files containing the bins used to generate the
# discrete trajectories.
file_suffix = analysis + "_" + args.cmp + "_bins.txt.gz"
infiles_bins = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
n_infiles_bins = len(infiles_bins)
if n_infiles_bins != n_infiles_dtrj:
    raise ValueError(
        "`n_infiles_bins` ({}) != `n_infiles_dtrj` ({})".format(
            n_infiles_bins, n_infiles_dtrj
        )
    )

# Assign state indices to layers/free-energy minima.
Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm.
bulk_start = Elctrd.BULK_START / 10  # A -> nm.

pk_pos_types = ("left", "right")
# Bin edges used for generating the lifetime histograms.
hists_bins = [None for sim in Sims.sims]
# Lifetime histograms in the bulk and for each layer/free-energy minimum
hists_bulk = [None for sim in Sims.sims]
hists_layer = [[None for sim in Sims.sims] for pkp_type in pk_pos_types]
hists_layer_state_ix = [
    [None for sim in Sims.sims] for pkp_type in pk_pos_types
]
# Lower and upper bin edges used for generating the discrete trajectory
# -> Free-energy maxima.
# Don't confuse position bins (used to generate the discrete trajectory)
# with time bins (used to generate the lifetime histograms).  Time bins
# will always be prefixed with "hist_" or "hists_".
bin_edges = [[[None, None] for sim in Sims.sims] for pkp_type in pk_pos_types]
for sim_ix, Sim in enumerate(Sims.sims):
    box_z = Sim.box[2] / 10  # A -> nm
    bulk_region = Sim.bulk_region / 10  # A -> nm

    # Calculate the lifetime histogram for each state from the discrete
    # trajectory.
    hists_sim, hists_bins_sim, states_sim = histograms(
        infiles_dtrj[sim_ix],
        uncensored=args.uncensored,
        intermittency=args.intermittency,
    )
    hists_bulk[sim_ix] = hists_sim[len(states_sim) // 2]
    hists_bins[sim_ix] = hists_bins_sim
    del hists_bins_sim

    # Read bin edges.
    bins_sim = np.loadtxt(infiles_bins[sim_ix], dtype=np.float32)
    bins_sim /= 10  # A -> nm.
    # Lower and upper bin edges.
    bin_edges_sim = np.column_stack([bins_sim[:-1], bins_sim[1:]])
    bin_mids = bins_sim[1:] - np.diff(bins_sim) / 2
    del bins_sim
    # Select all bins for which lifetime histograms are available.
    bin_edges_sim = bin_edges_sim[states_sim]
    bin_mids = bin_mids[states_sim]
    bin_is_left = bin_mids <= (box_z / 2)
    del states_sim
    if np.any(bin_mids <= elctrd_thk):
        raise ValueError(
            "Simulation: '{}'.\n"
            "At least one bin lies within the left electrode.  Bin mid points:"
            " {}.  Left electrode: {}".format(Sim.path, bin_mids, elctrd_thk)
        )
    if np.any(bin_mids >= box_z - elctrd_thk):
        raise ValueError(
            "Simulation: '{}'.\n"
            "At least one bin lies within the right electrode.  Bin mid"
            " points: {}.  Right electrode:"
            " {}".format(Sim.path, bin_mids, box_z - elctrd_thk)
        )
    del bin_mids

    for pkt_ix, pkp_type in enumerate(pk_pos_types):
        if pkp_type == "left":
            valid_bins = bin_is_left
            hists_sim_valid = hists_sim[valid_bins]
            bin_edges_sim_valid = bin_edges_sim[valid_bins]
            # Convert absolute bin positions to distances to the
            # electrodes.
            bin_edges_sim_valid -= elctrd_thk
            # Use lower bin edges for assigning bins to layers.
            bin_edges_sim_valid_assign = bin_edges_sim_valid[:, 0]
        elif pkp_type == "right":
            valid_bins = ~bin_is_left
            hists_sim_valid = hists_sim[valid_bins]
            bin_edges_sim_valid = bin_edges_sim[valid_bins]
            # Reverse the order of rows to sort bins as function of
            # the distance to the electrodes in ascending order.
            hists_sim_valid = hists_sim_valid[::-1]
            bin_edges_sim_valid = bin_edges_sim_valid[::-1]
            # Convert absolute bin positions to distances to the
            # electrodes.
            bin_edges_sim_valid += elctrd_thk
            bin_edges_sim_valid -= box_z
            bin_edges_sim_valid *= -1  # Ensure positive distance values
            # Use upper bin edges for assigning bins to layers.
            bin_edges_sim_valid_assign = bin_edges_sim_valid[:, 1]
        else:
            raise ValueError(
                "Unknown peak position type: '{}'".format(pkp_type)
            )
        tolerance = 1e-6
        if np.any(bin_edges_sim_valid_assign < -tolerance):
            raise ValueError(
                "Simulation: '{}'.\n".format(Sim.path)
                + "Peak-position type: '{}'.\n".format(pkp_type)
                + "At least one bin lies within the electrode.  This should"
                + " not have happened.\n"
                + "Bin edges: {}.\n".format(bin_edges_sim_valid_assign)
                + "Electrode: 0"
            )

        # Assign bins to layers/free-energy minima.
        pk_pos = peak_pos[pkp_col_ix][pkt_ix][sim_ix]
        ix = np.searchsorted(pk_pos, bin_edges_sim_valid_assign)
        # Bins that are sorted after the last free-energy minimum lie
        # inside the bulk or near the opposite electrode and are
        # therefore discarded.
        layer_ix = ix[ix < len(pk_pos)]
        if not np.array_equal(layer_ix, np.arange(len(pk_pos))):
            raise ValueError(
                "Simulation: '{}'.\n".format(Sim.path)
                + "Peak-position type: '{}'.\n".format(pkp_type)
                + "Could not match each layer/free-energy minimum to exactly"
                + " one bin.\n"
                + "Bin edges: {}.\n".format(bin_edges_sim_valid_assign)
                + "Free-energy minima: {}.\n".format(pk_pos)
                + "Assignment: {}".format(layer_ix)
            )
        hists_layer[pkt_ix][sim_ix] = hists_sim_valid[layer_ix]
        bin_edges[pkt_ix][sim_ix] = bin_edges_sim_valid[layer_ix]
