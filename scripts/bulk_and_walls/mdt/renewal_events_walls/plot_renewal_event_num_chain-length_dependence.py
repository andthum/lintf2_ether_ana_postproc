#!/usr/bin/env python3


"""
Plot the number of renewal events in each bin as function of the PEO
chain length.

Only bins that correspond to actual layers (i.e. free-energy minima) at
the electrode interface are taken into account.

Layers/free-energy minima are clustered based on their distance to the
electrode.
"""


# Standard libraries
import argparse
import os

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# First-party libraries
import lintf2_ether_ana_postproc as leap


def legend_title(surfq_sign):
    r"""
    Create a legend title string.

    Parameters
    ----------
    surfq_sign : {"+", "-", r"\pm"}
        The sign of the surface charge.

    Returns
    -------
    title : str
        The legend title.

    Notes
    -----
    This function relies on global variables!
    """
    if surfq_sign not in ("+", "-", r"\pm"):
        raise ValueError("Unknown `surfq_sign`: '{}'".format(surfq_sign))
    return (
        r"$\sigma_s = "
        + surfq_sign
        + r" %.2f$ $e$/nm$^2$, " % Sims.surfqs[0]
        + r"$r = %.2f$" % Sims.Li_O_ratios[0]
        + "\n"
        + r"$F_{"
        + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
        + r"}$ Minima"
    )


def equalize_yticks(ax):
    """
    Equalize y-ticks so that plots can be better stacked together.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` for which to equalize the y
        ticks.
    """
    ylim = np.asarray(ax.get_ylim())
    ylim_diff = ylim[-1] - ylim[0]
    yticks = np.asarray(ax.get_yticks())
    yticks_valid = (yticks >= ylim[0]) & (yticks <= ylim[-1])
    yticks = yticks[yticks_valid]
    if ylim_diff >= 10 and ylim_diff < 20:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if np.all(yticks >= 0) and np.all(yticks < 10) and ylim_diff > 2:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the number of renewal events in each bin as function of the PEO"
        " chain length."
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
    default="Li-ether",
    choices=("Li-ether",),
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--prob-thresh",
    type=float,
    required=False,
    default=0.5,
    help=(
        "Only consider the number of renewal events in layers/free-energy"
        " minima whose prominence is at least such high that only"
        " 100*PROB_THRESH percent of the particles have a higher 1-dimensional"
        " kinetic energy.  Default: %(default)s"
    ),
)
parser.add_argument(
    "--n-clusters",
    type=lambda val: mdt.fh.str2none_or_type(val, dtype=int),
    required=False,
    default=None,
    help=(
        "The maximum number of layer clusters to consider.  'None' means all"
        " layer clusters.  Default: %(default)s"
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
if args.n_clusters is not None and args.n_clusters <= 0:
    raise ValueError(
        "--n-cluster ({}) must be a positive integer".format(args.n_clusters)
    )

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "renewal_events"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
ana_path = os.path.join(analysis, analysis + analysis_suffix)
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    settings
    + "_lintf2_peoN_20-1_gra_"
    + args.surfq
    + "_sc80_renewal_event_num"  # TODO: renewal_times
    + analysis_suffix
    + "_cluster_pthresh_%.2f" % args.prob_thresh
)
if args.common_ylim:
    outfile += "_common_ylim.pdf"
else:
    outfile += ".pdf"

# Columns to read from the files that contain the free-energy extrema.
pkp_col = 1  # Column that contains the peak positions in nm.
cols_fe = (pkp_col,)  # Peak positions [nm].
pkp_col_ix = cols_fe.index(pkp_col)

# Columns to read from the files that contain the bin population.
cols_pop = (
    0,  # Lower bin edges in [A].
    1,  # Upper bin edges in [A].
    5,  # Average number of compounds in each bin.
)

# Columns to read from the files that contain the renewal event
# information for the surface simulations.
cols_ri_walls = (
    6,  # z position of the reference compound at t0 in [A].
    12,  # z displacement of reference compound during tau_3 in [A].
)

# Columns to read from the files that contain the renewal event
# information for the bulk simulations.
cols_ri_bulk = (0,)  # Index of reference compound.

# Columns to read from the file that contains the bulk renewal times.
cols_rt_bulk = (
    0,  # Number of ether oxygens per PEO chain.
    29,  # Renewal time from rate method in [ns].
)

# The method to use for calculating the distance between clusters.  See
# `scipy.cluster.hierarchy.linkage`.
clstr_dist_method = "single"

# Number of decimal places to use for the assignment of bin edges to
# free-energy maxima.
decimals = 3


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_gra_" + args.surfq + "_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, args.surfq, sort_key="O_per_chain"
)


print("Reading data...")
# Read positions of free-energy extrema.
minima, n_minima_max = leap.simulation.read_free_energy_extrema(
    Sims, cmp1, peak_type="minima", cols=cols_fe
)
maxima, n_maxima_max = leap.simulation.read_free_energy_extrema(
    Sims, cmp1, peak_type="maxima", cols=cols_fe
)
if np.any(n_minima_max != n_maxima_max):
    raise ValueError(
        "Any `n_minima_max` ({}) != `n_maxima_max`"
        " ({})".format(n_minima_max, n_maxima_max)
    )

# Get files that contain the average number of compounds in each bin.
analysis_bin_pop = "discrete-z"
file_suffix_bin_pop = analysis_bin_pop + "_Li_bin_population.txt.gz"
infiles_bin_pop = leap.simulation.get_ana_files(
    Sims, analysis_bin_pop, tool, file_suffix_bin_pop
)

# Get files that contain the renewal event information for the surface
# simulations.
file_suffix_ri = analysis + analysis_suffix + ".txt.gz"
infiles_ri_walls = leap.simulation.get_ana_files(
    Sims, ana_path, tool, file_suffix_ri
)

# Read bulk renewal times.
SimPaths = leap.simulation.SimPaths()
fpath_rt_bulk = SimPaths.PATHS["bulk"]
fname_rt_bulk = (
    settings
    + "_lintf2_peoN_20-1_sc80_renewal_times_"
    + args.cmp
    + "_continuous.txt.gz"
)
for file in (
    "plots",
    tool,
    analysis,
    "chain-length_dependence",
    fname_rt_bulk,
):
    fpath_rt_bulk = os.path.join(fpath_rt_bulk, file)
n_eo, rt_bulk = np.loadtxt(fpath_rt_bulk, usecols=cols_rt_bulk, unpack=True)
n_eo = n_eo.astype(np.uint32)
if not np.all(n_eo[:-1] <= n_eo[1:]):
    raise ValueError(
        "The bulk simulations are not sorted by the number of ether oxygens"
        " per PEO chain"
    )
if not np.all(Sims.O_per_chain[:-1] <= Sims.O_per_chain[1:]):
    raise ValueError(
        "The surface simulations are not sorted by the number of ether oxygens"
        " per PEO chain"
    )
if not np.all(np.isin(Sims.O_per_chain, n_eo)):
    raise ValueError(
        "The numbers of ether oxygens per PEO chain in the surface simulations"
        " ({}) do not match to those in the bulk simulations"
        " ({})".format(Sims.O_per_chain, n_eo)
    )
valid_bulk_sim = np.isin(n_eo, Sims.O_per_chain)
rt_bulk = rt_bulk[valid_bulk_sim]
if len(rt_bulk) != Sims.n_sims:
    raise ValueError(
        "The number of bulk renewal times ({}) does not match to the number of"
        " surface simulations ({})".format(len(rt_bulk), Sims.n_sims)
    )
del n_eo, valid_bulk_sim

Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm.
bulk_start = Elctrd.BULK_START / 10  # A -> nm.
pk_pos_types = ("left", "right")
n_data = 8
# Data:
# 1) Free-energy minima positions (distance to electrode).
# 2) No. of reference compounds per layer (N_cmp^layer).
# 3) N_cmp^layer / N_cmp^tot.
# 4) No. of renewal events per layer (N_events^layer).
# 5) N_events^layer / N_cmp^layer.
# 6) (N_events^layer / N_cmp^layer) / (N_events^bulk / N_cmp^bulk).
# 7) (N_events^bulk / N_cmp^bulk) / (N_events^layer / N_cmp^layer).
# 8) As 7) * tau_3^bulk.
ydata = [
    [[[] for sim in Sims.sims] for pkp_type in pk_pos_types]
    for dat_ix in range(n_data)
]
n_events_bulk = np.zeros(Sims.n_sims, dtype=np.uint32)
n_refcmps_bulk = np.zeros_like(n_events_bulk)
for sim_ix, Sim in enumerate(Sims.sims):
    # Get file that contains the renewal event information for the
    # corresponding bulk simulation.
    Sim_bulk = Sim._get_BulkSim()
    infile_ri_bulk = leap.simulation.get_ana_file(
        Sim_bulk, ana_path, tool, file_suffix_ri
    )
    n_events_bulk_sim = len(np.loadtxt(infile_ri_bulk, usecols=cols_ri_bulk))
    n_refcmps_bulk_sim = Sim_bulk.top_info["res"][cmp1.lower()]["n_res"]
    n_refcmps_walls_tot = Sim.top_info["res"][cmp1.lower()]["n_res"]
    n_events_bulk[sim_ix] = n_events_bulk_sim
    n_refcmps_bulk[sim_ix] = n_refcmps_bulk_sim

    # Read number of reference compounds in each bin from file.
    bins_low, bins_up, n_refcmps_bins = np.loadtxt(
        infiles_bin_pop[sim_ix], usecols=cols_pop, unpack=True
    )
    bins_low /= 10  # Angstrom -> nm.
    bins_up /= 10  # Angstrom -> nm.

    # Read renewal event information for the surface simulation.
    pos_t0, displ = np.loadtxt(
        infiles_ri_walls[sim_ix], usecols=cols_ri_walls, unpack=True
    )
    pos_tau = pos_t0 + displ  # z position of refcmps at the renewal event.
    pos_tau /= 10  # Angstrom -> nm.
    del pos_t0, displ

    box_z = Sim.box[2] / 10  # Angstrom -> nm.
    for pkt_ix, pkp_type in enumerate(pk_pos_types):
        # Convert absolute positions to distance to the electrode.
        if pkp_type == "left":
            bins = bins_up - elctrd_thk
            n_refcmps_bins_pkt = n_refcmps_bins
            pos = pos_tau - elctrd_thk
        elif pkp_type == "right":
            bins = bins_low + elctrd_thk
            bins -= box_z
            bins *= -1  # Ensure positive distance values.
            bins, n_refcmps_bins_pkt = bins[::-1], n_refcmps_bins[::-1]
            pos = pos_tau + elctrd_thk
            pos -= box_z
            pos *= -1  # Ensure positive distance values.
        else:
            raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))

        # Assign bin edges to free-energy maxima.
        # -> Get number of reference compounds in each layer.
        maxima_pos = maxima[pkp_col_ix][pkt_ix][sim_ix]
        if not np.all(maxima_pos[:-1] <= maxima_pos[1:]):
            raise ValueError(
                "The positions of the free-energy maxima are not sorted"
            )
        if not np.all(bins[:-1] <= bins[1:]):
            raise ValueError("The bin edges are not sorted")
        maxima_pos = np.round(maxima_pos, decimals=decimals, out=maxima_pos)
        bins = np.round(bins, decimals=decimals, out=bins)
        valid_bins = np.isin(bins, maxima_pos)
        first_valid = np.argmax(valid_bins) - 1
        if first_valid >= 0 and n_refcmps_bins_pkt[first_valid] >= 1e-9:
            raise ValueError("A populated bin was marked as invalid.")
        n_refcmps_layer = n_refcmps_bins_pkt[valid_bins]
        if len(n_refcmps_layer) != len(maxima_pos):
            raise ValueError(
                "The number of valid bins ({}) is not equal to the number of"
                " free-energy maxima"
                " ({})".format(len(n_refcmps_layer), len(maxima_pos))
            )

        # Get number of renewal events in each layer.
        if np.any(pos <= 0) or np.any(pos >= box_z - 2 * elctrd_thk):
            raise ValueError(
                "The position of at least one renewal event lies within the"
                " electrodes"
            )
        layer_edges = np.insert(maxima_pos, 0, 0)
        n_events_layer = np.histogram(pos, layer_edges, density=False)[0]
        if len(n_events_layer) != len(maxima_pos):
            raise ValueError(
                "`len(n_events_layer)` ({}) != `len(maxima_pos)`"
                " ({})".format(len(n_events_layer), len(maxima_pos))
            )

        # Store data in list.
        n_events_per_refcmp_bulk = n_events_bulk_sim / n_refcmps_bulk_sim
        n_events_per_refcmp_layer = n_events_layer / n_refcmps_layer
        ydata[pkp_col_ix][pkt_ix][sim_ix] = minima[pkp_col_ix][pkt_ix][sim_ix]
        ydata[1][pkt_ix][sim_ix] = n_refcmps_layer
        ydata[2][pkt_ix][sim_ix] = n_refcmps_layer / n_refcmps_walls_tot
        ydata[3][pkt_ix][sim_ix] = n_events_layer
        ydata[4][pkt_ix][sim_ix] = n_events_per_refcmp_layer
        ydata[5][pkt_ix][sim_ix] = (
            n_events_per_refcmp_layer / n_events_per_refcmp_bulk
        )
        ydata[6][pkt_ix][sim_ix] = (
            n_events_per_refcmp_bulk / n_events_per_refcmp_layer
        )
        ydata[7][pkt_ix][sim_ix] = (
            rt_bulk[sim_ix]
            * n_events_per_refcmp_bulk
            / n_events_per_refcmp_layer
        )


print("Discarding free-energy minima whose prominence is too low...")
prom_min = leap.misc.e_kin(args.prob_thresh)
# prominences = deepcopy(minima[pkm_col_ix])
# n_pks_max = [0 for prom_pkt in prominences]
# for col_ix, yd_col in enumerate(ydata):
#     for pkt_ix, yd_pkt in enumerate(yd_col):
#         for sim_ix, yd_sim in enumerate(yd_pkt):
#             valid = prominences[pkt_ix][sim_ix] >= prom_min
#             ydata[col_ix][pkt_ix][sim_ix] = yd_sim[valid]
#             n_pks_max[pkt_ix] = max(n_pks_max[pkt_ix], np.count_nonzero(valid))
# del minima, prominences


print("Clustering peak positions...")


# print("Created {}".format(outfile))
print("Done")
