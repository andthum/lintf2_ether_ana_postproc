#!/usr/bin/env python3


"""
Plot the number of renewal events in each bin as function of the
electrode surface charge.

Only bins that correspond to actual layers (i.e. free-energy minima) at
the electrode interface are taken into account.

Layers/free-energy minima are clustered based on their distance to the
electrode.
"""


# Standard libraries
import argparse
import os
import warnings
from copy import deepcopy

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
from scipy.cluster.hierarchy import dendrogram

# First-party libraries
import lintf2_ether_ana_postproc as leap


def equalize_xticks(ax):
    """
    Equalize x-ticks so that plots can be better stacked together.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` for which to equalize the x
        ticks.
    """
    ax.xaxis.set_major_locator(MultipleLocator(0.25))


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
        "Plot the number of renewal events in each bin as function of the"
        " electrode surface charge."
    )
)
parser.add_argument(
    "--sol",
    type=str,
    required=True,
    choices=("g1", "g4", "peo63"),
    help="Solvent name.",
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
    + "_lintf2_"
    + args.sol
    + "_20-1_gra_qX_sc80_renewal_event_num"
    + analysis_suffix
    + "_cluster_pthresh_%.2f" % args.prob_thresh
)
if args.common_ylim:
    outfile += "_common_ylim.pdf"
else:
    outfile += ".pdf"

# Columns to read from the files that contain the free-energy extrema.
pkp_col = 1  # Column that contains the peak positions in nm.
pkm_col = 3  # Column that contains the peak prominences in kT.
cols_fe = (pkp_col, pkm_col)  # Peak positions [nm].
pkp_col_ix = cols_fe.index(pkp_col)
pkm_col_ix = cols_fe.index(pkm_col)

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
sys_pat = "lintf2_" + args.sol + "_20-1_gra_q[0-9]*_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
sys_pat = os.path.join("q[0-9]*", sys_pat)
Sims = leap.simulation.get_sims(sys_pat, set_pat, "walls", sort_key="surfq")


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
if not np.all(Sims.O_per_chain == Sims.O_per_chain[0]):
    raise ValueError(
        "The number of ether oxygens per chain is not the same in all surface"
        " simulations"
    )
valid_bulk_sim = n_eo == Sims.O_per_chain[0]
rt_bulk = rt_bulk[valid_bulk_sim]
if len(rt_bulk) != 1:
    raise ValueError(
        "The number of bulk renewal times ({}) is not one".format(len(rt_bulk))
    )
rt_bulk = rt_bulk[0]
del valid_bulk_sim, n_eo

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
# 8) tau_3^layer = 7) * tau_3^bulk.
ydata = [
    [[[] for sim in Sims.sims] for pkp_type in pk_pos_types]
    for dat_ix in range(n_data)
]
n_events_per_refcmp_bulk = np.zeros(Sims.n_sims, dtype=np.float64)
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
    n_events_per_refcmp_bulk_sim = n_events_bulk_sim / n_refcmps_bulk_sim
    n_events_per_refcmp_bulk[sim_ix] = n_events_per_refcmp_bulk_sim

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
        n_events_per_refcmp_layer = n_events_layer / n_refcmps_layer
        ydata[pkp_col_ix][pkt_ix][sim_ix] = minima[pkp_col_ix][pkt_ix][sim_ix]
        ydata[1][pkt_ix][sim_ix] = n_refcmps_layer
        ydata[2][pkt_ix][sim_ix] = n_refcmps_layer / n_refcmps_walls_tot
        ydata[3][pkt_ix][sim_ix] = n_events_layer
        ydata[4][pkt_ix][sim_ix] = n_events_per_refcmp_layer
        ydata[5][pkt_ix][sim_ix] = (
            n_events_per_refcmp_layer / n_events_per_refcmp_bulk_sim
        )
        ydata[6][pkt_ix][sim_ix] = (
            n_events_per_refcmp_bulk_sim / n_events_per_refcmp_layer
        )
        ydata[7][pkt_ix][sim_ix] = (
            rt_bulk * n_events_per_refcmp_bulk_sim / n_events_per_refcmp_layer
        )

if not np.all(
    np.isclose(
        n_events_per_refcmp_bulk,
        n_events_per_refcmp_bulk[0],
        atol=0,
        rtol=1e-9,
    )
):
    raise ValueError(
        "not all `n_events_per_refcmp_bulk` ({}) =="
        " `n_events_per_refcmp_bulk[0]`"
        " ({})".format(n_events_per_refcmp_bulk, n_events_per_refcmp_bulk[0])
    )
n_events_per_refcmp_bulk = n_events_per_refcmp_bulk[0]


print("Discarding free-energy minima whose prominence is too low...")
prom_min = leap.misc.e_kin(args.prob_thresh)
prominences = deepcopy(minima[pkm_col_ix])
n_pks_max = [0 for prom_pkt in prominences]
for col_ix, yd_col in enumerate(ydata):
    for pkt_ix, yd_pkt in enumerate(yd_col):
        for sim_ix, yd_sim in enumerate(yd_pkt):
            valid = prominences[pkt_ix][sim_ix] >= prom_min
            ydata[col_ix][pkt_ix][sim_ix] = yd_sim[valid]
            n_pks_max[pkt_ix] = max(n_pks_max[pkt_ix], np.count_nonzero(valid))
del minima, prominences


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
        Sims.surfqs[sim_ix]
        for sim_ix, n_pks_sim in enumerate(n_pks_per_sim_pkt)
        for _ in range(n_pks_sim)
    ]
    xdata[pkt_ix] = np.array(xdata[pkt_ix])


print("Creating plot(s)...")
n_clstrs_plot = np.max(n_clstrs)
if args.n_clusters is not None:
    n_clstrs_plot = min(n_clstrs_plot, args.n_clusters)
if n_clstrs_plot <= 0:
    raise ValueError("`n_clstrs_plot` ({}) <= 0".format(n_clstrs_plot))

legend_title = (
    r"$n_{EO} = %d$, " % Sims.O_per_chain[0]
    + r"$r = %.2f$" % Sims.Li_O_ratios[0]
    + "\n"
    + r"$F_{"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
    + r"}$ Minima"
)
legend_title_suffix = " Positions / nm"
n_legend_handles_comb = sum(min(n_cl, n_clstrs_plot) for n_cl in n_clstrs)
n_legend_cols_comb = min(3, 1 + n_legend_handles_comb // (2 + 1))
legend_locs_sep = tuple("best" for col_ix in range(n_data))
legend_locs_comb = tuple("best" for col_ix in range(n_data))
if len(legend_locs_sep) != n_data:
    raise ValueError(
        "`len(legend_locs_sep)` ({}) != `n_data`"
        " ({})".format(len(legend_locs_sep), n_data)
    )
if len(legend_locs_comb) != n_data:
    raise ValueError(
        "`len(legend_locs_comb)` ({}) != `n_data`"
        " ({})".format(len(legend_locs_comb), n_data)
    )

xlabel_sep = r"Surface Charge $\sigma_s$ / $e$/nm$^2$"
xlabel_comb = r"Surface Charge $|\sigma_s|$ / $e$/nm$^2$"
xlim = np.array([-0.1, 1.1])
ylabels = (
    "Distance to Electrode / nm",
    r"$N_{%s}^{layer}$" % leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1],
    (
        r"$N_{%s}^{layer} / " % leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
        + r"N_{%s}^{tot}$" % leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
    ),
    r"$N_{events}^{layer}$",
    (
        r"$N_{events}^{layer} / N_{%s}^{layer}$"
        % leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
    ),
    (
        r"$(N_{evt}^{lyr} / N_{%s}^{lyr}) / "
        % leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
        + r"(N_{evt}^{blk} / N_{%s}^{blk})$"
        % leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
    ),
    (
        r"$(N_{evt}^{blk} / N_{%s}^{blk}) / "
        % leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
        + r"(N_{evt}^{lyr} / N_{%s}^{lyr})$"
        % leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
    ),
    r"$\tau_3^{layer}$ / ns",
)
if len(ylabels) != n_data:
    raise ValueError(
        "`len(ylabels)` ({}) != `n_data` ({})".format(len(ylabels), n_data)
    )

logy = (  # Whether to use log scale for the y-axis.
    False,  # Free-energy minima positions (distance to electrode).
    False,  # No. of reference compounds per layer (N_cmp^layer).
    False,  # N_cmp^layer / N_cmp^tot.
    True,  # No. of renewal events per layer (N_events^layer).
    True,  # N_events^layer / N_cmp^layer.
    True,  # (N_events^layer / N_cmp^layer) / (N_events^bulk / N_cmp^bulk).
    True,  # (N_events^bulk / N_cmp^bulk) / (N_events^layer / N_cmp^layer).
    True,  # tau_3^layer.
)
if len(logy) != n_data:
    raise ValueError(
        "`len(logy)` ({}) != `n_data` ({})".format(len(logy), n_data)
    )

if args.common_ylim:
    ylims = [
        (0, 3.6),  # Free-energy minima positions.
        (-2, 34),  # N_cmp^layer.
        (-0.01, 0.27),  # N_cmp^layer / N_cmp^tot.
        (6e-1, 7e4),  # (8e-1, 6e4),  # N_events^layer.
        (5e-2, 4e3),  # (1e-1, 2e3),  # N_events^layer / N_cmp^layer.
        (2e-2, 2e1),  # (3e-2, 1e1),  # layer / bulk.
        (7e-2, 4e1),  # (1e-1, 3e1),  # bulk / layer.
        (3e-1, 9e3),  # (4e-1, 6e3),  # tau_3^layer.
    ]
else:
    ylims = tuple((None, None) for col_ix in range(n_data))
if len(ylims) != n_data:
    raise ValueError(
        "`len(ylims)` ({}) != `n_data` ({})".format(len(ylims), n_data)
    )

linestyles_comb = ("solid", "dashed")
if len(linestyles_comb) != len(pk_pos_types):
    raise ValueError(
        "`len(linestyles_comb)` ({}) != `len(pk_pos_types)`"
        " ({})".format(len(linestyles_comb), len(pk_pos_types))
    )
markers = ("<", ">")
if len(markers) != len(pk_pos_types):
    raise ValueError(
        "`len(markers)` ({}) != `len(pk_pos_types)`"
        " ({})".format(len(markers), len(pk_pos_types))
    )

cmap = plt.get_cmap()
c_vals_sep = np.arange(n_clstrs_plot)
c_norm_sep = n_clstrs_plot - 1
c_vals_sep_normed = c_vals_sep / c_norm_sep
colors_sep = cmap(c_vals_sep_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for col_ix, yd_col in enumerate(ydata):
        # Peaks at left and right electrode combined in one plot.
        fig_comb, ax_comb = plt.subplots(clear=True)
        if ylabels[col_ix].startswith(r"$N_{events}^{layer} / N_{"):
            ax_comb.axhline(
                n_events_per_refcmp_bulk,
                linestyle="dotted",
                color="tab:red",
                label="Bulk",
            )
        elif ylabels[col_ix] == r"$\tau_3^{layer}$ / ns":
            ax_comb.axhline(
                rt_bulk,
                linestyle="dotted",
                color="tab:red",
                label="Bulk",
            )

        for pkt_ix, pkp_type in enumerate(pk_pos_types):
            yd_pkt = yd_col[pkt_ix]

            if pkp_type == "left":
                xdata_fac = 1
            elif pkp_type == "right":
                xdata_fac = -1
            else:
                raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))

            # Peaks at left and right electrode in separate plots.
            fig_sep, ax_sep = plt.subplots(clear=True)
            ax_sep.set_prop_cycle(color=colors_sep)
            if ylabels[col_ix].startswith(r"$N_{events}^{layer} / N_{"):
                ax_sep.axhline(
                    n_events_per_refcmp_bulk,
                    linestyle="dashed",
                    color="tab:red",
                    label="Bulk",
                )
            elif ylabels[col_ix] == r"$\tau_3^{layer}$ / ns":
                ax_sep.axhline(
                    rt_bulk,
                    linestyle="dashed",
                    color="tab:red",
                    label="Bulk",
                )

            c_vals_comb = c_vals_sep + 0.5 * pkt_ix
            c_norm_comb = c_norm_sep + 0.5 * (len(pk_pos_types) - 1)
            c_vals_comb_normed = c_vals_comb / c_norm_comb
            colors_comb = cmap(c_vals_comb_normed)
            ax_comb.set_prop_cycle(color=colors_comb)

            for cix_pkt in clstr_ix_unq[pkt_ix]:
                if cix_pkt >= n_clstrs_plot:
                    break

                valid = clstr_ix[pkt_ix] == cix_pkt
                if not np.any(valid):
                    raise ValueError(
                        "No valid peaks for peak type '{}' and cluster index"
                        " {}".format(pkp_type, cix_pkt)
                    )

                if pkp_type == "left":
                    label_sep = r"$<%.2f$" % clstr_bounds[pkt_ix][cix_pkt]
                    label_comb = r"$+, " + label_sep[1:]
                elif pkp_type == "right":
                    label_sep = r"$<%.2f$" % clstr_bounds[pkt_ix][cix_pkt]
                    label_comb = r"$-, " + label_sep[1:]
                else:
                    raise ValueError(
                        "Unknown `pkp_type`: '{}'".format(pkp_type)
                    )
                ax_sep.plot(
                    xdata_fac * xdata[pkt_ix][valid],
                    yd_pkt[valid],
                    linestyle="solid",
                    marker=markers[pkt_ix],
                    label=label_sep,
                )
                ax_comb.plot(
                    xdata[pkt_ix][valid],
                    yd_pkt[valid],
                    linestyle=linestyles_comb[pkt_ix],
                    marker=markers[pkt_ix],
                    label=label_comb,
                )

            if logy[col_ix]:
                ax_sep.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax_sep.set(
                xlabel=xlabel_sep,
                ylabel=ylabels[col_ix],
                xlim=xdata_fac * xlim,
                ylim=ylims[col_ix],
            )
            equalize_xticks(ax_sep)
            if not logy[col_ix]:
                equalize_yticks(ax_sep)
            legend_sep = ax_sep.legend(
                title=legend_title + legend_title_suffix,
                ncol=2,
                loc=legend_locs_sep[col_ix],
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend_sep.get_title().set_multialignment("center")
            pdf.savefig(fig_sep)
            plt.close(fig_sep)

        if logy[col_ix]:
            ax_comb.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax_comb.set(
            xlabel=xlabel_comb,
            ylabel=ylabels[col_ix],
            xlim=xlim,
            ylim=ylims[col_ix],
        )
        equalize_xticks(ax_comb)
        if not logy[col_ix]:
            equalize_yticks(ax_comb)
        legend_comb = ax_comb.legend(
            title=legend_title + legend_title_suffix,
            ncol=n_legend_cols_comb,
            loc=legend_locs_comb[col_ix],
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend_comb.get_title().set_multialignment("center")
        pdf.savefig(fig_comb)
        plt.close(fig_comb)

    # Plot clustering results.
    for pkt_ix, pkp_type in enumerate(pk_pos_types):
        if pkp_type == "left":
            xdata_fac = 1
            legend_title_clstrng = r"$+|\sigma_s|$, " + legend_title
        elif pkp_type == "right":
            xdata_fac = -1
            legend_title_clstrng = r"$-|\sigma_s|$, " + legend_title
        else:
            raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))
        cmap_norm = plt.Normalize(vmin=0, vmax=np.max(n_clstrs) - 1)

        # Dendrogram.
        fig, ax = plt.subplots(clear=True)
        dendrogram(
            linkage_matrices[pkt_ix],
            ax=ax,
            distance_sort="ascending",
            color_threshold=clstr_dist_thresh[pkt_ix],
        )
        ax.axhline(
            clstr_dist_thresh[pkt_ix],
            color="tab:gray",
            linestyle="dashed",
            label=r"Threshold $%.2f$ nm" % clstr_dist_thresh[pkt_ix],
        )
        ax.set(
            xlabel="Peak Number",
            ylabel="Peak Distance / nm",
            ylim=(0, ylims[pkp_col_ix][-1]),
        )
        legend = ax.legend(
            title=legend_title_clstrng,
            loc="best",
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

        # Scatter plot: Peak Positions vs. Peak Positions.
        fig, ax = plt.subplots(clear=True)
        for cix_pkt in clstr_ix_unq[pkt_ix]:
            if cix_pkt >= n_clstrs_plot:
                break
            valid = clstr_ix[pkt_ix] == cix_pkt
            if not np.any(valid):
                raise ValueError(
                    "No valid peaks for peak type '{}' and cluster index"
                    " {}".format(pkp_type, cix_pkt)
                )
            ax.scatter(
                ydata[pkp_col_ix][pkt_ix][valid],
                ydata[pkp_col_ix][pkt_ix][valid],
                color=cmap(cmap_norm(cix_pkt)),
                marker=markers[pkt_ix],
                label="$<%.2f$" % clstr_bounds[pkt_ix][cix_pkt],
            )
        ax.set(
            xlabel=ylabels[pkp_col_ix],
            ylabel=ylabels[pkp_col_ix],
            xlim=ylims[pkp_col_ix],
            ylim=ylims[pkp_col_ix],
        )
        legend = ax.legend(
            title=legend_title_clstrng + legend_title_suffix,
            ncol=2,
            loc="upper left",
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

        # Scatter plot: Peak Positions vs. `xdata`.
        fig, ax = plt.subplots(clear=True)
        for cix_pkt in clstr_ix_unq[pkt_ix]:
            if cix_pkt >= n_clstrs_plot:
                break
            valid = clstr_ix[pkt_ix] == cix_pkt
            if not np.any(valid):
                raise ValueError(
                    "No valid peaks for peak type '{}' and cluster index"
                    " {}".format(pkp_type, cix_pkt)
                )
            ax.scatter(
                xdata_fac * xdata[pkt_ix][valid],
                ydata[pkp_col_ix][pkt_ix][valid],
                color=cmap(cmap_norm(cix_pkt)),
                marker=markers[pkt_ix],
                label="$<%.2f$" % clstr_bounds[pkt_ix][cix_pkt],
            )
        ax.set(
            xlabel=xlabel_sep,
            ylabel=ylabels[pkp_col_ix],
            xlim=xdata_fac * xlim,
            ylim=ylims[pkp_col_ix],
        )
        equalize_xticks(ax)
        equalize_yticks(ax)
        legend = ax.legend(
            title=legend_title + legend_title_suffix,
            ncol=2,
            loc=legend_locs_sep[pkp_col_ix],
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

print("Created {}".format(outfile))
print("Done")
