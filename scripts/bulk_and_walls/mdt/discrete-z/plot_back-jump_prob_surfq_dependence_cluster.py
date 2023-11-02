#!/usr/bin/env python3


"""
Plot the probability of a given compound to jump back to its previous
layer as function of the electrode surface charge.

Only bins that correspond to actual layers (i.e. free-energy minima) at
the electrode interface are taken into account.

Layers/free-energy minima are clustered based on their distance to the
electrode.
"""


# Standard libraries
import argparse
import os
import warnings

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
        "Plot the probability of a given compound to jump back to its previous"
        " layer as function of the electrode surface charge."
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
        "Only consider the back-jump probability for layers/free-energy minima"
        " whose prominence is at least such high that only 100*PROB_THRESH"
        " percent of the particles have a higher 1-dimensional kinetic energy."
        "  Default: %(default)s"
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
    "--continuous",
    required=False,
    default=False,
    action="store_true",
    help="Use the 'continuous' (true) back-jump probability.",
)
parser.add_argument(
    "--common-ylim",
    required=False,
    default=False,
    action="store_true",
    help="Use common y limits for all plots.",
)
args = parser.parse_args()
if args.prob_thresh < 0 or args.prob_thresh > 1:
    raise ValueError(
        "--prob-thresh ({}) must be between 0 and 1".format(args.prob_thresh)
    )
if args.n_clusters is not None and args.n_clusters <= 0:
    raise ValueError(
        "--n-cluster ({}) must be a positive integer".format(args.n_clusters)
    )

if args.continuous:
    con = "_continuous"
else:
    con = ""

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "discrete-z"  # Analysis name.
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    settings
    + "_lintf2_"
    + args.sol
    + "_20-1_gra_qX_sc80_back_jump_probs_"
    + args.cmp
    + "_cluster_pthresh_%.2f" % args.prob_thresh
    + con
)
if args.common_ylim:
    outfile += "_common_ylim.pdf"
else:
    outfile += ".pdf"

# The method to use for calculating the distance between clusters.  See
# `scipy.cluster.hierarchy.linkage`.
clstr_dist_method = "single"

# Columns to read from the file containing the free-energy extrema
# (output file of `scripts/gmx/density-z/get_free-energy_extrema.py`).
ylabels = ("Distance to Electrode / nm",)
pkp_col = 1  # Column that contains the peak positions in nm.
cols_fe = (pkp_col,)  # Peak positions [nm].
pkp_col_ix = cols_fe.index(pkp_col)

# Lag times in trajectory steps for which to explicitly plot the
# back-jump probability as function of the chain length for all layers.
bj_probs_cluster = np.arange(1, 11, dtype=np.uint8)
ylabels += (r"State Index (Bin No.$ - 1$)",)
ylabels += tuple(
    [
        r"$p_{back}(\Delta t = %d \mathrm{ps})$" % (2 * frame)
        for frame in bj_probs_cluster
    ]
)
# Number of "columns" (i.e. data series that will be clustered).
n_cols = len(ylabels)


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_" + args.sol + "_20-1_gra_q[0-9]*_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
sys_pat = os.path.join("q[0-9]*", sys_pat)
Sims = leap.simulation.get_sims(sys_pat, set_pat, "walls", sort_key="surfq")


print("Reading data...")
# Read free-energy minima positions.
prom_min = leap.misc.e_kin(args.prob_thresh)
peak_pos, n_pks_max = leap.simulation.read_free_energy_extrema(
    Sims, args.cmp, peak_type="minima", cols=cols_fe, prom_min=prom_min
)

# Get filenames of the files containing the back-jump probabilities.
file_suffix = (
    analysis + "_" + args.cmp + "_back_jump_prob_discrete" + con + ".txt.gz"
)
infiles_bj_prob = leap.simulation.get_ana_files(
    Sims, analysis, tool, file_suffix
)
n_infiles_bj_prob = len(infiles_bj_prob)

# Get filenames of the files containing the bins used to calculate the
# back-jump probabilities.
file_suffix = analysis + "_" + args.cmp + "_bins.txt.gz"
infiles_bins = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
n_infiles_bins = len(infiles_bins)
if n_infiles_bins != n_infiles_bj_prob:
    raise ValueError(
        "`n_infiles_bins` ({}) != `n_infiles_bj_prob` ({})".format(
            n_infiles_bins, n_infiles_bj_prob
        )
    )

# Assign state indices to layers/free-energy minima.
Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm.
bulk_start = Elctrd.BULK_START / 10  # A -> nm.

pk_pos_types = ("left", "right")
ydata = [
    [[None for sim in Sims.sims] for pkp_type in pk_pos_types]
    for col_ix in range(n_cols)
]
for sim_ix, Sim in enumerate(Sims.sims):
    box_z = Sim.box[2] / 10  # A -> nm
    bulk_region = Sim.bulk_region / 10  # A -> nm

    # Read back-jump probabilities
    ret = leap.simulation.read_time_state_matrix(
        infiles_bj_prob[sim_ix],
        time_conv=2e-3,  # trajectory steps -> ns.
        amin=0,
        amax=1,
    )
    bj_probs_sim, times_sim, states_sim = ret
    bj_probs_sim = bj_probs_sim[bj_probs_cluster]
    data_sim = np.row_stack([states_sim, bj_probs_sim])
    del ret, bj_probs_sim, times_sim

    # Read bin edges.
    bins = np.loadtxt(infiles_bins[sim_ix])
    bins /= 10  # A -> nm.
    bin_edges_lower = bins[:-1]
    bin_edges_upper = bins[1:]
    bin_mids = bins[1:] - np.diff(bins) / 2
    # Select all bins for which back-jump probabilities are available.
    bin_edges_lower = bin_edges_lower[states_sim]
    bin_edges_upper = bin_edges_upper[states_sim]
    bin_mids = bin_mids[states_sim]
    bin_is_left = bin_mids <= (box_z / 2)
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

    for pkt_ix, pkp_type in enumerate(pk_pos_types):
        if pkp_type == "left":
            valid_bins = bin_is_left
            data_sim_valid = data_sim[:, valid_bins]
            # Convert absolute bin positions to distances to the
            # electrodes.
            bin_edges_valid = bin_edges_lower[valid_bins]
            bin_edges_valid -= elctrd_thk
        elif pkp_type == "right":
            valid_bins = ~bin_is_left
            data_sim_valid = data_sim[:, valid_bins]
            # Reverse the order of rows to sort bins as function of the
            # distance to the electrodes in ascending order.
            data_sim_valid = data_sim_valid[:, ::-1]
            # Convert absolute bin positions to distances to the
            # electrodes.
            bin_edges_valid = bin_edges_upper[valid_bins][::-1]
            bin_edges_valid += elctrd_thk
            bin_edges_valid -= box_z
            bin_edges_valid *= -1  # Ensure positive distance values.
        else:
            raise ValueError(
                "Unknown peak position type: '{}'".format(pkp_type)
            )
        tolerance = 1e-6
        if np.any(bin_edges_valid < -tolerance):
            raise ValueError(
                "Simulation: '{}'.\n"
                "Peak-position type: '{}'.\n"
                "At least one bin lies within the electrode.  This should not"
                " have happened.  Bin edges: {}.  Electrode:"
                " 0".format(Sim.path, pkp_type, bin_edges_valid)
            )

        # Assign bins to layers/free-energy minima.
        pk_pos = peak_pos[pkp_col_ix][pkt_ix][sim_ix]
        ix = np.searchsorted(pk_pos, bin_edges_valid)
        # Bins that are sorted after the last free-energy minimum lie
        # inside the bulk or near the opposite electrode and are
        # therefore discarded.
        layer_ix = ix[ix < len(pk_pos)]
        if not np.array_equal(layer_ix, np.arange(len(pk_pos))):
            raise ValueError(
                "Simulation: '{}'.\n"
                "Peak-position type: '{}'.\n"
                "Could not match each layer/free-energy minimum to exactly one"
                " bin.  Bin edges: {}.  Free-energy minima: {}.  Assignment:"
                " {}".format(
                    Sim.path, pkp_type, bin_edges_valid, pk_pos, layer_ix
                )
            )
        data_sim_valid = data_sim_valid[:, layer_ix]

        ydata[pkp_col_ix][pkt_ix][sim_ix] = pk_pos
        col_indices = np.arange(n_cols)
        col_indices = np.delete(col_indices, pkp_col_ix)
        for col_ix, dat_col_sim_valid in zip(col_indices, data_sim_valid):
            ydata[col_ix][pkt_ix][sim_ix] = dat_col_sim_valid


print("Clustering peak positions...")
(
    ydata,
    clstr_ix,
    linkage_matrices,
    n_clstrs,
    n_pks_per_sim,
    clstr_dist_thresh,
) = leap.clstr.peak_pos(ydata, pkp_col_ix, return_dist_thresh=True)
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
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
    + r"}$ Minima"
)
legend_title_suffix = " Positions / nm"
n_legend_handles_comb = sum(min(n_cl, n_clstrs_plot) for n_cl in n_clstrs)
n_legend_cols_comb = min(3, 1 + n_legend_handles_comb // (2 + 1))
legend_locs_sep = tuple("best" for col_ix in range(n_cols))
legend_locs_comb = tuple("best" for col_ix in range(n_cols))
if len(legend_locs_sep) != n_cols:
    raise ValueError(
        "`len(legend_locs_sep)` ({}) != `n_cols`"
        " ({})".format(len(legend_locs_sep), n_cols)
    )
if len(legend_locs_comb) != n_cols:
    raise ValueError(
        "`len(legend_locs_comb)` ({}) != `n_cols`"
        " ({})".format(len(legend_locs_comb), n_cols)
    )

xlabel_sep = r"Surface Charge $\sigma_s$ / $e$/nm$^2$"
xlabel_comb = r"Surface Charge $|\sigma_s|$ / $e$/nm$^2$"
xlim = np.array([-0.1, 1.1])
if args.common_ylim:
    raise NotImplementedError("--common-ylim is not implemented yet")
else:
    ylims = tuple((None, None) for col_ix in range(n_cols))

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

            ax_sep.set(
                xlabel=xlabel_sep,
                ylabel=ylabels[col_ix],
                xlim=xdata_fac * xlim,
                ylim=ylims[col_ix],
            )
            equalize_xticks(ax_sep)
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

        ax_comb.set(
            xlabel=xlabel_comb,
            ylabel=ylabels[col_ix],
            xlim=xlim,
            ylim=ylims[col_ix],
        )
        equalize_xticks(ax_comb)
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
        equalize_xticks(ax_comb)
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
