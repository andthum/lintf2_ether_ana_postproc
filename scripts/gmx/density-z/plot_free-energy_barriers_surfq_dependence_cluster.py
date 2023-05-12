#!/usr/bin/env python3


"""
Plot the positions and heights of the free-energy barriers for a given
compound as function of the electrode surface charge.

The barrier positions are clustered based on their distance to the
electrodes.
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
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
from scipy.cluster.hierarchy import dendrogram

# First-party libraries
import lintf2_ether_ana_postproc as leap


def legend_title(direction=None):
    r"""
    Create a legend title string.

    Parameters
    ----------
    direction : {None, "l2r", "r2l"}, optional
        Direction of travel, "left to right" (l2r) or "right to left"
        (r2l).

    Returns
    -------
    title : str
        The legend title.

    Notes
    -----
    This function relies on global variables!
    """
    if direction not in (None, "l2r", "r2l"):
        raise ValueError("Unknown `direction`: '{}'".format(direction))

    title = (
        r"$n_{EO} = %d$, " % Sims.O_per_chain[0]
        + r"$r = %.2f$" % Sims.Li_O_ratios[0]
        + "\n"
    )

    if direction is not None:
        # Because peak positions are not given in absolute coordinates
        # but as distance to the electrode, peaks at the right electrode
        # behave as peaks at the left electrode.
        if direction == "l2r":
            drctn = "From Electrode"
        else:
            drctn = "To Electrode"
        title += drctn + "\n"

    title += (
        r"$F_{" + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp] + r"}$ Barrier"
    )
    return title


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
        "Plot the positions and heights of the free-energy barriers for a"
        " given compound as function of the electrode surface charge."
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
    choices=("Li", "NBT", "OBT", "OE"),
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--prob-thresh",
    type=float,
    required=False,
    default=0.5,
    help=(
        "Only consider barriers that are at least such high that only"
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
        "The maximum number of barrier clusters to consider.  'None' means all"
        " barrier clusters.  Default: %(default)s"
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
if args.prob_thresh < 0 or args.prob_thresh > 1:
    raise ValueError(
        "--prob-thresh ({}) must be between 0 and 1".format(args.prob_thresh)
    )
if args.n_clusters is not None and args.n_clusters <= 0:
    raise ValueError(
        "--n-cluster ({}) must be a positive integer".format(args.n_clusters)
    )

settings = "pr_nvt423_nh"  # Simulation settings.
outfile = (  # Output file name.
    settings
    + "_lintf2_"
    + args.sol
    + "_20-1_gra_qX_sc80_free_energy_barriers_"
    + args.cmp
    + "_cluster_pthresh_%.2f" % args.prob_thresh
)
if args.common_ylim:
    outfile += "_common_ylim.pdf"
else:
    outfile += ".pdf"

pkp_col = 1  # Column that contains the peak positions in nm.
pkh_col = 2  # Column that contains the peak heights in kT.
cols = (pkp_col, pkh_col)  # Columns to read from the input file(s).
pkp_col_ix = cols.index(pkp_col)
pkh_col_ix = cols.index(pkh_col)
ylabels = (
    "Distance to Electrode / nm",
    r"Barrier Height $\Delta F_{"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
    + r"}$ / $k_B T$",
)
if len(ylabels) != len(cols):
    raise ValueError(
        "`len(ylabels)` ({}) != `len(cols)`"
        " ({})`".format(len(ylabels), len(cols))
    )

# The method to use for calculating the distance between clusters.  See
# `scipy.cluster.hierarchy.linkage`.
clstr_dist_method = "single"


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_" + args.sol + "_20-1_gra_q[0-9]*_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
sys_pat = os.path.join("q[0-9]*", sys_pat)
Sims = leap.simulation.get_sims(sys_pat, set_pat, "walls", sort_key="surfq")


print("Reading data...")
minima, n_minima_max = leap.simulation.read_free_energy_extrema(
    Sims, args.cmp, peak_type="minima", cols=cols
)
maxima, n_maxima_max = leap.simulation.read_free_energy_extrema(
    Sims, args.cmp, peak_type="maxima", cols=cols
)
if np.any(n_minima_max != n_maxima_max):
    raise ValueError(
        "Any `n_minima_max` ({}) != `n_maxima_max`"
        " ({})".format(n_minima_max, n_maxima_max)
    )


print("Calculating free-energy barriers...")
barrier_thresh = leap.misc.e_kin(args.prob_thresh)
barriers_l2r, barriers_r2l = leap.misc.free_energy_barriers(
    minima, maxima, pkp_col_ix, pkh_col_ix, thresh=barrier_thresh
)
ydata = [barriers_l2r, barriers_r2l]
directions = ("l2r", "r2l")  # Left to right and right to left.
if len(directions) != len(ydata):
    raise ValueError(
        "`len(directions)` ({}) != `len(ydata)`"
        " ({})".format(len(directions), len(ydata))
    )


print("Clustering barrier positions...")
Elctrd = leap.simulation.Electrode()
bulk_start = Elctrd.BULK_START / 10  # A -> nm

clstr_ix = [None for yd_drctn in ydata]
clstr_ix_unq = [None for yd_drctn in ydata]
linkage_matrices = [None for yd_drctn in ydata]
n_clstrs = [None for yd_drctn in ydata]
n_pks_per_sim = [None for yd_drctn in ydata]
clstr_dist_thresh = [None for yd_drctn in ydata]
for drctn_ix, yd_drctn in enumerate(ydata):
    clstrng_results = leap.clstr.peak_pos(
        yd_drctn, pkp_col_ix, return_dist_thresh=True, method=clstr_dist_method
    )
    ydata[drctn_ix] = clstrng_results[0]
    clstr_ix[drctn_ix] = clstrng_results[1]
    linkage_matrices[drctn_ix] = clstrng_results[2]
    n_clstrs[drctn_ix] = clstrng_results[3]
    n_pks_per_sim[drctn_ix] = clstrng_results[4]
    clstr_dist_thresh[drctn_ix] = clstrng_results[5]
    clstr_ix_unq[drctn_ix] = [
        np.unique(clstr_ix_pkt) for clstr_ix_pkt in clstr_ix[drctn_ix]
    ]
del clstrng_results


# Sort clusters by ascending average peak position and get cluster
# boundaries.
clstr_bounds = [
    [None for clstr_ix_pkt in clstr_ix_drctn] for clstr_ix_drctn in clstr_ix
]
for drctn_ix, clstr_ix_drctn in enumerate(clstr_ix):
    for pkt_ix, clstr_ix_pkt in enumerate(clstr_ix_drctn):
        _dists, clstr_ix[drctn_ix][pkt_ix], bounds = leap.clstr.dists_succ(
            ydata[drctn_ix][pkp_col_ix][pkt_ix],
            clstr_ix_pkt,
            method=clstr_dist_method,
            return_ix=True,
            return_bounds=True,
        )
        clstr_bounds[drctn_ix][pkt_ix] = np.append(bounds, bulk_start)

xdata = [
    [None for n_pks_per_sim_pkt in n_pks_per_sim_drctn]
    for n_pks_per_sim_drctn in n_pks_per_sim
]
for drctn_ix, n_pks_per_sim_drctn in enumerate(n_pks_per_sim):
    for pkt_ix, n_pks_per_sim_pkt in enumerate(n_pks_per_sim_drctn):
        xdata[drctn_ix][pkt_ix] = [
            Sims.surfqs[sim_ix]
            for sim_ix, n_pks_sim in enumerate(n_pks_per_sim_pkt)
            for _ in range(n_pks_sim)
        ]
        xdata[drctn_ix][pkt_ix] = np.array(xdata[drctn_ix][pkt_ix])


print("Creating plot(s)...")
n_clstrs_plot = [None for n_clstrs_drctn in n_clstrs]
for drctn_ix, n_clstrs_drctn in enumerate(n_clstrs):
    n_clstrs_plot[drctn_ix] = np.max(n_clstrs_drctn)
    if args.n_clusters is not None:
        n_clstrs_plot[drctn_ix] = min(n_clstrs_plot[drctn_ix], args.n_clusters)
    if n_clstrs_plot[drctn_ix] <= 0:
        raise ValueError(
            "`n_clstrs_plot[{}]` ({}) <="
            " 0".format(drctn_ix, n_clstrs_plot[drctn_ix])
        )

legend_title_suffix = " Positions / nm"
xlabel = r"Surface Charge $\sigma_s$ / $e$/nm$^2$"
xlim = np.array([-0.1, 1.1])
if args.common_ylim:
    if args.cmp == "Li":
        ylims = [
            (0, 3.6),  # Barrier positions [nm]
            (0, 11),  # Barrier heights [kT]
        ]
    else:
        raise NotImplementedError(
            "Common ylim not implemented for compounds other than Li"
        )
else:
    ylims = tuple((None, None) for col in cols)
if len(ylims) != len(cols):
    raise ValueError(
        "`len(ylims)` ({}) != len(cols)" " ({})".format(len(ylims), len(cols))
    )

pk_pos_types = ("left", "right")  # Peak at left or right electrode.
markers = (
    (">", "<"),  # Peak at left electrode.
    ("<", ">"),  # Peak at right electrode.
)
shape = (len(pk_pos_types), len(directions))
if len(markers) != len(directions):
    raise ValueError(
        "`len(markers)` ({}) != `len(directions)`"
        " ({})".format(len(markers), len(directions))
    )

cmap = plt.get_cmap()
c_vals = np.arange(np.max(n_clstrs_plot))
c_norm = np.max(n_clstrs_plot) - 1
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for col_ix in range(len(cols)):
        for pkt_ix, pkp_type in enumerate(pk_pos_types):
            if pkp_type == "left":
                xdata_fac = 1
            elif pkp_type == "right":
                xdata_fac = -1
            else:
                raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))

            for drctn_ix, drctn in enumerate(directions):
                xd = xdata[drctn_ix][pkt_ix]
                yd = ydata[drctn_ix][col_ix][pkt_ix]

                fig, ax = plt.subplots(clear=True)
                ax.set_prop_cycle(color=colors)

                for cix_pkt in clstr_ix_unq[drctn_ix][pkt_ix]:
                    if cix_pkt >= n_clstrs_plot[drctn_ix]:
                        break
                    valid = clstr_ix[drctn_ix][pkt_ix] == cix_pkt
                    if not np.any(valid):
                        raise ValueError(
                            "Direction: {}\n"
                            "Peak-position type: {}\n"
                            "Cluster: {}\n"
                            "No valid peaks".format(drctn, pkp_type, cix_pkt)
                        )
                    ax.plot(
                        xdata_fac * xd[valid],
                        yd[valid],
                        marker=markers[pkt_ix][drctn_ix],
                        label=(
                            r"$<%.2f$"
                            % clstr_bounds[drctn_ix][pkt_ix][cix_pkt]
                        ),
                    )

                ax.set(
                    xlabel=xlabel,
                    ylabel=ylabels[col_ix],
                    xlim=xdata_fac * xlim,
                    ylim=ylims[col_ix],
                )
                equalize_xticks(ax)
                equalize_yticks(ax)
                legend = ax.legend(
                    title=legend_title(drctn) + legend_title_suffix,
                    ncol=2,
                    **mdtplt.LEGEND_KWARGS_XSMALL,
                )
                legend.get_title().set_multialignment("center")
                pdf.savefig()
                plt.close()

    # Plot clustering results.
    n_clstrs_max = [np.max(n_clstrs_drctn) for n_clstrs_drctn in n_clstrs]
    n_clstrs_max = np.max(n_clstrs_max)
    for pkt_ix, pkp_type in enumerate(pk_pos_types):
        if pkp_type == "left":
            xdata_fac = 1
            legend_title_prefix = r"$+|\sigma_s|$, "
        elif pkp_type == "right":
            xdata_fac = -1
            legend_title_prefix = r"$-|\sigma_s|$, "
        else:
            raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))
        cmap_norm = plt.Normalize(vmin=0, vmax=n_clstrs_max - 1)
        for drctn_ix, drctn in enumerate(directions):
            xd = xdata[drctn_ix][pkt_ix]
            yd = ydata[drctn_ix][pkp_col_ix][pkt_ix]

            # Dendrogram.
            fig, ax = plt.subplots(clear=True)
            dendrogram(
                linkage_matrices[drctn_ix][pkt_ix],
                ax=ax,
                distance_sort="ascending",
                color_threshold=clstr_dist_thresh[drctn_ix][pkt_ix],
            )
            ax.axhline(
                clstr_dist_thresh[drctn_ix][pkt_ix],
                color="tab:gray",
                linestyle="dashed",
                label=(
                    r"Threshold $%.2f$ nm"
                    % clstr_dist_thresh[drctn_ix][pkt_ix]
                ),
            )
            ax.set(
                xlabel="Peak Number",
                ylabel="Peak Distance / nm",
                ylim=(0, ylims[pkp_col_ix][-1]),
            )
            legend = ax.legend(
                title=legend_title_prefix + legend_title(drctn) + "s",
                loc="best",
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend.get_title().set_multialignment("center")
            pdf.savefig()
            plt.close()

            # Scatter plot: Peak Positions vs. Peak Positions.
            fig, ax = plt.subplots(clear=True)
            for cix_pkt in clstr_ix_unq[drctn_ix][pkt_ix]:
                if cix_pkt >= n_clstrs_plot[drctn_ix]:
                    break
                valid = clstr_ix[drctn_ix][pkt_ix] == cix_pkt
                if not np.any(valid):
                    raise ValueError(
                        "Direction: {}\n"
                        "Peak-position type: {}\n"
                        "Cluster: {}\n"
                        "No valid peaks".format(drctn, pkp_type, cix_pkt)
                    )
                ax.scatter(
                    yd[valid],
                    yd[valid],
                    color=cmap(cmap_norm(cix_pkt)),
                    marker=markers[pkt_ix][drctn_ix],
                    label="$<%.2f$" % clstr_bounds[drctn_ix][pkt_ix][cix_pkt],
                )
            ax.set(
                xlabel=ylabels[pkp_col_ix],
                ylabel=ylabels[pkp_col_ix],
                xlim=ylims[pkp_col_ix],
                ylim=ylims[pkp_col_ix],
            )
            legend = ax.legend(
                title=(
                    legend_title_prefix
                    + legend_title(drctn)
                    + legend_title_suffix
                ),
                ncol=2,
                loc="upper left",
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend.get_title().set_multialignment("center")
            pdf.savefig()
            plt.close()

            # Scatter plot: Peak Positions vs. `xdata`.
            fig, ax = plt.subplots(clear=True)
            for cix_pkt in clstr_ix_unq[drctn_ix][pkt_ix]:
                if cix_pkt >= n_clstrs_plot[drctn_ix]:
                    break
                valid = clstr_ix[drctn_ix][pkt_ix] == cix_pkt
                if not np.any(valid):
                    raise ValueError(
                        "Direction: {}\n"
                        "Peak-position type: {}\n"
                        "Cluster: {}\n"
                        "No valid peaks".format(drctn, pkp_type, cix_pkt)
                    )
                scatter = ax.scatter(
                    xdata_fac * xd[valid],
                    yd[valid],
                    color=cmap(cmap_norm(cix_pkt)),
                    marker=markers[pkt_ix][drctn_ix],
                    label="$<%.2f$" % clstr_bounds[drctn_ix][pkt_ix][cix_pkt],
                )
            ax.set(
                xlabel=xlabel,
                ylabel=ylabels[pkp_col_ix],
                xlim=xdata_fac * xlim,
                ylim=ylims[pkp_col_ix],
            )
            equalize_xticks(ax)
            equalize_yticks(ax)
            legend = ax.legend(
                title=legend_title(drctn) + legend_title_suffix,
                ncol=2,
                loc="best",
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend.get_title().set_multialignment("center")
            pdf.savefig()
            plt.close()

print("Created {}".format(outfile))
print("Done")
