#!/usr/bin/env python3


r"""
Calculate and plot the transition rate out of a free-energy minimum from
the free-energy barriers that surround the minimum for a given compound
as function of the salt concentration.

The free-energy minima are clustered based on their distance to the
electrodes.

The average transition rate :math:`\Gamma_i` out of minimum :math:`i` is
calculated as the average of the transition rates to the left and right:

.. math::

    \Gamma_i =
    \Gamma_{i,i-1} + \Gamma_{i,i+1} \proto
    e^{-\beta \Delta F_{i,i-1}} + e^{-\beta \Delta F_{i,i+1}}

The inverse of the average transition rate should be directly
proportional to the residence time of the given compound in the
free-energy minimum.
"""


# Standard libraries
import argparse
import warnings
from copy import deepcopy

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
from scipy import constants
from scipy.cluster.hierarchy import dendrogram

# First-party libraries
import lintf2_ether_ana_postproc as leap


def legend_title(surfq_sign, direction=None):
    r"""
    Create a legend title string.

    Parameters
    ----------
    surfq_sign : {"+", "-", r"\pm"}
        The sign of the surface charge.
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
    if surfq_sign not in ("+", "-", r"\pm"):
        raise ValueError("Unknown `surfq_sign`: '{}'".format(surfq_sign))
    if direction not in (None, "l2r", "r2l"):
        raise ValueError("Unknown `direction`: '{}'".format(direction))

    title = (
        r"$\sigma_s = "
        + surfq_sign
        + r" %.2f$ $e$/nm$^2$, " % Sims.surfqs[0]
        + r"$n_{EO} = %d$" % Sims.O_per_chain[0]
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
        r"$F_{" + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp] + r"}$ Minima"
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
    ax.xaxis.set_major_locator(MultipleLocator(0.1))


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
        "Calculate and plot the transition rate out of a free-energy minimum"
        " from the free-energy barriers that surround the minimum for a given"
        " compound as function of the salt concentration."
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
    "--surfq",
    type=str,
    required=True,
    choices=("q0", "q1"),
    help="Surface charge in e/nm^2.",
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
        "Only consider free-energy minima whose prominence is at least such"
        " high that only 100*PROB_THRESH percent of the particles have a"
        " higher 1-dimensional kinetic energy.  Default: %(default)s"
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

temp = 423
settings = "pr_nvt%d_nh" % temp  # Simulation settings.
outfile = (  # Output file name.
    settings
    + "_lintf2_"
    + args.sol
    + "_r_gra_"
    + args.surfq
    + "_sc80_free-energy_trans_rate_"
    + args.cmp
    + "_cluster_pthresh_%.2f" % args.prob_thresh
)
if args.common_ylim:
    outfile += "_common_ylim.pdf"
else:
    outfile += ".pdf"

# Columns to read from the input file(s).
pkp_col = 1  # Column that contains the peak positions in nm.
pkh_col = 2  # Column that contains the peak heights in kT.
pkm_col = 3  # Column that contains the peak prominences in kT.
cols = (pkp_col, pkh_col, pkm_col)
pkp_col_ix = cols.index(pkp_col)
pkh_col_ix = cols.index(pkh_col)
pkm_col_ix = cols.index(pkm_col)

# The method to use for calculating the distance between clusters.  See
# `scipy.cluster.hierarchy.linkage`.
clstr_dist_method = "single"

# beta = k * T
beta = constants.k * temp
beta = 1  # Because the free energy is already given in units of kT.


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_" + args.sol + "_[0-9]*-[0-9]*_gra_" + args.surfq + "_sc80"
excl_pat = "lintf2_[A-z]*[0-9]*_80-1_gra_q1_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, args.surfq, excl_pat, sort_key="Li_O_ratio"
)


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
barriers_l2r, barriers_r2l = leap.misc.free_energy_barriers(
    minima, maxima, pkp_col_ix, pkh_col_ix
)


print("Calculating transition rates...")
# The "columns" (data sets) are:
#   0 Free-energy minima positions (the basis for clustering).
#   1 Free-energy minima heights.
#   2 Free-energy maxima positions.
#   3 Free-energy maxima heights.
#   4 Free-energy barrier heights, left to right (away from electrode).
#   5 Free-energy barrier heights, right to left (toward electrode).
#   6 Transition rates, left to right.
#   7 Transition rates, right to left.
#   8 Transition rates, mean.
#   9 Inverse transition rates, left to right.
#  10 Inverse transition rates, right to left.
#  11 Inverse transition rates, mean.
if pkp_col_ix != 0:
    raise ValueError("`pkp_col_ix` ({}) != 0".format(pkp_col_ix))
if pkh_col_ix != 1:
    raise ValueError("`pkp_col_ix` ({}) != 1".format(pkh_col_ix))
ylabels = (
    "Minima Positions / nm",  # 0
    r"Free Energy Minima / $k_B T$",  # 1
    "Maxima Positions / nm",  # 2
    r"Free Energy Maxima / $k_B T$",  # 3
)
ylabels += 2 * (
    r"Barrier Height $\Delta F_{"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
    + r"}$ / $k_B T$",  # 4, 5
)
ylabels += 2 * ("Transition Rate / a.u.",)  # 6, 7
ylabels += ("Mean Trans. Rate / a.u.",)  # 8
ylabels += 2 * ("Inv. Trans. Rate / a.u.",)  # 9, 10
ylabels += ("Inv. Mean Trans. Rate / a.u.",)  # 11
ydata = [
    [
        [None for minima_pkp_sim in minima_pkp_pkt]
        for minima_pkp_pkt in minima[pkp_col_ix]
    ]
    for ylabel in ylabels
]
for pkt_ix, _minima_pkp_pkt in enumerate(minima[pkp_col_ix]):
    for sim_ix, _minima_pkp_sim in enumerate(_minima_pkp_pkt):
        bars_l2r_sim = barriers_l2r[pkh_col_ix][pkt_ix][sim_ix]
        bars_r2l_sim = barriers_r2l[pkh_col_ix][pkt_ix][sim_ix]
        rates_l2r = np.exp(-beta * bars_l2r_sim)
        rates_r2l = np.exp(-beta * bars_r2l_sim)
        rates_mean = np.exp(-beta * bars_l2r_sim[1:])
        rates_mean += np.exp(-beta * bars_r2l_sim[:-1])
        rates_mean /= 2
        rates_mean = np.insert(rates_mean, 0, np.exp(-beta * bars_l2r_sim[0]))

        # In the first layer/free-energy minimum at the electrode, the
        # free-energy barrier for transitions toward (i.e. inside) the
        # electrode is infinity and the corresponding transition rate is
        # accordingly zero.
        bars_r2l_sim = np.insert(bars_r2l_sim, 0, np.inf)
        # Discard the last free-energy barrier.  This is the "barrier"
        # that has to be overcome when entering the layering region
        # coming from the bulk region.
        bars_r2l_sim = bars_r2l_sim[:-1]
        rates_r2l = np.insert(rates_r2l, 0, 0)
        rates_r2l = rates_r2l[:-1]

        if np.any(rates_l2r < 0) or np.any(rates_l2r > 1):
            print()
            print("rates_l2r =", rates_l2r)
            raise ValueError("np.any(rates_l2r < 0) or np.any(rates_l2r > 1)")
        if np.any(rates_r2l < 0) or np.any(rates_r2l > 1):
            print()
            print("rates_r2l =", rates_r2l)
            raise ValueError("np.any(rates_r2l < 0) or np.any(rates_r2l > 1)")
        if np.any(rates_mean < 0) or np.any(rates_mean > 1):
            print()
            print("rates_mean =", rates_mean)
            raise ValueError(
                "np.any(rates_mean < 0) or np.any(rates_mean > 1)"
            )

        ydata[pkp_col_ix][pkt_ix][sim_ix] = minima[pkp_col_ix][pkt_ix][sim_ix]
        ydata[pkh_col_ix][pkt_ix][sim_ix] = minima[pkh_col_ix][pkt_ix][sim_ix]
        ydata[2][pkt_ix][sim_ix] = maxima[pkp_col_ix][pkt_ix][sim_ix]
        ydata[3][pkt_ix][sim_ix] = maxima[pkh_col_ix][pkt_ix][sim_ix]
        ydata[4][pkt_ix][sim_ix] = bars_l2r_sim
        ydata[5][pkt_ix][sim_ix] = bars_r2l_sim
        ydata[6][pkt_ix][sim_ix] = rates_l2r
        ydata[7][pkt_ix][sim_ix] = rates_r2l
        ydata[8][pkt_ix][sim_ix] = rates_mean
        ydata[9][pkt_ix][sim_ix] = np.divide(1, rates_l2r)
        ydata[10][pkt_ix][sim_ix] = np.divide(1, rates_r2l)
        ydata[11][pkt_ix][sim_ix] = np.divide(1, rates_mean)
del maxima, n_minima_max, n_maxima_max, barriers_l2r, barriers_r2l


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
Elctrd = leap.simulation.Electrode()
bulk_start = Elctrd.BULK_START / 10  # A -> nm

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
        Sims.Li_O_ratios[sim_ix]
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

direction = 4 * (None,)
direction += ("l2r", "r2l")
direction += 2 * ("l2r", "r2l", None)
if len(direction) != len(ylabels):
    raise ValueError(
        "`len(direction)` ({}) != `len(ylabels)`"
        " ({})".format(len(direction), len(ylabels))
    )
legend_title_suffix = " Positions / nm"
n_legend_handles_comb = sum(min(n_cl, n_clstrs_plot) for n_cl in n_clstrs)
n_legend_cols_comb = min(3, 1 + n_legend_handles_comb // (2 + 1))
legend_locs_sep = tuple("best" for ylabel in ylabels)
legend_locs_comb = tuple("best" for ylabel in ylabels)
if len(legend_locs_sep) != len(ylabels):
    raise ValueError(
        "`len(legend_locs_sep)` ({}) != `len(ylabels)`"
        " ({})".format(len(legend_locs_sep), len(ylabels))
    )
if len(legend_locs_comb) != len(ylabels):
    raise ValueError(
        "`len(legend_locs_comb)` ({}) != `len(ylabels)`"
        " ({})".format(len(legend_locs_comb), len(ylabels))
    )

xlabel = r"Li-to-EO Ratio $r$"
xlim = (0, 0.4 + 0.0125)
if args.common_ylim:
    if args.cmp == "Li":
        ylims = [
            (0, 3.6),  # 0 Free-energy minima positions.
            (-6, 4.5),  # 1 Free-energy minima heights.
            (0, 3.6),  # 2 Free-energy maxima positions.
            (-0.5, 7.5),  # 3 Free-energy maxima heights.
            (0, 11),  # 4 Free-energy barrier heights, l2r.
            (0, 11),  # 5 Free-energy barrier heights, r2l.
            (-0.05, 1.05),  # 6 Transition rates, l2r.
            (-0.05, 1.05),  # 7 Transition rates, r2l.
            (-0.05, 1.05),  # 8 Transition rates, mean.
            (8e-1, 6e4),  # 9 Inverse transition rates, l2r.
            (8e-1, 6e3),  # (9e-1, 6e2),  # 10 Inverse transition rates, r2l.
            (8e-1, 6e4),  # 11 Inverse transition rates, mean.
        ]
    else:
        raise NotImplementedError(
            "Common ylim not implemented for compounds other than Li"
        )
else:
    ylims = tuple((None, None) for ylabel in ylabels)
if len(ylims) != len(ylabels):
    raise ValueError(
        "`len(ylims)` ({}) != len(ylabels)"
        " ({})".format(len(ylims), len(ylabels))
    )

# Whether to use log scale for the y-axis.
logy = (
    False,  # 0 Free-energy minima positions.
    False,  # 1 Free-energy minima heights.
    False,  # 2 Free-energy maxima positions.
    False,  # 3 Free-energy maxima heights.
    False,  # 4 Free-energy barrier heights, l2r.
    False,  # 5 Free-energy barrier heights, r2l.
    False,  # 6 Transition rates, l2r.
    False,  # 7 Transition rates, r2l.
    False,  # 8 Transition rates, mean.
    True,  # 9 Inverse transition rates, l2r.
    True,  # 10 Inverse transition rates, r2l.
    True,  # 11 Inverse transition rates, mean.
)

pk_pos_types = ("left", "right")  # Peak at left or right electrode.
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

                yd_is_finite = np.isfinite(yd_pkt[valid])
                if not np.any(yd_is_finite):
                    ax_sep.plot([], [])  # Increment color cycle.
                    ax_comb.plot([], [])  # Increment color cycle.
                    continue

                marker = markers[pkt_ix]
                if pkp_type == "left":
                    label_sep = r"$<%.2f$" % clstr_bounds[pkt_ix][cix_pkt]
                    label_comb = r"$+, " + label_sep[1:]
                    if direction[col_ix] == "l2r":
                        marker = ">"
                    elif direction[col_ix] == "r2l":
                        marker = "<"
                elif pkp_type == "right":
                    label_sep = r"$<%.2f$" % clstr_bounds[pkt_ix][cix_pkt]
                    label_comb = r"$-, " + label_sep[1:]
                    if direction[col_ix] == "l2r":
                        marker = "<"
                    elif direction[col_ix] == "r2l":
                        marker = ">"
                else:
                    raise ValueError(
                        "Unknown `pkp_type`: '{}'".format(pkp_type)
                    )
                ax_sep.plot(
                    xdata[pkt_ix][valid][yd_is_finite],
                    yd_pkt[valid][yd_is_finite],
                    linestyle="solid",
                    marker=marker,
                    label=label_sep,
                )
                ax_comb.plot(
                    xdata[pkt_ix][valid][yd_is_finite],
                    yd_pkt[valid][yd_is_finite],
                    linestyle=linestyles_comb[pkt_ix],
                    marker=marker,
                    label=label_comb,
                )

            if pkp_type == "left":
                legend_title_sep = legend_title(r"+", direction[col_ix])
            elif pkp_type == "right":
                legend_title_sep = legend_title(r"-", direction[col_ix])
            else:
                raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))
            if logy[col_ix]:
                ax_sep.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax_sep.set(
                xlabel=xlabel,
                ylabel=ylabels[col_ix],
                xlim=xlim,
                ylim=ylims[col_ix],
            )
            equalize_xticks(ax_sep)
            if not logy[col_ix]:
                equalize_yticks(ax_sep)
            legend_sep = ax_sep.legend(
                title=legend_title_sep + legend_title_suffix,
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
            xlabel=xlabel,
            ylabel=ylabels[col_ix],
            xlim=xlim,
            ylim=ylims[col_ix],
        )
        equalize_xticks(ax_comb)
        if not logy[col_ix]:
            equalize_yticks(ax_comb)
        legend_comb = ax_comb.legend(
            title=(
                legend_title(r"\pm", direction[col_ix]) + legend_title_suffix
            ),
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
            legend_title_clstrng = legend_title("+")
        elif pkp_type == "right":
            legend_title_clstrng = legend_title("-")
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
                xdata[pkt_ix][valid],
                ydata[pkp_col_ix][pkt_ix][valid],
                color=cmap(cmap_norm(cix_pkt)),
                marker=markers[pkt_ix],
                label="$<%.2f$" % clstr_bounds[pkt_ix][cix_pkt],
            )
        ax.set(
            xlabel=xlabel,
            ylabel=ylabels[pkp_col_ix],
            xlim=xlim,
            ylim=ylims[pkp_col_ix],
        )
        equalize_xticks(ax)
        equalize_yticks(ax)
        legend = ax.legend(
            title=legend_title_clstrng + legend_title_suffix,
            ncol=2,
            loc=legend_locs_sep[pkp_col_ix],
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

print("Created {}".format(outfile))
print("Done")
