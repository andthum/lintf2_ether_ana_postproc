#!/usr/bin/env python3


"""
Plot a given component of the mean displacement and the displacement
variance in each bin at a constant diffusion time as function of the PEO
chain length.

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
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.cluster.hierarchy import dendrogram

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

    if args.cmp in ("NBT", "OBT", "OE"):
        legend_title = (
            "$" + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp] + "$" + ", "
        )
    else:
        legend_title = leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp] + ", "
    legend_title += (
        r"$\Delta t = %.2f$ ns" % args.time
        + "\n"
        + r"$\sigma_s = "
        + surfq_sign
        + r" %.2f$ $e$/nm$^2$, " % Sims.surfqs[0]
        + r"$r = %.2f$" % Sims.Li_O_ratios[0]
        + "\n"
        + r"$F_{"
        + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp_fe]
        + r"}$ Minima"
    )
    return legend_title


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
        "Plot a given component of the mean displacement and the displacement"
        " variance in each bin at a constant diffusion time as function of the"
        " PEO chain length."
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
    choices=("Li", "NTf2", "ether", "NBT", "OBT", "OE"),
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--msd-component",
    type=str,
    required=False,
    default="z",
    choices=("xy", "z"),
    help="The MSD component to use for the analysis.  Default: %(default)s",
)
parser.add_argument(
    "--time",
    type=float,
    required=False,
    default=1000,
    help=(
        "Diffusion time in ps for which to plot the displacements as function"
        " of the initial particle position.  If no data are present at the"
        " given diffusion time, the next nearest diffusion time for which data"
        " are present is used.  Default: %(default)s"
    ),
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
if args.prob_thresh < 0 or args.prob_thresh > 1:
    raise ValueError(
        "--prob-thresh ({}) must be between 0 and 1".format(args.prob_thresh)
    )
if args.n_clusters is not None and args.n_clusters <= 0:
    raise ValueError(
        "--n-cluster ({}) must be a positive integer".format(args.n_clusters)
    )

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "msd_layer"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
ana_path = os.path.join(analysis, analysis + analysis_suffix)
tool = "mdt"  # Analysis software.
outfile = (
    settings
    + "_lintf2_peoN_20-1_gra_"
    + args.surfq
    + "_sc80_"
    + args.cmp
    + "_displvar"
    + args.msd_component
    + "_cross_section_"
    + "%.0fps" % args.time
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

# The method to use for calculating the distance between clusters.  See
# `scipy.cluster.hierarchy.linkage`.
clstr_dist_method = "single"

# Number of decimal places to use for the assignment of bin edges to
# free-energy maxima.
decimals = 3

args.time /= 1e3  # ps -> ns.

if args.msd_component == "xy":
    dimensions = ("x", "y")
elif args.msd_component == "z":
    dimensions = ("z",)
else:
    raise ValueError(
        "Unknown --msd-component: '{}'".format(args.msd_component)
    )


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_gra_" + args.surfq + "_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, args.surfq, sort_key="O_per_chain"
)


print("Reading data...")
# Read positions of free-energy extrema.
cmp_fe = "Li"
minima, n_minima_max = leap.simulation.read_free_energy_extrema(
    Sims, cmp=cmp_fe, peak_type="minima", cols=cols_fe
)
maxima, n_maxima_max = leap.simulation.read_free_energy_extrema(
    Sims, cmp=cmp_fe, peak_type="maxima", cols=cols_fe
)
if np.any(n_minima_max != n_maxima_max):
    raise ValueError(
        "Any `n_minima_max` ({}) != `n_maxima_max`"
        " ({})".format(n_minima_max, n_maxima_max)
    )

Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm.
bulk_start = Elctrd.BULK_START / 10  # A -> nm.
pk_pos_types = ("left", "right")
n_data = 3
# Data:
# 1) Free-energy minima positions (distance to electrode).
# 2) Mean displacements.
# 3) Displacement variances.
ydata = [
    [[[] for sim in Sims.sims] for pkp_type in pk_pos_types]
    for dat_ix in range(n_data)
]
for sim_ix, Sim in enumerate(Sims.sims):
    # Read displacements at the given time from file.
    for dim_ix, dim in enumerate(dimensions):
        (
            times_dim,
            bins_dim,
            md_data,
            msd_data,
        ) = leap.simulation.read_displvar_single(Sim, args.cmp, dim)
        if dim_ix == 0:
            times, bins = times_dim, bins_dim
            time, tix = mdt.nph.find_nearest(
                times, args.time, return_index=True
            )
            if not np.isclose(time, args.time, atol=0):
                raise ValueError(
                    "The time given with --time ({} ns) is not contained in"
                    " the input files.  The closest time is {}"
                    " ns".format(args.time, time)
                )
            md_at_const_time = md_data[tix]
            msd_at_const_time = msd_data[tix]
        else:
            if bins_dim.shape != bins.shape:
                raise ValueError(
                    "The input files do not contain the same number of bins"
                )
            if not np.allclose(bins_dim, bins, atol=0):
                raise ValueError(
                    "The bin edges are not the same in all input files"
                )
            if times_dim.shape != times.shape:
                raise ValueError(
                    "The input files do not contain the same number of lag"
                    " times"
                )
            if not np.allclose(times_dim, times, atol=0):
                raise ValueError(
                    "The lag times are not the same in all input files"
                )
            time_dim, tix = mdt.nph.find_nearest(
                times_dim, args.time, return_index=True
            )
            if not np.isclose(time_dim, time, atol=0):
                raise ValueError(
                    "The chosen lag time is not the same in all input files"
                )
            md_at_const_time += md_data[tix]
            msd_at_const_time += msd_data[tix]
    del times, times_dim, bins_dim, md_data, msd_data

    box_z = Sim.box[2] / 10  # Angstrom -> nm.
    for pkt_ix, pkp_type in enumerate(pk_pos_types):
        # Convert absolute positions to distance to the electrode.
        if pkp_type == "left":
            bins_pkt = bins[1:] - elctrd_thk  # Upper bin edges.
            md_at_const_time_pkt = md_at_const_time
            msd_at_const_time_pkt = msd_at_const_time
        elif pkp_type == "right":
            bins_pkt = bins[:-1] + elctrd_thk  # Lower bin edges.
            bins_pkt -= box_z
            bins_pkt *= -1  # Ensure positive distance values.
            bins_pkt = bins_pkt[::-1]
            md_at_const_time_pkt = md_at_const_time[::-1]
            msd_at_const_time_pkt = msd_at_const_time[::-1]
        else:
            raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))

        # Assign bin edges to free-energy maxima.
        maxima_pkt = maxima[pkp_col_ix][pkt_ix][sim_ix]
        if not np.all(maxima_pkt[:-1] <= maxima_pkt[1:]):
            raise ValueError(
                "The positions of the free-energy maxima are not sorted"
            )
        if not np.all(bins_pkt[:-1] <= bins_pkt[1:]):
            raise ValueError("The bin edges are not sorted")
        maxima_pkt = np.round(maxima_pkt, decimals=decimals, out=maxima_pkt)
        bins_pkt = np.round(bins_pkt, decimals=decimals, out=bins_pkt)
        valid_bins = np.isin(bins_pkt, maxima_pkt)
        if np.count_nonzero(valid_bins) != len(maxima_pkt):
            raise ValueError(
                "The number of valid bins ({}) is not equal to the number of"
                " free-energy maxima"
                " ({})".format(np.count_nonzero(valid_bins), len(maxima_pkt))
            )
        first_valid = np.argmax(valid_bins) - 1
        if (
            args.cmp == cmp_fe
            and args.time != 0  # At t=0 all displacements are zero.
            and first_valid >= 0
            and np.isfinite(md_at_const_time_pkt[first_valid])
        ):
            raise ValueError("A populated bin was marked as invalid.")

        # Store data in list.
        ydata[pkp_col_ix][pkt_ix][sim_ix] = minima[pkp_col_ix][pkt_ix][sim_ix]
        ydata[1][pkt_ix][sim_ix] = md_at_const_time_pkt[valid_bins]
        ydata[2][pkt_ix][sim_ix] = msd_at_const_time_pkt[valid_bins]
del maxima


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
        Sims.O_per_chain[sim_ix]
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

xlabel = r"Ether Oxygens per Chain $n_{EO}$"
xlim = (1, 200)

# 1) Free-energy minima positions (distance to electrode).
ylabels = ["Distance to Electrode / nm"]
# 2) Mean displacements.
ylabel = r"$\langle \Delta "
if args.msd_component == "xy":
    ylabel += r"%s \rangle" % args.msd_component[0]
    ylabel += r"+ \langle \Delta %s" % args.msd_component[1]
elif args.msd_component == "z":
    ylabel += r"%s" % args.msd_component
else:
    raise ValueError(
        "Unknown --msd-component: '{}'".format(args.msd_component)
    )
ylabel += r" \rangle$ / nm"
ylabels.append(ylabel)
# 3) Displacement variances.
ylabel = r"Var$[\Delta "
if args.msd_component == "xy":
    ylabel += r"\mathbf{r}_{%s}" % args.msd_component
elif args.msd_component == "z":
    ylabel += r"%s" % args.msd_component
else:
    raise ValueError(
        "Unknown --msd-component: '{}'".format(args.msd_component)
    )
ylabel += r"]$ / nm$^2$"
ylabels.append(ylabel)
if len(ylabels) != n_data:
    raise ValueError(
        "`len(ylabels)` ({}) != `n_data` ({})".format(len(ylabels), n_data)
    )

logy = (  # Whether to use log scale for the y-axis.
    False,  # Free-energy minima positions (distance to electrode).
    False,  # Mean displacements.
    True,  # Displacement variances.
)
if len(logy) != n_data:
    raise ValueError(
        "`len(logy)` ({}) != `n_data` ({})".format(len(logy), n_data)
    )

if args.common_ylim:
    ylims = [(0, 3.6)]  # Free-energy minima positions.
    if args.cmp == "Li" and np.isclose(args.time, 0.1, rtol=0):
        ylims += [(-0.325, 0.325)]  # Mean displacements.
        if args.msd_component == "xy":
            ylims += [(1e-2, 2e0)]  # Displacement variances.
        elif args.msd_component == "z":
            ylims += [(4e-4, 5e-1)]  # Displacement variances.
        else:
            raise ValueError(
                "Unknown --msd-component: '{}'".format(args.msd_component)
            )
    elif args.cmp == "OE" and np.isclose(args.time, 0.1, rtol=0):
        if args.msd_component == "xy":
            ylims += [
                (-0.08, 0.08),  # Mean displacements.
                (1e-2, 4e0),  # Displacement variances.
            ]
        elif args.msd_component == "z":
            ylims += [
                (-0.8, 0.8),  # Mean displacements.
                (1e-2, 2e0),  # Displacement variances.
            ]
        else:
            raise ValueError(
                "Unknown --msd-component: '{}'".format(args.msd_component)
            )
    else:
        ylims += [
            (None, None),  # Mean displacements.
            (None, None),  # Displacement variances.
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
                    xdata[pkt_ix][valid],
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

            if pkp_type == "left":
                legend_title_sep = legend_title(r"+")
            elif pkp_type == "right":
                legend_title_sep = legend_title(r"-")
            else:
                raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))
            if col_ix == 0:
                # Remove compound and lag time from legend title.
                legend_title_sep = legend_title_sep.split("\n")[1:]
                legend_title_sep = "\n".join(legend_title_sep)
            ax_sep.set_xscale("log", base=10, subs=np.arange(2, 10))
            if logy[col_ix]:
                ax_sep.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax_sep.set(
                xlabel=xlabel,
                ylabel=ylabels[col_ix],
                xlim=xlim,
                ylim=ylims[col_ix],
            )
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

        legend_title_comb = legend_title(r"\pm")
        if col_ix == 0:
            # Remove compound and lag time from legend title.
            legend_title_comb = legend_title_comb.split("\n")[1:]
            legend_title_comb = "\n".join(legend_title_comb)
        ax_comb.set_xscale("log", base=10, subs=np.arange(2, 10))
        if logy[col_ix]:
            ax_comb.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax_comb.set(
            xlabel=xlabel,
            ylabel=ylabels[col_ix],
            xlim=xlim,
            ylim=ylims[col_ix],
        )
        if not logy[col_ix]:
            equalize_yticks(ax_comb)
        legend_comb = ax_comb.legend(
            title=legend_title_comb + legend_title_suffix,
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
        # Remove compound and lag time from legend title.
        legend_title_clstrng = legend_title_clstrng.split("\n")[1:]
        legend_title_clstrng = "\n".join(legend_title_clstrng)
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
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set(
            xlabel=xlabel,
            ylabel=ylabels[pkp_col_ix],
            xlim=xlim,
            ylim=ylims[pkp_col_ix],
        )
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
