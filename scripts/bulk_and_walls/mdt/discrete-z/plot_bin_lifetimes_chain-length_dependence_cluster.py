#!/usr/bin/env python3


"""
Plot the bin residence times of a given compound as function of the PEO
chain length.

Only bins that correspond to actual layers (i.e. free-energy minima) at
the electrode interface are taken into account.

Layers/free-energy minima are clustered based on their distance to the
electrode.
"""


# Standard libraries
import argparse
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
        "Plot the bin residence times of a given compound as function of the"
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
    choices=("Li",),
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--prob-thresh",
    type=float,
    required=False,
    default=0.5,
    help=(
        "Only consider the residence times of layers/free-energy minima whose"
        " prominence is at least such high that only 100*PROB_THRESH percent"
        " of the particles have a higher 1-dimensional kinetic energy."
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
    "--method",
    type=str,
    required=False,
    default="rate",
    choices=("count", "rate", "e", "area", "fit"),
    help=(
        "The method used to calculate the residence times.  Default:"
        " %(default)s"
    ),
)
parser.add_argument(
    "--continuous",
    required=False,
    default=False,
    action="store_true",
    help=(
        "Use the residence times calculated from the 'continuous' definition"
        " of the remain probability function.  Only relevant for methods"
        " 'area' and 'fit'."
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

if args.continuous and args.method in ("area", "fit"):
    con = "_continuous"
else:
    con = ""

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "discrete-z"  # Analysis name.
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    settings
    + "_lintf2_peoN_20-1_gra_"
    + args.surfq
    + "_sc80_bin_lifetimes_"
    + args.cmp
    + "_cluster_pthresh_%.2f" % args.prob_thresh
    + "_"
    + args.method
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
# Whether the columns contain a standard deviation or not.
col_fe_is_sd = (False,)

# Columns to read from the file containing the bin residence times.
ylabels += ("Residence Time / ns",)
cols_lt_bins = (
    1,  # Lower bin edges [A].
    2,  # Upper bin edges [A].
)
if args.method == "count":
    cols_lt_data = (
        7,  # 1st moment <tau_cnt> [ns].
        8,  # 2nd moment <tau_cnt^2> [ns^2].
    )
    col_lt_data_is_sd = (False, True)
elif args.method == "rate":
    cols_lt_data = (
        10,  # 1st moment <tau_k> [ns].
        # If <tau_cnt> = <tau_k> +/- 1%, use <tau_cnt^2> as estimate for
        # <tau_k^2>.
        8,  # 2nd moment <tau_cnt^2> [ns^2].
        7,  # 1st moment <tau_cnt> [ns].
        8,  # 2nd moment <tau_cnt^2> [ns^2].
    )
    ylabels += ("Res. Time (Count) / ns",)
    col_lt_data_is_sd = (False, True, False, True)
elif args.method == "e":
    cols_lt_data = (11,)  # 1st moment <tau_e> [ns].
    col_lt_data_is_sd = (False,)
elif args.method == "area":
    cols_lt_data = (
        12,  # 1st moment <tau_int> [ns].
        13,  # 2nd moment <tau_int^2> [ns^2].
    )
    col_lt_data_is_sd = (False, True)
elif args.method == "fit":
    ylabels += (
        r"Fit Parameter $\tau_0$ / ns",
        r"Fit Parameter $\beta$",
        r"Coeff. of Determ. $R^2$",
        r"Mean Squared Error / ns$^2$",
        "Fit Start / ns",
        "Fit End / ns",
    )
    cols_lt_data = (
        15,  # 1st moment <tau_exp> [ns].
        16,  # 2nd moment <tau_exp^2> [ns^2].
        18,  # Fit parameter tau0 [ns].
        19,  # Standard deviation of tau0 [ns].
        20,  # Fit parameter beta.
        21,  # Standard deviation of beta.
        22,  # Coefficient of determination of the fit (R^2 value).
        23,  # Mean squared error of the fit [ns^2].
        24,  # Start of fit region (inclusive) [ns].
        25,  # End of fit region (exclusive) [ns].
    )
    col_lt_data_is_sd = (
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    )
else:
    raise ValueError("Invalid --method ({})".format(args.method))
if len(col_lt_data_is_sd) != len(cols_lt_data):
    raise ValueError(
        "`len(col_lt_data_is_sd)` ({}) != `len(cols_lt_data)`"
        " ({})".format(len(col_lt_data_is_sd), len(cols_lt_data))
    )
cols_lt = cols_lt_bins + cols_lt_data

col_is_sd = col_fe_is_sd + col_lt_data_is_sd
# Whether the column has a standard deviation or not.
col_has_sd = tuple(np.roll(col_is_sd, shift=-1))
# Total number of columns/data sets.
n_cols_tot = len(col_is_sd)
# Number of columns that contain "real" data (no standard deviations).
n_cols_data = n_cols_tot - np.count_nonzero(col_is_sd)
if len(ylabels) != n_cols_data:
    raise ValueError(
        "`len(ylabels)` ({}) != `n_cols_data`"
        " ({})`".format(len(ylabels), n_cols_data)
    )


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

# Get filenames of the files containing the bin residence times.
file_suffix = analysis + "_" + args.cmp + "_lifetimes" + con + ".txt.gz"
infiles = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
n_infiles = len(infiles)

# Assign lifetimes to layers/free-energy minima.
Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm.
bulk_start = Elctrd.BULK_START / 10  # A -> nm.

pk_pos_types = ("left", "right")
ydata = [
    [[None for sim in Sims.sims] for pkp_type in pk_pos_types]
    for col_ix in range(n_cols_tot)
]
for sim_ix, Sim in enumerate(Sims.sims):
    box_z = Sim.box[2] / 10  # A -> nm
    bulk_region = Sim.bulk_region / 10  # A -> nm

    # Due to `unpack=True`, columns in the input file become rows in the
    # created array and rows become columns.
    data_sim = np.loadtxt(
        infiles[sim_ix], usecols=cols_lt, unpack=True, ndmin=2
    )
    data_sim[:2] /= 10  # Lower and upper bin edges, A -> nm.
    bin_mids = data_sim[0] + np.squeeze(np.diff(data_sim[:2], axis=0)) / 2
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
            bin_edges_valid = data_sim_valid[0]  # Lower bin edges.
            bin_edges_valid -= elctrd_thk
        elif pkp_type == "right":
            valid_bins = ~bin_is_left
            data_sim_valid = data_sim[:, valid_bins]
            # Reverse the order of rows to sort bins as function of the
            # distance to the electrodes in ascending order.
            data_sim_valid = data_sim_valid[:, ::-1]
            # Convert absolute bin positions to distances to the
            # electrodes.
            bin_edges_valid = data_sim_valid[1]  # Upper bin edges.
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
                "Could not match each layer/free-energy minimum to exactly one"
                " bin.  Bin edges: {}.  Free-energy minima: {}."
                "  Assignment: {}".format(bin_edges_valid, pk_pos, layer_ix)
            )
        data_sim_valid = data_sim_valid[:, layer_ix]

        # Discard the lower and upper bin edges.
        data_sim_valid = data_sim_valid[2:]

        # Get standard deviations of the residence times (if available).
        if args.method == "rate":
            # If <tau_cnt> = <tau_k> +/- 1%, use <tau_cnt^2> as estimate
            # for <tau_k^2>.
            atol = 0
            rtol = 0.01
            cnt_equals_k = np.isclose(
                data_sim_valid[2], data_sim_valid[0], rtol=rtol, atol=atol
            )
            data_sim_valid[1][~cnt_equals_k] = np.nan
            # Replace the 2nd moment of the residence time calculated by
            # the count method with the standard deviation.
            data_sim_valid[3] = np.sqrt(
                data_sim_valid[3] - data_sim_valid[2] ** 2
            )
        if len(col_lt_data_is_sd) > 1 and col_lt_data_is_sd[1]:
            # Replace the 2nd moment of the residence time with the
            # standard deviation.
            data_sim_valid[1] = np.sqrt(
                data_sim_valid[1] - data_sim_valid[0] ** 2
            )

        ydata[pkp_col_ix][pkt_ix][sim_ix] = pk_pos
        col_indices = np.arange(n_cols_tot)
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
legend_locs_sep = tuple("best" for col_ix in range(n_cols_data))
legend_locs_comb = tuple("best" for col_ix in range(n_cols_data))
if len(legend_locs_sep) != n_cols_data:
    raise ValueError(
        "`len(legend_locs_sep)` ({}) != `n_cols_data`"
        " ({})".format(len(legend_locs_sep), n_cols_data)
    )
if len(legend_locs_comb) != n_cols_data:
    raise ValueError(
        "`len(legend_locs_comb)` ({}) != `n_cols_data`"
        " ({})".format(len(legend_locs_comb), n_cols_data)
    )

xlabel = r"Ether Oxygens per Chain $n_{EO}$"
xlim = (1, 200)
if args.common_ylim:
    if args.cmp == "Li":
        # Only common for chain-length and surfq dependence.
        ylims = [
            (0, 3.6),  # Peak positions [nm].
            (7e-3, 2e1),  # 1st moment <tau> [ns].
        ]
        if args.method == "rate":
            ylims += [(7e-3, 2e1)]  # <tau_cnt> [ns].
        elif args.method == "fit":
            ylims += [
                (7e-3, 2e1),  # Fit parameter tau0 [ns].
                (0, 1.05),  # Fit parameter beta.
                (0.976, 1.001),  # R^2 value.
                (3e-7, 6e-3),  # Mean squared error of the fit [ns^2].
                (-0.02, 0.2),  # Start of fit region (inclusive) [ns].
                (3e-2, 8e1),  # End of fit region (exclusive) [ns].
            ]
        # Common for chain-length, conc and surfq dependence.
        # ylims = [
        #     (0, 3.6),  # Peak positions [nm].
        #     (4e-3, 4e4),  # 1st moment <tau> [ns].
        # ]
        # if args.method == "rate":
        #     ylims += [(4e-3, 4e4)]  # <tau_cnt> [ns].
        # elif args.method == "fit":
        #     ylims += [
        #         (4e-3, 2e3),  # Fit parameter tau0 [ns].
        #         (0, 1.05),  # Fit parameter beta.
        #         (0.920, 1.005),  # R^2 value.
        #         (3e-7, 6e-3),  # Mean squared error of the fit [ns^2].
        #         (-0.02, 0.2),  # Start of fit region (inclusive) [ns].
        #         (3e-2, 2e3),  # End of fit region (exclusive) [ns].
        #     ]
    else:
        raise NotImplementedError(
            "Common ylim not implemented for compounds other than Li"
        )
else:
    ylims = tuple((None, None) for col_ix in range(n_cols_data))
if len(ylims) != n_cols_data:
    raise ValueError(
        "`len(ylims)` ({}) != `n_cols_data`"
        " ({})".format(len(ylims), n_cols_data)
    )

# Whether to use log scale for the y-axis.
logy = (
    False,  # Peak positions [nm].
    True,  # 1st moment <tau> [ns].
)
if args.method == "rate":
    logy += (True,)  # Residence time from count method <tau_cnt> [ns].
elif args.method == "fit":
    logy += (
        True,  # Fit parameter tau0 [ns].
        False,  # Fit parameter beta.
        False,  # Coeff. of determ. of the fit (R^2 value).
        True,  # Mean squared error of the fit [ns^2].
        False,  # Start of fit region (inclusive) [ns].
        True,  # End of fit region (exclusive) [ns].
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
    # Index for "real" data columns (no standard deviations).
    col_ix_data = -1
    for col_ix, yd_col in enumerate(ydata):
        if col_is_sd[col_ix]:
            # Column contains a standard deviation.
            continue
        col_ix_data += 1

        # Layers at left and right electrode combined in one plot.
        fig_comb, ax_comb = plt.subplots(clear=True)

        for pkt_ix, pkp_type in enumerate(pk_pos_types):
            yd_pkt = yd_col[pkt_ix]

            # Layers at left and right electrode in separate plots.
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
                if col_has_sd[col_ix]:
                    ax_sep.errorbar(
                        xdata[pkt_ix][valid],
                        yd_pkt[valid],
                        yerr=ydata[col_ix + 1][pkt_ix][valid],
                        linestyle="solid",
                        marker=markers[pkt_ix],
                        label=label_sep,
                    )
                    ax_comb.errorbar(
                        xdata[pkt_ix][valid],
                        yd_pkt[valid],
                        yerr=ydata[col_ix + 1][pkt_ix][valid],
                        linestyle=linestyles_comb[pkt_ix],
                        marker=markers[pkt_ix],
                        label=label_comb,
                    )
                else:
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
            ax_sep.set_xscale("log", base=10, subs=np.arange(2, 10))
            if logy[col_ix_data]:
                ax_sep.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax_sep.set(
                xlabel=xlabel,
                ylabel=ylabels[col_ix_data],
                xlim=xlim,
                ylim=ylims[col_ix_data],
            )
            if not logy[col_ix_data]:
                equalize_yticks(ax_sep)
            legend_sep = ax_sep.legend(
                title=legend_title_sep + legend_title_suffix,
                ncol=2,
                loc=legend_locs_sep[col_ix_data],
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend_sep.get_title().set_multialignment("center")
            pdf.savefig(fig_sep)
            plt.close(fig_sep)

        ax_comb.set_xscale("log", base=10, subs=np.arange(2, 10))
        if logy[col_ix_data]:
            ax_comb.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax_comb.set(
            xlabel=xlabel,
            ylabel=ylabels[col_ix_data],
            xlim=xlim,
            ylim=ylims[col_ix_data],
        )
        if not logy[col_ix_data]:
            equalize_yticks(ax_comb)
        legend_comb = ax_comb.legend(
            title=legend_title(r"\pm") + legend_title_suffix,
            ncol=n_legend_cols_comb,
            loc=legend_locs_comb[col_ix_data],
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
