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
n_data_per_plot = len(trans_drctn_types) * len(trans_types)
n_data = 1  # Barrier positions.
n_data += len(rows) * n_data_per_plot * 2
n_data -= n_data_per_plot  # No Std for Type 5
n_data += 2  # Plot type 6.
ydata = [
    [[[] for sim in Sims.sims] for pkp_type in pk_pos_types]
    for dat_ix in range(n_data)
]
# Boolean array that indicates which of the data are standard deviations
data_is_sd = np.zeros(n_data, dtype=bool)
data_is_sd[2 : 3 * n_data_per_plot * 2 + 1 : 2] = True
data_is_sd[-n_data_per_plot * 2 + 1 :: 2] = True
n_data_not_sd = n_data - np.count_nonzero(data_is_sd)

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
xlabel = r"Ether Oxygens per Chain $n_{EO}$"
xlim = (1, 200)
if args.common_ylim:
    if cmp1 == "Li":
        ylims = [
            (0, 3.6),  # Barrier positions [nm].
        ]
    else:
        raise NotImplementedError(
            "Common ylim not implemented for compounds other than Li"
        )
    ylims += [(None, None) for dat_ix in range(n_data_not_sd - 1)]
else:
    ylims = tuple((None, None) for dat_ix in range(n_data_not_sd))

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    ####################################################################
    # Plot ligand exchange information for each free-energy barrier.
    labels = (
        "Dissociated",
        "Associated",
        "Remained",
        "No. of Transitions",
        "Fraction of Succ. Trans.",
        "Crossover Time",
    )
    col_ix_lower = 1
    for lbl_ix, label in enumerate(labels):
        if label == "No. of Transitions":
            col_ix_upper = col_ix_lower + n_data_per_plot
        elif label == "Fraction of Succ. Trans.":
            col_ix_upper = col_ix_lower + 2
        else:
            col_ix_upper = col_ix_lower + 2 * n_data_per_plot

        figs, axes = [], []
        for pkt_ix in range(len(pk_pos_types)):
            figs.append([])
            axes.append([])
            for _cix_pkt in clstr_ix_unq[pkt_ix]:
                fig, ax = plt.subplots(clear=True)
                figs[pkt_ix].append(fig)
                axes[pkt_ix].append(ax)

        for col_ix, yd_col in enumerate(
            ydata[col_ix_lower:col_ix_upper], start=col_ix_lower
        ):
            if data_is_sd[col_ix]:
                # Column contains a standard deviation.
                continue
            for pkt_ix, pkp_type in enumerate(pk_pos_types):
                yd_pkt = yd_col[pkt_ix]
                for cix_pkt in clstr_ix_unq[pkt_ix]:
                    valid = clstr_ix[pkt_ix] == cix_pkt
                    if not np.any(valid):
                        raise ValueError(
                            "No valid peaks for peak type '{}' and cluster"
                            " index {}".format(pkp_type, cix_pkt)
                        )
                    clstr_barrier_pos = np.mean(
                        ydata[pkp_col_ix][pkt_ix][valid]
                    )

                    ax = axes[pkt_ix][cix_pkt]
                    ax.errorbar(
                        xdata[pkt_ix][valid],
                        yd_pkt[valid],
                        yerr=ydata[col_ix + 1][pkt_ix][valid],
                    )

        for pkt_ix, ax_pkt in enumerate(axes):
            for cix_pkt, ax in enumerate(ax_pkt):
                ax.set_xscale("log", base=10, subs=np.arange(2, 10))
                ax.set(xlabel=xlabel, xlim=xlim)
                equalize_yticks(ax)
                legend = ax.legend(
                    title=label, ncol=2, **mdtplt.LEGEND_KWARGS_XSMALL
                )
                legend.get_title().set_multialignment("center")
                pdf.savefig(figs[pkt_ix][cix_pkt])
                plt.close(figs[pkt_ix][cix_pkt])

        col_ix_lower = col_ix_upper

    ####################################################################
    # Plot barrier positions.
    ylabels = ("Distance to Electrode / nm",)
    legend_title_suffix = (
        r"$F_{"
        + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
        + r"}$ Maxima Positions / nm"
    )
    linestyles_comb = ("solid", "dashed")
    markers = ("<", ">")

    n_clstrs_plot = np.max(n_clstrs)
    cmap = plt.get_cmap()
    c_vals_sep = np.arange(n_clstrs_plot)
    c_norm_sep = n_clstrs_plot - 1
    c_vals_sep_normed = c_vals_sep / c_norm_sep
    colors_sep = cmap(c_vals_sep_normed)

    for col_ix, yd_col in enumerate(
        ydata[pkp_col_ix : pkp_col_ix + 1], start=pkp_col_ix
    ):
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
            ax_sep.set_xscale("log", base=10, subs=np.arange(2, 10))
            ax_sep.set(
                xlabel=xlabel,
                ylabel=ylabels[col_ix],
                xlim=xlim,
                ylim=ylims[col_ix],
            )
            equalize_yticks(ax_sep)
            legend_sep = ax_sep.legend(
                title=legend_title_sep + legend_title_suffix,
                ncol=2,
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend_sep.get_title().set_multialignment("center")
            pdf.savefig(fig_sep)
            plt.close(fig_sep)

        ax_comb.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax_comb.set(
            xlabel=xlabel,
            ylabel=ylabels[col_ix],
            xlim=xlim,
            ylim=ylims[col_ix],
        )
        equalize_yticks(ax_comb)
        legend_comb = ax_comb.legend(
            title=legend_title(r"\pm") + legend_title_suffix,
            ncol=2,
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend_comb.get_title().set_multialignment("center")
        pdf.savefig(fig_comb)
        plt.close(fig_comb)

    ####################################################################
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
            title=(
                legend_title_clstrng
                + r"$F_{"
                + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
                + r"}$ Maxima"
            ),
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
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()
