#!/usr/bin/env python3


"""
Plot the lifetime histogram obtained from the count method for selected
bins for various chain lengths.
"""


# Standard libraries
import argparse
import warnings

# Third-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize import curve_fit

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
outfile += ".pdf"

# Time conversion factor to convert from trajectory steps to ns.
time_conv = 2e-3
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

Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm.
bulk_start = Elctrd.BULK_START / 10  # A -> nm.

pk_pos_types = ("left", "right")
# Bin edges used for generating the lifetime histograms.
# Don't confuse position bins (used to generate the discrete trajectory)
# with time bins (used to generate the lifetime histograms).  Time bins
# will always be prefixed with "hist_" or "hists_".
hists_bins = [None for sim in Sims.sims]
# Lifetime histograms in the bulk and for each layer/free-energy minimum
hists_bulk = [None for sim in Sims.sims]
hists_layer = [[None for sim in Sims.sims] for pkp_type in pk_pos_types]
hists_layer_states = [[None for sim in Sims.sims] for pkp_type in pk_pos_types]
n_states_max = [0 for pkp_type in pk_pos_types]
# Data to be clustered: Bin midpoints, Simulation indices,
# state/bin indices.
n_data_clstr = 3
bin_mid_ix = 0
data_clstr = [
    [[None for sim in Sims.sims] for pkp_type in pk_pos_types]
    for col_ix in range(n_data_clstr)
]
for sim_ix, Sim in enumerate(Sims.sims):
    # Calculate the lifetime histogram for each state from the discrete
    # trajectory.
    hists_sim, hists_bins_sim, states_sim = leap.lifetimes.histograms(
        infiles_dtrj[sim_ix],
        uncensored=args.uncensored,
        intermittency=args.intermittency,
        time_conv=time_conv,
    )
    hists_bins[sim_ix] = hists_bins_sim
    hists_bulk[sim_ix] = hists_sim[len(states_sim) // 2]
    del hists_bins_sim

    # Read bin edges.
    bins = np.loadtxt(infiles_bins[sim_ix], dtype=np.float32)
    bins /= 10  # A -> nm.
    if len(bins) - 1 < len(states_sim):
        raise ValueError(
            "Simulation: '{}'.\n".format(Sim.path)
            + "The number of bins in the bin file is less than the number of"
            + " states in the discrete trajectory.\n"
            + "Bins:   {}.\n".format(bins)
            + "States: {}.".format(states_sim)
        )
    box_z = Sim.box[2] / 10  # A -> nm
    tolerance = 1e-4
    if np.any(bins <= -tolerance):
        raise ValueError(
            "Simulation: '{}'.\n".format(Sim.path)
            + "At least one bin edge is less than zero.\n"
            + "Bin edges: {}.".format(bins)
        )
    if np.any(bins >= box_z + tolerance):
        raise ValueError(
            "Simulation: '{}'.\n".format(Sim.path)
            + "At least one bin edge is greater than the box length"
            + " ({}).\n".format(box_z)
            + "Bin edges: {}.".format(bins)
        )
    bin_mids = bins[1:] - np.diff(bins) / 2
    del bins
    # Select all bins for which lifetime histograms are available.
    bin_mids = bin_mids[states_sim]
    bin_is_left = bin_mids <= (box_z / 2)

    for pkt_ix, pkp_type in enumerate(pk_pos_types):
        if pkp_type == "left":
            valid_bins = bin_is_left
            # Discard bins in the bulk region.
            valid_bins &= bin_mids < elctrd_thk + bulk_start
            hists_sim_valid = hists_sim[valid_bins]
            states_sim_valid = states_sim[valid_bins]
            bin_mids_valid = bin_mids[valid_bins]
            # Convert absolute bin positions to distances to the
            # electrodes.
            bin_mids_valid -= elctrd_thk
        elif pkp_type == "right":
            valid_bins = ~bin_is_left
            # Discard bins in the bulk region.
            valid_bins &= bin_mids > box_z - elctrd_thk - bulk_start
            hists_sim_valid = hists_sim[valid_bins]
            states_sim_valid = states_sim[valid_bins]
            bin_mids_valid = bin_mids[valid_bins]
            # Reverse the order of rows to sort bins as function of the
            # distance to the electrodes in ascending order.
            hists_sim_valid = hists_sim_valid[::-1]
            states_sim_valid = states_sim_valid[::-1]
            bin_mids_valid = bin_mids_valid[::-1]
            # Convert absolute bin positions to distances to the
            # electrodes.
            bin_mids_valid += elctrd_thk
            bin_mids_valid -= box_z
            bin_mids_valid *= -1  # Ensure positive distance values.
        else:
            raise ValueError(
                "Unknown bin position type: '{}'".format(pkp_type)
            )
        if np.any(bin_mids_valid <= 0):
            raise ValueError(
                "Simulation: '{}'.\n".format(Sim.path)
                + "Bin-position type: '{}'.\n".format(pkp_type)
                + "At least one bin lies within the electrode.  This should"
                + " not have happened.\n"
                + "Bin edges: {}.\n".format(bin_mids_valid)
                + "Electrode: 0"
            )

        hists_layer[pkt_ix][sim_ix] = hists_sim_valid
        hists_layer_states[pkt_ix][sim_ix] = states_sim_valid

        n_states_valid = len(states_sim_valid)
        n_states_max[pkt_ix] = max(n_states_valid, n_states_max[pkt_ix])
        data_clstr[bin_mid_ix][pkt_ix][sim_ix] = bin_mids_valid
        data_clstr[1][pkt_ix][sim_ix] = np.full(
            n_states_valid, sim_ix, dtype=np.uint16
        )
        data_clstr[2][pkt_ix][sim_ix] = states_sim_valid


print("Clustering bin positions...")
(
    data_clstr,
    clstr_ix,
    linkage_matrices,
    n_clstrs,
    n_pks_per_sim,
    clstr_dist_thresh,
) = leap.clstr.peak_pos(data_clstr, bin_mid_ix, return_dist_thresh=True)
clstr_ix_unq = [np.unique(clstr_ix_pkt) for clstr_ix_pkt in clstr_ix]

# Sort clusters by ascending average bin position.
for pkt_ix, clstr_ix_pkt in enumerate(clstr_ix):
    _clstr_dists, clstr_ix[pkt_ix] = leap.clstr.dists_succ(
        data_clstr[bin_mid_ix][pkt_ix],
        clstr_ix_pkt,
        method=clstr_dist_method,
        return_ix=True,
    )

if np.any(n_clstrs < n_states_max):
    warnings.warn(
        "Any `n_clstrs` ({}) < `n_states_max` ({}).  This means different"
        " bins of the same simulation were assigned to the same cluster."
        "  Try to decrease the threshold distance"
        " ({})".format(n_clstrs, n_states_max, clstr_dist_thresh),
        RuntimeWarning,
        stacklevel=2,
    )

xdata = [None for n_pks_per_sim_pkt in n_pks_per_sim]
for pkt_ix, n_pks_per_sim_pkt in enumerate(n_pks_per_sim):
    if np.max(n_pks_per_sim_pkt) != n_states_max[pkt_ix]:
        raise ValueError(
            "`np.max(n_pks_per_sim[{}])` ({}) != `n_states_max[{}]` ({})."
            "  This should not have happened".format(
                pkt_ix, np.max(n_pks_per_sim_pkt), pkt_ix, n_states_max[pkt_ix]
            )
        )
    xdata[pkt_ix] = [
        Sims.O_per_chain[sim_ix]
        for sim_ix, n_pks_sim in enumerate(n_pks_per_sim_pkt)
        for _ in range(n_pks_sim)
    ]
    xdata[pkt_ix] = np.array(xdata[pkt_ix])


print("Creating plot(s)...")
xlabel = "Residence Time / ns"
ylabel = "PDF"
xlim = (time_conv, 1e3)
ylim = (1e-9, 1e0)

legend_title_suffix = "\n" + r"$n_{EO}$"
legend_loc = "best"
n_legend_cols = 1 + Sims.n_sims // (4 + 1)

cmap = plt.get_cmap()
c_vals = np.arange(Sims.n_sims)
c_norm = max(1, Sims.n_sims - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot lifetime histograms for bulk bin.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, hist in enumerate(hists_bulk):
        ax.stairs(
            hist,
            hists_bins[sim_ix],
            fill=False,
            label=r"$%d$" % Sims.O_per_chain[sim_ix],
            alpha=leap.plot.ALPHA,
            rasterized=False,
        )
        # Use for linear bins
        # ax.plot(
        #     hists_bins[sim_ix],
        #     hist,
        #     label=r"$%d$" % Sims.O_per_chain[sim_ix],
        #     alpha=leap.plot.ALPHA,
        #     rasterized=False,
        # )
        if Sims.O_per_chain[sim_ix] == 64:
            # Fit histogram by power law.
            start, stop = 0, 10
            hist_fit = hist[start:stop]
            hist_bin_mids_fit = np.copy(hists_bins[sim_ix][1:])
            hist_bin_mids_fit -= np.diff(hists_bins[sim_ix]) / 2
            hist_bin_mids_fit = hist_bin_mids_fit[start:stop]
            popt, pcov = curve_fit(
                f=leap.misc.straight_line,
                xdata=np.log(hist_bin_mids_fit),
                ydata=np.log(hist_fit),
                p0=(-1.5, np.log(hist_fit[0])),
            )
            hist_fit = leap.misc.power_law(
                hist_bin_mids_fit, popt[0], np.exp(popt[1])
            )
    ax.plot(
        hist_bin_mids_fit,
        hist_fit,
        color="black",
        linestyle="dashed",
        alpha=leap.plot.ALPHA,
    )
    ax.text(
        hist_bin_mids_fit[1] / 1.2,
        hist_fit[1] * 1.3,
        r"$\propto t^{%.2f}$" % popt[0],
        # rotation=np.rad2deg(np.arctan(popt[0])) / 1.5,
        rotation_mode="anchor",
        transform_rotates_text=False,
        horizontalalignment="left",
        verticalalignment="bottom",
        fontsize="small",
    )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
    )
    legend_title_pdf = (
        legend_title(r"\pm") + "Bulk Region" + legend_title_suffix
    )
    legend = ax.legend(
        title=legend_title_pdf,
        loc=legend_loc,
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot lifetime histograms for each bin.
    for pkt_ix, pkp_type in enumerate(pk_pos_types):
        if pkp_type == "left":
            legend_title_pdf_base = legend_title("+")
        elif pkp_type == "right":
            legend_title_pdf_base = legend_title("-")
        else:
            raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))

        for cix_pkt in clstr_ix_unq[pkt_ix]:
            valid_clstr = clstr_ix[pkt_ix] == cix_pkt
            if not np.any(valid_clstr):
                raise ValueError(
                    "No valid bins for bin type '{}' and cluster index"
                    " {}".format(pkp_type, cix_pkt)
                )
            clstr_bin_mids = data_clstr[bin_mid_ix][pkt_ix][valid_clstr]
            clstr_position = np.mean(clstr_bin_mids)
            clstr_sim_indices = data_clstr[1][pkt_ix][valid_clstr]
            clstr_states = data_clstr[2][pkt_ix][valid_clstr]
            if not np.array_equal(
                clstr_sim_indices, np.unique(clstr_sim_indices)
            ):
                raise ValueError(
                    "Different bins of the same simulation were assigned to"
                    " the same cluster."
                )

            fig, ax = plt.subplots(clear=True)
            for six, sim_ix in enumerate(clstr_sim_indices):
                hist_state = clstr_states[six]
                valid_hist = hists_layer_states[pkt_ix][sim_ix] == hist_state
                if np.count_nonzero(valid_hist) != 1:
                    raise ValueError(
                        "The number of valid histograms in the current cluster"
                        " for the current simulation is not one"
                    )
                hist_ix = np.flatnonzero(valid_hist)[0]
                hist = hists_layer[pkt_ix][sim_ix][hist_ix]
                ax.stairs(
                    hist,
                    hists_bins[sim_ix],
                    fill=False,
                    label=r"$%d$" % Sims.O_per_chain[sim_ix],
                    color=colors[sim_ix],
                    alpha=leap.plot.ALPHA,
                    rasterized=False,
                )
                # Use for linear bins
                # ax.plot(
                #     hists_bins[sim_ix],
                #     hist,
                #     label=r"$%d$" % Sims.O_per_chain[sim_ix],
                #     color=colors[sim_ix],
                #     alpha=leap.plot.ALPHA,
                #     rasterized=False,
                # )
            # Plot Fit of bulk histogram.
            ax.plot(
                hist_bin_mids_fit,
                hist_fit,
                color="black",
                linestyle="dashed",
                alpha=leap.plot.ALPHA,
            )
            ax.text(
                hist_bin_mids_fit[1] / 1.2,
                hist_fit[1] * 1.3,
                r"$\propto t^{%.2f}$" % popt[0],
                # rotation=np.rad2deg(np.arctan(popt[0])) / 1.5,
                rotation_mode="anchor",
                transform_rotates_text=False,
                horizontalalignment="left",
                verticalalignment="bottom",
                fontsize="small",
            )
            ax.set_xscale("log", base=10, subs=np.arange(2, 10))
            ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax.set(
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylim,
            )
            legend_title_pdf = (
                legend_title_pdf_base
                # + r"$F_{"
                # + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
                # + r"}$ Minimum $\sim %.2f$ nm" % clstr_position
                + r"Dist. to Electrode $\sim %.2f$ nm" % clstr_position
                + legend_title_suffix
            )
            legend = ax.legend(
                title=legend_title_pdf,
                loc=legend_loc,
                ncol=n_legend_cols,
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend.get_title().set_multialignment("center")
            pdf.savefig()
            plt.close()

    # Plot clustering results.
    xlabel = r"Ether Oxygens per Chain $n_{EO}$"
    ylabel = "Distance to Electrode / nm"
    xlim = (1, 200)
    ylim = (0, bulk_start)
    legend_title_suffix = "Bin Positions / nm"
    markers = ("<", ">")
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
        ax.set(xlabel="Bin Number", ylabel="Bin Distance / nm", ylim=(0, None))
        legend = ax.legend(
            title=legend_title_clstrng + "Bins",
            loc="best",
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

        # Scatter plot: Bin Positions vs. Bin Positions.
        fig, ax = plt.subplots(clear=True)
        for cix_pkt in clstr_ix_unq[pkt_ix]:
            valid_clstr = clstr_ix[pkt_ix] == cix_pkt
            if not np.any(valid_clstr):
                raise ValueError(
                    "No valid bins for bin type '{}' and cluster index"
                    " {}".format(pkp_type, cix_pkt)
                )
            clstr_bin_mids = data_clstr[bin_mid_ix][pkt_ix][valid_clstr]
            clstr_position = np.mean(clstr_bin_mids)
            ax.scatter(
                clstr_bin_mids,
                clstr_bin_mids,
                color=cmap(cmap_norm(cix_pkt)),
                marker=markers[pkt_ix],
                label=r"$\sim %.2f$" % clstr_position,
            )
        ax.set(xlabel=ylabel, ylabel=ylabel, xlim=ylim, ylim=ylim)
        legend = ax.legend(
            title=legend_title_clstrng + legend_title_suffix,
            ncol=2,
            loc="upper left",
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

        # Scatter plot: Bin Positions vs. `xdata`.
        fig, ax = plt.subplots(clear=True)
        for cix_pkt in clstr_ix_unq[pkt_ix]:
            valid_clstr = clstr_ix[pkt_ix] == cix_pkt
            if not np.any(valid_clstr):
                raise ValueError(
                    "No valid bins for bin type '{}' and cluster index"
                    " {}".format(pkp_type, cix_pkt)
                )
            clstr_bin_mids = data_clstr[bin_mid_ix][pkt_ix][valid_clstr]
            clstr_position = np.mean(clstr_bin_mids)
            ax.scatter(
                xdata[pkt_ix][valid_clstr],
                clstr_bin_mids,
                color=cmap(cmap_norm(cix_pkt)),
                marker=markers[pkt_ix],
                label=r"$\sim %.2f$" % clstr_position,
            )
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
        legend = ax.legend(
            title=legend_title_clstrng + legend_title_suffix,
            ncol=2,
            loc="best",
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

print("Created {}".format(outfile))
print("Done")
