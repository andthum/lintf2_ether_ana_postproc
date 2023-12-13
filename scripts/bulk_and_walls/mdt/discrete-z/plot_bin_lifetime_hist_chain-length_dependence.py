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

# First-party libraries
import lintf2_ether_ana_postproc as leap


def legend_title(surfq_sign, position):
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
    title = (
        r"$\sigma_s = "
        + surfq_sign
        + r" %.2f$ $e$/nm$^2$, " % Sims.surfqs[0]
        + r"$r = %.2f$" % Sims.Li_O_ratios[0]
        + "\n"
    )
    if position == "bulk":
        title += "Bulk\n"
    else:
        title += r"Dist. to Electrode: $\sim %.2f$ nm" % position + "\n"
    title += r"$n_{EO}$"
    return title


def histograms(dtrj_file, uncensored=False, intermittency=0, time_conv=1):
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
    if n_states == 0:
        raise ValueError("`n_states` == 0")
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
    bins *= time_conv
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
hists_bins = [None for sim in Sims.sims]
# Lifetime histograms in the bulk and for each layer/free-energy minimum
hists_bulk = [None for sim in Sims.sims]
hists_layer = [[None for sim in Sims.sims] for pkp_type in pk_pos_types]
hist_layer_states = [[None for sim in Sims.sims] for pkp_type in pk_pos_types]
n_states_max = [0 for pkp_type in pk_pos_types]
# Lower and upper bin edges used for generating the discrete trajectory
# -> Free-energy maxima.
# Don't confuse position bins (used to generate the discrete trajectory)
# with time bins (used to generate the lifetime histograms).  Time bins
# will always be prefixed with "hist_" or "hists_".
bin_edges = [[None for sim in Sims.sims] for pkp_type in pk_pos_types]
# Data to be clustered: Bin midpoints, Simulation indices, bin indices.
n_data_clstr = 3
pkp_col_ix = 0
data_clstr = [
    [[None for sim in Sims.sims] for pkp_type in pk_pos_types]
    for col_ix in range(n_data_clstr)
]
for sim_ix, Sim in enumerate(Sims.sims):
    # Calculate the lifetime histogram for each state from the discrete
    # trajectory.
    hists_sim, hists_bins_sim, states_sim = histograms(
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
    if len(bins) - 1 < len(states_sim):
        raise ValueError(
            "Simulation: '{}'.\n".format(Sim.path)
            + "The number of bins in the bin file is less than the number of"
            + " states in the discrete trajectory.\n"
            + "Bins:   {}.\n".format(bins)
            + "States: {}.".format(states_sim)
        )
    bins /= 10  # A -> nm.
    bin_edges_lower = bins[:-1]
    bin_edges_upper = bins[1:]
    bin_mids = bins[1:] - np.diff(bins) / 2
    bin_data = np.column_stack([bin_edges_lower, bin_edges_upper, bin_mids])
    del bin_edges_lower, bin_edges_upper, bin_mids
    # Select all bins for which lifetime histograms are available.
    bin_data = bin_data[states_sim]
    box_z = Sim.box[2] / 10  # A -> nm
    bin_is_left = bin_data[:, -1] <= (box_z / 2)
    tolerance = 1e-6
    if np.any(bin_data <= -tolerance):
        raise ValueError(
            "Simulation: '{}'.\n".format(Sim.path)
            + "At least one bin edge is less than zero.\n"
            + "Bin edges: {}.".format(bins)
        )
    if np.any(bin_data >= box_z + tolerance):
        raise ValueError(
            "Simulation: '{}'.\n".format(Sim.path)
            + "At least one bin edge is greater than the box length"
            + " ({}).\n".format(box_z)
            + "Bin edges: {}.".format(bins)
        )
    del bins

    for pkt_ix, pkp_type in enumerate(pk_pos_types):
        if pkp_type == "left":
            valid_bins = bin_is_left
            hists_sim_valid = hists_sim[valid_bins]
            states_sim_valid = states_sim[valid_bins]
            bin_data_valid = bin_data[valid_bins]
            # Convert absolute bin positions to distances to the
            # electrodes.
            bin_data_valid -= elctrd_thk
        elif pkp_type == "right":
            valid_bins = ~bin_is_left
            hists_sim_valid = hists_sim[valid_bins]
            states_sim_valid = states_sim[valid_bins]
            bin_data_valid = bin_data[valid_bins]
            # Reverse the order of rows to sort bins as function of the
            # distance to the electrodes in ascending order.
            hists_sim_valid = hists_sim_valid[::-1]
            states_sim_valid = states_sim_valid[::-1]
            bin_data_valid = bin_data_valid[::-1]
            # Convert absolute bin positions to distances to the
            # electrodes.
            bin_data_valid += elctrd_thk
            bin_data_valid -= box_z
            bin_data_valid *= -1  # Ensure positive distance values.
        else:
            raise ValueError(
                "Unknown peak position type: '{}'".format(pkp_type)
            )
        if np.any(bin_data_valid < -tolerance):
            raise ValueError(
                "Simulation: '{}'.\n".format(Sim.path)
                + "Peak-position type: '{}'.\n".format(pkp_type)
                + "At least one bin lies within the electrode.  This should"
                + " not have happened.\n"
                + "Bin edges: {}.\n".format(bin_data_valid)
                + "Electrode: 0"
            )

        hists_layer[pkt_ix][sim_ix] = hists_sim_valid
        hist_layer_states[pkt_ix][sim_ix] = states_sim_valid
        bin_edges[pkt_ix][sim_ix] = bin_data_valid[:, :2]

        n_states_valid = len(states_sim_valid)
        n_states_max[pkt_ix] = max(n_states_valid, n_states_max[pkt_ix])
        data_clstr[pkp_col_ix][pkt_ix][sim_ix] = bin_data_valid[:, -1]
        data_clstr[1][pkt_ix][sim_ix] = np.full(
            n_states_valid, sim_ix, dtype=np.uint16
        )
        data_clstr[2][pkt_ix][sim_ix] = states_sim_valid


print("Clustering peak positions...")
(
    data_clstr,
    clstr_ix,
    linkage_matrices,
    n_clstrs,
    n_pks_per_sim,
    clstr_dist_thresh,
) = leap.clstr.peak_pos(data_clstr, pkp_col_ix, return_dist_thresh=True)
clstr_ix_unq = [np.unique(clstr_ix_pkt) for clstr_ix_pkt in clstr_ix]

# Sort clusters by ascending average peak position.
for pkt_ix, clstr_ix_pkt in enumerate(clstr_ix):
    _clstr_dists, clstr_ix[pkt_ix] = leap.clstr.dists_succ(
        data_clstr[pkp_col_ix][pkt_ix],
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


print("Creating plot(s)...")
xlabel = "Residence Time / ns"
ylabel = "PDF"
xlim = (time_conv, 1e3)
ylim = (None, None)

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
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
    )
    legend = ax.legend(
        title=legend_title(r"\pm", "bulk"),
        loc="upper right",
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot lifetime histograms for each bin.
    for pkt_ix, pkp_type in enumerate(pk_pos_types):
        hists_layer_pkt = hists_layer[pkt_ix]
        hist_layer_states_pkt = hist_layer_states[pkt_ix]
        bin_edges_pkt = bin_edges[pkt_ix]

        if pkp_type == "left":
            surfq_sign = "+"
        elif pkp_type == "right":
            surfq_sign = "-"
        else:
            raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))

        for cix_pkt in clstr_ix_unq[pkt_ix]:
            valid_clstr = clstr_ix[pkt_ix] == cix_pkt
            if not np.any(valid_clstr):
                raise ValueError(
                    "No valid peaks for peak type '{}' and cluster index"
                    " {}".format(pkp_type, cix_pkt)
                )
            clstr_bin_mids = data_clstr[pkp_col_ix][pkt_ix][valid_clstr]
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
                valid_hist = hist_layer_states_pkt[sim_ix] == hist_state
                if np.count_nonzero(valid_hist) != 1:
                    raise ValueError(
                        "The number of valid histograms in the current cluster"
                        " for the current simulation is not one"
                    )
                hist_ix = np.flatnonzero(valid_hist)[0]
                hist = hists_layer_pkt[sim_ix][hist_ix]
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
            ax.set_xscale("log", base=10, subs=np.arange(2, 10))
            ax.set_yscale("log", base=10, subs=np.arange(2, 10))
            ax.set(
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylim,
            )
            legend = ax.legend(
                title=legend_title(surfq_sign, clstr_position),
                loc="upper right",
                ncol=n_legend_cols,
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend.get_title().set_multialignment("center")
            pdf.savefig()
            plt.close()

print("Created {}".format(outfile))
print("Done")
