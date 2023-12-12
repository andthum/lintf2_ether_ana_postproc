#!/usr/bin/env python3


"""
Plot the lifetime histogram obtained from the count method for selected
bins for various chain lengths.
"""


# Standard libraries
import argparse

# Third-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# First-party libraries
import lintf2_ether_ana_postproc as leap


def histogram(dtrj, use_states, uncensored=False):
    """TODO"""
    dtrj = mdt.check.dtrj(dtrj)
    n_frames = dtrj.shape[-1]
    lts_per_state, states = mdt.dtrj.lifetimes_per_state(
        dtrj, uncensored=uncensored, return_states=True
    )
    del dtrj

    use_states = np.unique(mdt.check.array(use_states, dim=1))
    state_ix = np.flatnonzero(np.isin(states, use_states, assume_unique=True))
    if len(state_ix) != len(use_states):
        raise RuntimeError(
            "Not all of the given states ({}) are contained in the discrete"
            " trajectory ({})".format(use_states, states)
        )
    states = states[state_ix]
    lts_per_state = [
        lts_state for i, lts_state in enumerate(lts_per_state) if i in state_ix
    ]
    n_states = len(states)
    del use_states, state_ix

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
    return hists, bins


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


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_gra_" + args.surfq + "_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, args.surfq, sort_key="O_per_chain"
)


print("Reading data and creating plot(s)...")
# Get filenames of the discrete trajectories.
file_suffix = analysis + "_" + args.cmp + "_dtrj.npz"
infiles = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
n_infiles = len(infiles)

cmap = plt.get_cmap()
c_vals = np.arange(n_infiles)
c_norm = max(1, n_infiles - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        # Read discrete trajectory from file.
        dtrj = mdt.fh.load_dtrj(infiles[sim_ix])
        if args.intermittency > 0:
            print("Correcting for intermittency...")
            dtrj = mdt.dyn.correct_intermittency(
                dtrj.T, args.intermittency, inplace=True, verbose=True
            )
            dtrj = dtrj.T

        states = np.unique(dtrj)
        # Indices of the bins for which to plot the lifetime histogram.
        use_states = np.array([states[len(states) // 2], states[-1]])
        hists, bins = histogram(
            dtrj, use_states=use_states, uncensored=args.uncensored
        )
        del dtrj

        bin_mids = bins[1:] - np.diff(bins) / 2
        unit_bins = True if np.allclose(np.diff(bins), 1) else False
        for state_ix, hist in enumerate(hists):
            state_num = use_states[state_ix]
            if not unit_bins:
                ax.stairs(
                    hist,
                    bins,
                    fill=False,
                    label=r"$%d$" % (state_num + 1),
                    alpha=leap.plot.ALPHA,
                    rasterized=False,
                )
            else:
                ax.plot(
                    bin_mids,
                    hist,
                    label=r"$%d$" % (state_num + 1),
                    alpha=leap.plot.ALPHA,
                    rasterized=False,
                )
