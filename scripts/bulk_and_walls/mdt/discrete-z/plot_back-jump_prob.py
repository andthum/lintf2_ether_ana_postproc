#!/usr/bin/env python3


"""
Plot the probability of a given compound to jump back to its previous
layer as function of time for a single simulation.
"""


# Standard libraries
import argparse

# Third-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# First-party libraries
import lintf2_ether_ana_postproc as leap


def scaling_law(x, c, m):
    """
    Fit function to fit the time dependence of the back-jump
    probability.
    """
    return c * x**m


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the probability of a given compound to jump back to its previous"
        " layer as function of time for a single simulation."
    )
)
parser.add_argument(
    "--system",
    type=str,
    required=True,
    help="Name of the simulated system, e.g. lintf2_g1_20-1_gra_q1_sc80.",
)
parser.add_argument(
    "--settings",
    type=str,
    required=False,
    default="pr_nvt423_nh",
    help=(
        "String describing the used simulation settings.  Default:"
        " %(default)s."
    ),
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
    "--continuous",
    required=False,
    default=False,
    action="store_true",
    help="Use the 'continuous' (true) back-jump probability.",
)
args = parser.parse_args()

if args.continuous:
    con = "_continuous"
else:
    con = ""

analysis = "discrete-z"  # Analysis name.
tool = "mdt"  # Analysis software.
outfile = (
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis
    + "_"
    + args.cmp
    + "_back_jump_prob_discrete"
    + con
    + ".pdf"
)


print("Creating Simulation instance(s)...")
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    top_path = "q%g" % surfq
else:
    surfq = None
    top_path = "bulk"
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, top_path)


print("Reading data...")
file_suffix = (
    analysis + "_" + args.cmp + "_back_jump_prob_discrete" + con + ".txt.gz"
)
infile = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
bj_probs, times, states = leap.simulation.read_time_state_matrix(
    infile,
    time_conv=2e-3,  # trajectory steps -> ns.
    amin=0,
    amax=1,
)
bj_probs = np.ascontiguousarray(bj_probs.T)
n_states = len(states)


print("Get scaling law of the back-jump probability...")
# Select a state from the center of the simulation box
bj_prob_fit = bj_probs[n_states // 2][1:50]
times_fit = times[1:50]
popt, pcov = curve_fit(
    f=scaling_law,
    xdata=times_fit,
    ydata=bj_prob_fit,
    p0=(bj_prob_fit[0], -1),
)
perr = np.sqrt(np.diag(pcov))
bj_prob_fit = scaling_law(times_fit, *popt)


print("Creating plots...")
if args.continuous:
    xmax_ylin = 0.2
    xmax_ylog = 200
    ymax = 0.5
    ymin_ylog = np.min(bj_probs[bj_probs > 0])
else:
    xmax_ylin = 0.2
    xmax_ylog = times[-1]
    ymax = 0.5
    ymin_ylog = np.min(bj_probs[bj_probs > 0])

if surfq is None:
    legend_title = ""
else:
    legend_title = r"$\sigma_s = \pm %.2f$ $e$/nm$^2$" % surfq + "\n"
legend_title = (
    legend_title
    + r"$n_{EO} = %d$, " % Sim.O_per_chain
    + r"$r = %.4f$" % Sim.Li_O_ratio
    + "\n"
    + "Bin Number"
)

cmap = plt.get_cmap()
c_vals = np.arange(n_states)
c_norm = max(1, n_states - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for six, state_ix in enumerate(states):
        ax.plot(
            times,
            bj_probs[six],
            label=r"$%d$" % (state_ix + 1),
            alpha=leap.plot.ALPHA,
            rasterized=not args.continuous,
        )
    ax.set(
        xlabel="Lag Time / ns",
        ylabel="Back-Jump Probability",
        xlim=(times[0], xmax_ylin),
        ylim=(0, ymax),
    )
    legend = ax.legend(
        title=legend_title,
        loc="upper right",
        ncol=1 + n_states // (4 + 1),
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()

    # Log scale x.
    ax.set_xlim(times[1], xmax_ylin)
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()

    # Log scale y.
    ax.relim()
    ax.autoscale()
    ax.set_xscale("linear")
    ax.set_xlim(times[0], xmax_ylog)
    ax.set_ylim(ymin_ylog, ymax)
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()

    # Log scale xy.
    ax.plot(
        times_fit,
        bj_prob_fit * 2,
        color="black",
        linestyle="dashed",
        alpha=leap.plot.ALPHA,
    )
    ax.text(
        times_fit[-1],
        bj_prob_fit[-1] * 2,
        r"$\propto t^{%.2f}$" % popt[1],
        rotation=np.rad2deg(np.arctan(popt[1])) / 1.2,
        rotation_mode="anchor",
        transform_rotates_text=False,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize="small",
    )
    ax.set_xlim(times[1], xmax_ylog)
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
