#!/usr/bin/env python3


r"""
Plot free-energy profiles for different chain lengths.

The free-energy profiles are calculated from the density profiles
according to

.. math::

    F(z) = -k_B T \ln\left[ \frac{\rho(z)}{\rho^\circ} \right]

where :math:`\rho^\circ` is the average density in the bulk region (i.e.
the free energy in the bulk region is effectively set to zero).
"""


# Standard libraries
import argparse
import glob
import os

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description="Plot free-energy profiles for different chain lengths."
)
parser.add_argument(
    "--surfq",
    type=str,
    required=True,
    choices=("q0", "q0.25", "q0.5", "q0.75", "q1"),
    help="Surface charge in e/nm^2.",
)
parser.add_argument(
    "--common-ylim",
    required=False,
    default=False,
    action="store_true",
    help="Use common y limits for all plots.",
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "density-z"  # Analysis name.
analysis_suffix = "_number"  # Analysis name specification.
tool = "gmx"  # Analysis software.
outfile = (  # Output file name.
    settings + "_lintf2_peoN_20-1_gra_" + args.surfq + "_sc80_free_energy"
)
if args.common_ylim:
    outfile += "_common_ylim.pdf"
else:
    outfile += ".pdf"

cols = (  # Columns to read from the input file(s).
    # 0,  # bin edges [nm]
    2,  # Li number density [nm^-3]
    5,  # NBT number density [nm^-3]
    6,  # OBT number density [nm^-3]
    7,  # OE number density [nm^-3]
)
compounds = ("Li", "NBT", "OBT", "OE")
if len(compounds) != len(cols):
    raise ValueError(
        "`len(compounds)` ({}) != `len(cols)`"
        " ({})".format(len(compounds), len(cols))
    )


print("Creating Simulation instance(s)...")
SimPaths = leap.simulation.SimPaths()
pattern_system = "lintf2_[gp]*[0-9]*_20-1_gra_" + args.surfq + "_sc80"
pattern_settings = "[0-9][0-9]_" + settings + "_" + pattern_system
pattern = os.path.join(
    SimPaths.PATHS[args.surfq], pattern_system, pattern_settings
)
paths = glob.glob(pattern)
Sims = leap.simulation.Simulations(*paths, sort_key="O_per_chain")


print("Assembling input file name(s)...")
infiles = []
file_suffix = analysis + analysis_suffix + ".xvg.gz"
for i, path in enumerate(Sims.paths_ana):
    fname = Sims.fnames_ana_base[i] + file_suffix
    fpath = os.path.join(path, tool, analysis, fname)
    if not os.path.isfile(fpath):
        raise FileNotFoundError("No such file: '{}'".format(fpath))
    infiles.append(fpath)
n_infiles = len(infiles)


print("Reading data and creating plot(s)...")
Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm
box_z_max = np.max(Sims.boxes_z)

plot_sections = ("left", "right", "full")
xmin = 0
xmax = Elctrd.BULK_START / 10  # A -> nm
if args.surfq == "q0":
    ymin = tuple((-2, -1.5, -1, -1.1) for _ in plot_sections)
    ymax = tuple((2.6, 2.3, 1.2, 0.9) for _ in plot_sections)
elif args.surfq == "q0.25":
    ymin = ((-1.4, -2.2, -2.4, -1.1), (-2.6, -0.8, -0.6, -1.6))
    ymin += (tuple(np.min(ymin, axis=0)),)
    ymax = ((1.8, 2.8, 1.2, 1.3), (2.6, 2.8, 3.6, 0.6))
    ymax += (tuple(np.max(ymax, axis=0)),)
elif args.surfq == "q0.5":
    ymin = ((-1.6, -2.75, -3.2, -1.4), (-3.5, -1.25, -0.8, -1.8))
    ymin += (tuple(np.min(ymin, axis=0)),)
    ymax = ((2.2, 3.5, 1.4, 2), (4, 0.6, 0.4, 0.6))
    ymax += (tuple(np.max(ymax, axis=0)),)
elif args.surfq == "q0.75":
    ymin = ((-2, -3.5, -3.8, -1.6), (-3.5, -1.8, -1.4, -2))
    ymin += (tuple(np.min(ymin, axis=0)),)
    ymax = ((5, 4.5, 2, 2.8), (4.75, 4, 4.2, 0.8))
    ymax += (tuple(np.max(ymax, axis=0)),)
elif args.surfq == "q1":
    ymin = ((-2, -4.5, -4.5, -1.75), (-5, -1.8, -1.8, -2))
    ymin += (tuple(np.min(ymin, axis=0)),)
    ymax = ((5.5, 8.5, 3.5, 4.75), (5.5, 2.2, 0.8, 2.4))
    ymax += (tuple(np.max(ymax, axis=0)),)
else:
    raise ValueError("Unknown surface charge --surfq: '{}'".format(args.surfq))
if args.common_ylim:
    # ymin = tuple(
    #     tuple(None for _cmp in compounds)
    #     for _plt_sec in plot_sections
    # )
    # ymax = tuple(
    #     tuple(None for _cmp in compounds)
    #     for _plt_sec in plot_sections
    # )
    # ymin = tuple(
    #     tuple(-6 for _cmp in compounds) for _plt_sec in plot_sections
    # )
    # ymax = tuple(
    #     tuple(8.5 for _cmp in compounds) for _plt_sec in plot_sections
    # )
    ymin = tuple((-6, -6, -6, -2.2) for _plt_sec in plot_sections)
    ymax = tuple((7.5, 9, 5.5, 5.5) for _plt_sec in plot_sections)

cmap = plt.get_cmap()
mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for ps_ix, plt_sec in enumerate(plot_sections):
        for cmp_ix, cmp in enumerate(compounds):
            fig, ax = plt.subplots(clear=True)
            ax.set_prop_cycle(
                color=[cmap(i / (n_infiles - 1)) for i in range(n_infiles)]
            )

            if plt_sec in ("left", "right"):
                # Also, for the plot of the right electrode, the
                # electrode position will be shifted to zero.
                leap.plot.elctrd_left(ax)

            for sim_ix, Sim in enumerate(Sims.sims):
                x, y = np.loadtxt(
                    infiles[sim_ix],
                    comments=["#", "@"],
                    usecols=(0, cols[cmp_ix]),
                    unpack=True,
                )
                y = leap.misc.dens2free_energy(
                    x, y, bulk_region=Sim.bulk_region / 10  # A -> nm
                )
                if plt_sec == "left":
                    y = y[: len(y) // 2]
                    x = x[: len(x) // 2]
                    x -= elctrd_thk
                elif plt_sec == "right":
                    y = y[len(y) // 2 :]
                    x = x[len(x) // 2 :]
                    x += elctrd_thk
                    x -= Sim.box[2] / 10  # A -> nm
                    x *= -1  # Ensure positive x-axis.

                if args.surfq == "q0" and Sim.O_per_chain == 6 and cmp == "Li":
                    color = "tab:red"
                    ax.plot([], [])  # Increment color cycle.
                elif (
                    args.surfq == "q1"
                    and Sim.O_per_chain == 2
                    and cmp in ("NBT", "OBT")
                    and plt_sec == "right"
                ):
                    color = "tab:red"
                    ax.plot([], [])  # Increment color cycle.
                else:
                    color = None
                ax.plot(
                    x,
                    y,
                    label=r"$%d$" % Sim.O_per_chain,
                    color=color,
                    alpha=leap.plot.ALPHA,
                )

            if plt_sec == "left":
                xlabel = r"Distance to Electrode / nm"
                xlim = (xmin, xmax)
            elif plt_sec == "right":
                xlabel = r"Distance to Electrode / nm"
                # Reverse x-axis.
                xlim = (xmax, xmin)
            else:
                xlabel = r"$z$ / nm"
                xlim = (0, box_z_max)
            ax.set(
                xlabel=xlabel,
                ylabel=(
                    r"Free Energy $F_{"
                    + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp]
                    + r"}$ / $k_B T$"
                ),
                xlim=xlim,
                ylim=(ymin[ps_ix][cmp_ix], ymax[ps_ix][cmp_ix]),
            )

            # Equalize x- and y-ticks so that plots can be stacked
            # together.
            xlim_diff = np.diff(ax.get_xlim())
            if xlim_diff > 2.5 and xlim_diff < 5:
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.xaxis.set_minor_locator(MultipleLocator(0.2))
            ylim_diff = np.diff(ax.get_ylim())
            if ylim_diff > 10 and ylim_diff < 20:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if not args.common_ylim:
                if all(np.abs(ax.get_ylim()) < 10) and ylim_diff > 2:
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

            legend_title = (
                r"%.2f$" % Sims.surfqs[0]
                + r" $e$/nm$^2$"
                + "\n"
                + r"$r = %.2f$" % Sims.Li_O_ratios[0]
                + "\n"
                + r"$n_{EO}$"
            )
            if plt_sec == "left":
                if args.surfq == "q0":
                    legend_title = r"$\sigma_s = " + legend_title
                else:
                    legend_title = r"$\sigma_s = +" + legend_title
                legend_loc = "right"
            elif plt_sec == "right":
                if args.surfq == "q0":
                    legend_title = r"$\sigma_s = " + legend_title
                else:
                    legend_title = r"$\sigma_s = -" + legend_title
                legend_loc = "left"
            else:
                if args.surfq == "q0":
                    legend_title = r"$\sigma_s = " + legend_title
                else:
                    legend_title = r"$\sigma_s = \pm" + legend_title
                legend_loc = "center"
            if abs(ax.get_ylim()[1]) > abs(ax.get_ylim()[0]):
                legend_loc = "upper " + legend_loc
            else:
                legend_loc = "lower " + legend_loc
            legend = ax.legend(
                title=legend_title,
                ncol=1 + n_infiles // (4 + 1),
                loc=legend_loc,
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend.get_title().set_multialignment("center")

            pdf.savefig()
            plt.close()

print("Created {}".format(outfile))
print("Done")
