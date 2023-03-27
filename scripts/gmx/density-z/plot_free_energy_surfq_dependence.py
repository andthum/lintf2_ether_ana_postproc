#!/usr/bin/env python3


r"""
Plot free-energy profiles for different surface charges.

The free-energy profiles is calculated from the density profiles
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
    description="Plot free energy profiles for different surface charges."
)
parser.add_argument(
    "--sol",
    type=str,
    required=True,
    choices=("g1", "g4", "peo63"),
    help="Solvent name.",
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
    settings + "_lintf2_" + args.sol + "_20-1_gra_qX_sc80_free_energy"
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
# Distance to the electrodes in nm at which the bulk region starts.
bulk_start = 4


print("Creating Simulation instances...")
SimPaths = leap.simulation.SimPaths()
pattern_system = "lintf2_" + args.sol + "_20-1_gra_q[0-9]*_sc80"
pattern_settings = "[0-9][0-9]_" + settings + "_" + pattern_system
pattern = os.path.join(
    SimPaths.PATHS["walls"], "q[0-9]*", pattern_system, pattern_settings
)
paths = glob.glob(pattern)
Sims = leap.simulation.Simulations(*paths, sort_key="surfq")


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
if args.sol == "g1":
    ymin = ((-2, -4.5, -4.5, -1.5), (-5, -2, -2, -1.9))
    ymin += (tuple(np.min(ymin, axis=0)),)
    ymax = ((3.6, 8.5, 3.5, 5), (5, 4.5, 4.5, 1))
    ymax += (tuple(np.max(ymax, axis=0)),)
elif args.sol == "g4":
    ymin = ((-1.5, -4, -4.1, -1.6), (-4, -2, -1.4, -1.8))
    ymin += (tuple(np.min(ymin, axis=0)),)
    ymax = ((4.5, 7.5, 1.9, 4.4), (4, 5, 3.6, 1))
    ymax += (tuple(np.max(ymax, axis=0)),)
elif args.sol == "peo63":
    ymin = ((-1.6, -4, -4, -1.6), (-3.5, -1.5, -1, -2))
    ymin += (tuple(np.min(ymin, axis=0)),)
    ymax = ((2.2, 7, 2, 4), (5, 5.5, 3.6, 1))
    ymax += (tuple(np.max(ymax, axis=0)),)
else:
    raise ValueError("Unknown solvent --sol: '{}'".format(args.sol))
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
                leap.plot.plot_elctrd_left(ax)
            else:
                leap.plot.plot_elctrds(
                    ax,
                    offset_left=elctrd_thk,
                    offset_right=box_z_max - elctrd_thk,
                )

            for sim_ix, Sim in enumerate(Sims.sims):
                x, y = np.loadtxt(
                    infiles[sim_ix],
                    comments=["#", "@"],
                    usecols=(0, cols[cmp_ix]),
                    unpack=True,
                )
                bulk_begin = elctrd_thk + bulk_start
                bulk_end = Sim.box[2] / 10  # A -> nm
                bulk_end -= elctrd_thk + bulk_start
                y = leap.misc.dens2free_energy(
                    x, y, bulk_region=(bulk_begin, bulk_end)
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

                label = r"%.2f$" % Sims.surfqs[sim_ix]
                if plt_sec == "left":
                    label = r"$+" + label
                elif plt_sec == "right":
                    label = r"$-" + label
                else:
                    label = r"$\pm" + label
                ax.plot(x, y, label=label, alpha=leap.plot.ALPHA)

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
                    + leap.plot.atom_type2display_name[cmp]
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
                r"$n_{EO} = %d$" % Sims.O_per_chain[0]
                + "\n"
                + r"$r = %.2f$" % Sims.Li_O_ratios[0]
                + "\n"
                + r"$\sigma_s$ / $e$/nm$^2$"
            )
            if plt_sec == "left":
                legend_loc = "right"
            elif plt_sec == "right":
                legend_loc = "left"
            else:
                legend_loc = "center"
            if None in (ymax[ps_ix][cmp_ix], ymin[ps_ix][cmp_ix]):
                legend_loc = "upper " + legend_loc
            elif abs(ymax[ps_ix][cmp_ix]) > abs(ymin[ps_ix][cmp_ix]):
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
