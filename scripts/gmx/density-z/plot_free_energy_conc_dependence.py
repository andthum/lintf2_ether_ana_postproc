#!/usr/bin/env python3


r"""
Plot free-energy profiles for different salt concentrations.

The free-energy profiles are calculated from the density profiles
according to

.. math::

    F(z) = -k_B T \ln\left[ \frac{\rho(z)}{\rho^\circ} \right]

where :math:`\rho^\circ` is the average density in the bulk region (i.e.
the free energy in the bulk region is effectively set to zero).
"""


# Standard libraries
import argparse

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
    description="Plot free-energy profiles for different salt concentrations."
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
    settings
    + "_lintf2_"
    + args.sol
    + "_r_gra_"
    + args.surfq
    + "_sc80_free_energy"
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
sys_pat = "lintf2_" + args.sol + "_[0-9]*-[0-9]*_gra_" + args.surfq + "_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, args.surfq, sort_key="Li_O_ratio"
)


print("Reading data and creating plot(s)...")
file_suffix = analysis + analysis_suffix + ".xvg.gz"
infiles = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
n_infiles = len(infiles)

Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm
box_z_max = np.max(Sims.boxes_z)

plot_sections = ("left", "right", "full")
xmin = 0
xmax = Elctrd.BULK_START / 10  # A -> nm
if args.sol == "g1":
    if args.surfq == "q0":
        ymin = tuple((-2.2, -2, -1.2, -1.4) for _ in plot_sections)
        ymax = tuple((1.8, 1.7, 0.7, 0.8) for _ in plot_sections)
    elif args.surfq == "q1":
        ymin = ((-5.5, -10, -10, -1.5), (-10, -5, -4.2, -2.2))
        ymin += (tuple(np.min(ymin, axis=0)),)
        ymax = ((3.5, 8.5, 4, 5), (7.5, 3.5, 1.4, 3.4))
        ymax += (tuple(np.max(ymax, axis=0)),)
    else:
        raise ValueError(
            "Unknown surface charge --surfq: '{}'".format(args.surfq)
        )
elif args.sol == "g4":
    if args.surfq == "q0":
        ymin = tuple((-1.7, -1.5, -1.5, -1.2) for _ in plot_sections)
        ymax = tuple((2.4, 2.2, 2.2, 1.6) for _ in plot_sections)
    elif args.surfq == "q1":
        ymin = ((-4, -8, -8, -2), (-8, -3, -2.5, -2.2))
        ymin += (tuple(np.min(ymin, axis=0)),)
        ymax = ((3.5, 9, 3.5, 5.5), (6, 5.5, 5.5, 3.2))
        ymax += (tuple(np.max(ymax, axis=0)),)
    else:
        raise ValueError(
            "Unknown surface charge --surfq: '{}'".format(args.surfq)
        )
elif args.sol == "peo63":
    if args.surfq == "q0":
        ymin = tuple((-1.8, -1.6, -1.2, -1.2) for _ in plot_sections)
        ymax = tuple((2.6, 2.6, 1.7, 1) for _ in plot_sections)
    elif args.surfq == "q1":
        ymin = ((-3, -7.5, -7.5, -1.75), (-7, -2, -1.8, -2.2))
        ymin += (tuple(np.min(ymin, axis=0)),)
        ymax = ((2, 8.5, 3.5, 5.5), (4, 3.4, 3.8, 1.8))
        ymax += (tuple(np.max(ymax, axis=0)),)
    else:
        raise ValueError(
            "Unknown surface charge --surfq: '{}'".format(args.surfq)
        )
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

                if args.surfq == "q1" and np.isclose(
                    Sim.Li_O_ratio, 1 / 80, rtol=0
                ):
                    linestyle = "dotted"
                else:
                    linestyle = "solid"
                ax.plot(
                    x,
                    y,
                    label="$%.4f$" % Sim.Li_O_ratio,
                    linestyle=linestyle,
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
                + r"$n_{EO} = %d$" % Sims.O_per_chain[0]
                + "\n"
                + r"$r$"
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
