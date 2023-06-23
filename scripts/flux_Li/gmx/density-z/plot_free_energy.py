#!/usr/bin/env python3


r"""
Plot free-energy profiles for a Non-Equilibrium Molecular Dynamics
(NEMD) simulation and its equilibrium counterpart(s) (EQMD).

The free-energy profiles is calculated from the density profiles
according to

.. math::

    F(z) = -k_B T \ln\left[ \frac{\rho(z)}{\rho^\circ} \right]

where :math:`\rho^\circ` is the average density in the bulk region (i.e.
the free energy in the bulk region is effectively set to zero).
"""


# Standard libraries
import argparse
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


def equalize_xticks(ax):
    """
    Equalize x-ticks so that plots can be better stacked together.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` for which to equalize the x
        ticks.
    """
    xlim = np.asarray(ax.get_xlim())
    xlim_diff = xlim[-1] - xlim[0]
    if xlim_diff > 2.5 and xlim_diff < 5:
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))


def equalize_yticks(ax):
    """
    Equalize y-ticks so that plots can be better stacked together.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` for which to equalize the y
        ticks.

    Notes
    -----
    This function relies on global variables!
    """
    ylim = np.asarray(ax.get_ylim())
    ylim_diff = ylim[-1] - ylim[0]
    yticks = np.asarray(ax.get_yticks())
    yticks_valid = (yticks >= ylim[0]) & (yticks <= ylim[-1])
    yticks = yticks[yticks_valid]
    if ylim_diff >= 10 and ylim_diff < 20:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if not args.common_ylim:
        if np.all(yticks >= 0) and np.all(yticks < 10) and ylim_diff > 2:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot free-energy profiles for a NEMD simulation and its EQMD"
        " counterpart(s)"
    ),
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
base_name_nemd = (  # NEMD base name.
    settings + "_lintf2_" + args.sol + "_20-1_gra_q1_sc80_flux_"
)
outfile = base_name_nemd + "free_energy"  # Output file name.
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
sys_pat = "lintf2_" + args.sol + "_20-1_gra_q[0-9]*_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
sys_pat = os.path.join("q[0-9]*", sys_pat)
Sims = leap.simulation.get_sims(sys_pat, set_pat, "walls", sort_key="surfq")


print("Reading data and creating plot(s)...")
file_suffix = analysis + analysis_suffix + ".xvg.gz"
infiles = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
n_eqmd_sims = len(Sims.sims)
infile_nemd = base_name_nemd + analysis + analysis_suffix + ".xvg.gz"
infiles.append(infile_nemd)
n_infiles = len(infiles)

Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm
box_z = Sims.boxes_z[0]
if not np.allclose(Sims.boxes_z, box_z, rtol=0):
    raise ValueError("The simulations have different z box lengths")
bulk_region = Sims.bulk_regions[0]

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
    ymin = tuple((-6, -6, -6, -2.2) for _plt_sec in plot_sections)
    ymax = tuple((7.5, 9, 5.5, 5.5) for _plt_sec in plot_sections)

cmap = plt.get_cmap()
c_vals = np.arange(n_eqmd_sims)
c_norm = n_eqmd_sims - 1
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for ps_ix, plt_sec in enumerate(plot_sections):
        for cmp_ix, cmp in enumerate(compounds):
            fig, ax = plt.subplots(clear=True)
            ax.set_prop_cycle(color=colors)

            if plt_sec in ("left", "right"):
                # Also, for the plot of the right electrode, the
                # electrode position will be shifted to zero.
                leap.plot.elctrd_left(ax)
            else:
                leap.plot.elctrds(
                    ax,
                    offset_left=elctrd_thk,
                    offset_right=box_z - elctrd_thk,
                )

            for sim_ix, infile in enumerate(infiles):
                x, y = np.loadtxt(
                    infile,
                    comments=["#", "@"],
                    usecols=(0, cols[cmp_ix]),
                    unpack=True,
                )
                y = leap.misc.dens2free_energy(
                    x, y, bulk_region=bulk_region, tol=0.05
                )
                if plt_sec == "left":
                    y = y[: len(y) // 2]
                    x = x[: len(x) // 2]
                    x -= elctrd_thk
                elif plt_sec == "right":
                    y = y[len(y) // 2 :]
                    x = x[len(x) // 2 :]
                    x += elctrd_thk
                    x -= box_z
                    x *= -1  # Ensure positive x-axis.

                if sim_ix < n_infiles - 1:
                    if plt_sec == "left":
                        label = r"$+"
                    elif plt_sec == "right":
                        label = r"$-"
                    else:
                        label = r"$\pm"
                    label += r"%.2f$" % Sims.surfqs[sim_ix]
                    color = None
                    marker = None
                else:
                    label = "NEMD"
                    color = "tab:red"
                    marker = "."
                ax.plot(
                    x,
                    y,
                    label=label,
                    color=color,
                    marker=marker,
                    alpha=leap.plot.ALPHA,
                )

            if plt_sec == "left":
                xlabel = r"Distance to Electrode / nm"
                xlim = (xmin, xmax)
            elif plt_sec == "right":
                xlabel = r"Distance to Electrode / nm"
                xlim = (xmax, xmin)  # Reverse x-axis.
            else:
                xlabel = r"$z$ / nm"
                xlim = (0, box_z)
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
            equalize_xticks(ax)
            equalize_yticks(ax)

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
