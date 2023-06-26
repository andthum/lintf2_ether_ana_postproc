#!/usr/bin/env python3


r"""
Plot the charge density, the electric field and the electric potential
as function of the distance to the electrodes for different salt
concentrations.

Notes
-----
The electric field :math:`E(z)` and potential :math:`\phi(z)` are
calculated from the charge density :math:`\rho(z)` by integrating
Poisson's equation

.. math::

    \frac{\text{d}^2 \phi(z)}{\text{d}z^2} =
    -\frac{\rho(z)}{\epsilon_r \epsilon_0}

where :math:`\epsilon_r` is the relative permittivity of the medium and
:math:`\epsilon_0` is the vacuum permittivity.

Given a charge density :math:`\rho(z)`, the resulting electric field can
be calculated by:

.. math::

    E(z) = -\frac{\text{d} \phi(z)}{\text{d}z}
    = -\int{\frac{\text{d}^2 \phi(z)}{\text{d}z^2} \text{ d}z} - c
    = \frac{1}{\epsilon_r \epsilon_0} \int{\rho(z) \text{ d}z} - c

The integration constant :math:`c` is determined by the boundary
conditions.

Integrating the electric field (or integrating Poisson's equation twice)
yields the electric potential:

.. math::

    \phi(z) = -\int{E(z) \text{ d}z} - k
    = -\int{
        \int{
            \frac{\rho(z)}{\epsilon_r \epsilon_0}
        \text{ d}z} - c
    \text{ d}z} - k
    = -\frac{1}{\epsilon_r \epsilon_0}
    \int{\int{\rho(z) \text{ d}z} \text{ d}z}
    + cz - k

As above, the integration constant :math:`k` is determined by the
boundary conditions.

In this script, the boundary conditions are chosen such that the
electric field and the electric potential are zero in the bulk region.
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
        "Plot the charge density, the electric field and the electric"
        " potential as function of the distance to the electrodes for"
        " different salt concentrations."
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
analysis_suffix = "_charge"  # Analysis name specification.
tool = "gmx"  # Analysis software.
outfile = (  # Output file name.
    settings
    + "_lintf2_"
    + args.sol
    + "_r_gra_"
    + args.surfq
    + "_sc80_"
    + analysis
    + analysis_suffix
    + "_potential"
)
if args.common_ylim:
    outfile += "_common_ylim.pdf"
else:
    outfile += ".pdf"


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

# ylabels = (
#     r"Charge Density $\rho_q$ / $e$ nm$^{-3}$",
#     r"Electric Field $\epsilon_r E$ / V nm$^{-1}$",
#     r"Electric Potential $\epsilon_r \phi$ / V",
# )
ylabels = (
    r"Charge Density $\rho_q$ / $e$ nm$^{-3}$",
    r"Electric Field $E$ / V nm$^{-1}$",
    r"Electric Potential $\phi$ / V",
)
if args.sol == "g1":
    if args.surfq == "q0":
        ymin = ((None, None, None), (None, None, None))
        ymin += ((None, None, None),)
        ymax = ((None, None, None), (None, None, None))
        ymax += ((None, None, None),)
    elif args.surfq == "q1":
        ymin = ((-60, None, None), (-30, None, None))
        ymin += ((None, None, None),)
        ymax = ((55, None, None), (45, None, None))
        ymax += ((None, None, None),)
    else:
        raise ValueError(
            "Unknown surface charge --surfq: '{}'".format(args.surfq)
        )
elif args.sol == "g4":
    if args.surfq == "q0":
        ymin = ((None, None, None), (None, None, None))
        ymin += ((None, None, None),)
        ymax = ((None, None, None), (None, None, None))
        ymax += ((None, None, None),)
    elif args.surfq == "q1":
        ymin = ((-65, None, None), (-35, None, None))
        ymin += ((None, None, None),)
        ymax = ((55, None, None), (45, None, None))
        ymax += ((None, None, None),)
    else:
        raise ValueError(
            "Unknown surface charge --surfq: '{}'".format(args.surfq)
        )
elif args.sol == "peo63":
    if args.surfq == "q0":
        ymin = ((None, None, None), (None, None, None))
        ymin += ((None, None, None),)
        ymax = ((None, None, None), (None, None, None))
        ymax += ((None, None, None),)
    elif args.surfq == "q1":
        ymin = ((-65, None, None), (-22, None, None))
        ymin += ((None, None, None),)
        ymax = ((50, None, None), (32, None, None))
        ymax += ((None, None, None),)
    else:
        raise ValueError(
            "Unknown surface charge --surfq: '{}'".format(args.surfq)
        )
else:
    raise ValueError("Unknown solvent --sol: '{}'".format(args.sol))
if args.common_ylim:
    # ymin = tuple(
    #     tuple(None for _ylabel in ylabels)
    #     for _plt_sec in plot_sections
    # )
    # ymax = tuple(
    #     tuple(None for _ylabel in ylabels)
    #     for _plt_sec in plot_sections
    # )
    ymin = tuple((-11.5, -8, -0.7) for _plt_sec in plot_sections)
    ymax = tuple((10.5, 8.5, 0.55) for _plt_sec in plot_sections)

cmap = plt.get_cmap()
c_vals = np.arange(n_infiles)
c_norm = n_infiles - 1
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for ylb_ix, ylabel in enumerate(ylabels):
        for ps_ix, plt_sec in enumerate(plot_sections):
            fig, ax = plt.subplots(clear=True)
            ax.set_prop_cycle(color=colors)

            if plt_sec in ("left", "right"):
                # Also, for the plot of the right electrode, the
                # electrode position will be shifted to zero.
                leap.plot.elctrd_left(ax)

            for sim_ix, Sim in enumerate(Sims.sims):
                x, y = np.loadtxt(
                    infiles[sim_ix],
                    comments=["#", "@"],
                    usecols=(0, 11),
                    unpack=True,
                )
                if "Field" in ylabel:
                    y = leap.misc.qdens2field(
                        x, y, bulk_region=Sim.bulk_region / 10  # A -> nm
                    )
                elif "Potential" in ylabel:
                    y = leap.misc.qdens2pot(
                        x, y, bulk_region=Sim.bulk_region / 10  # A -> nm
                    )
                elif "Density" not in ylabel:
                    raise ValueError("Unknown `ylabel`: {}".format(ylabel))
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
                xlim = (xmax, xmin)  # Reverse x-axis.
            else:
                xlabel = r"$z$ / nm"
                xlim = (0, box_z_max)
            ax.set(
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=(ymin[ps_ix][ylb_ix], ymax[ps_ix][ylb_ix]),
            )
            equalize_xticks(ax)
            equalize_yticks(ax)

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
