#!/usr/bin/env python3


"""Plot density profiles"""


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

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description="Plot density profiles for different chain lengths."
)
parser.add_argument(
    "-q",
    type=str,
    required=True,
    choices=("q0", "q0.25", "q0.5", "q0.75", "q1"),
    help="Surface charge in e/nm^2.",
)
args = parser.parse_args()

common_ymax = True
settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "density-z"  # Analysis name.
analysis_suffix = "_number"  # Analysis name specification.
tool = "gmx"  # Analysis software.
outfile = (  # Output file name.
    settings
    + "_lintf2_peoN_20-1_gra_"
    + args.q
    + "_sc80_"
    + analysis
    + analysis_suffix
)
if common_ymax:
    outfile += "_common_ymax.pdf"
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


print("Creating Simulation instances...")
SimPaths = leap.simulation.SimPaths()
pattern_system = "lintf2_*[0-9]*_20-1_gra_q[0-9]*_sc80"
pattern_settings = "[0-9][0-9]_" + settings + "_" + pattern_system
pattern = os.path.join(
    SimPaths.PATHS[args.q], pattern_system, pattern_settings
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

plot_sections = ("left", "right", "full")
# xmin = -elctrd_thk
xmin = 0
xmax = 4
if args.q == "q0":
    ymax = tuple((6.25, 4, 3, 3) for _ in plot_sections)
elif args.q == "q0.25":
    ymax = ((4, 7, 10.5, 2.8), (12.5, 2, 1.8, 4.8))
    ymax += (tuple(np.max(ymax, axis=0)),)
elif args.q == "q0.5":
    ymax = ((4.2, 11.5, 20, 4), (25, 3.1, 2, 6))
    ymax += (tuple(np.max(ymax, axis=0)),)
elif args.q == "q0.75":
    ymax = ((4, 23, 36, 4.6), (23, 4.6, 3.1, 7))
    ymax += (tuple(np.max(ymax, axis=0)),)
elif args.q == "q1":
    ymax = ((4.6, 46, 62.5, 4.6), (82.5, 4.6, 4.6, 6.25))
    ymax += (tuple(np.max(ymax, axis=0)),)
else:
    raise ValueError("Unknown surface charge -q: '{}'".format(args.q))
if common_ymax:
    # ymax = tuple(
    #     tuple(None for _cmp in compounds)
    #     for _plt_sec in plot_sections
    # )
    ymax = tuple(
        tuple(6.25 for _cmp in compounds) for _plt_sec in plot_sections
    )

cmap = plt.get_cmap()
mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for y_normed in (True, False):
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

                for sim_ix, Sim in enumerate(Sims.sims):
                    x, y = np.loadtxt(
                        infiles[sim_ix],
                        comments=["#", "@"],
                        usecols=(0, cols[cmp_ix]),
                        unpack=True,
                    )
                    if plt_sec == "left":
                        x -= elctrd_thk
                    elif plt_sec == "right":
                        x += elctrd_thk
                        x -= Sim.box[2] / 10  # A -> nm
                        x *= -1  # Ensure positive x-axis.
                    if y_normed:
                        bulk_dens = Sim.dens["atm_type"][cmp]["num"]
                        bulk_dens *= 1e3  # 1/A^3 -> 1/nm^3
                        y /= bulk_dens
                    if args.q == "q0" and Sim.O_per_chain == 6 and cmp == "Li":
                        color = "tab:red"
                        ax.plot([], [])  # Increment color cycle.
                    elif (
                        args.q == "q1"
                        and Sim.O_per_chain == 2
                        and cmp in ("NBT", "OBT")
                    ):
                        color = "tab:red"
                        ax.plot([], [])  # Increment color cycle.
                    else:
                        color = None
                    ax.plot(
                        x,
                        y,
                        label=str(Sim.O_per_chain),
                        color=color,
                        linewidth=1.5,
                        alpha=2 / 3,
                    )

                ylabel = (
                    r"Density $\rho_{"
                    + leap.plot.atom_type2display_name[cmp]
                    + r"}"
                )
                if y_normed:
                    ylabel += (
                        r" / \rho_{"
                        + leap.plot.atom_type2display_name[cmp]
                        + r"}^{bulk}$"
                    )
                    ylim = (0, ymax[ps_ix][cmp_ix])
                else:
                    ylabel += r"$ / nm$^{-3}$"
                    ylim = (0, None)
                if plt_sec == "left":
                    xlabel = r"Distance to Electrode / nm"
                    xlim = (xmin, xmax)
                elif plt_sec == "right":
                    xlabel = r"Distance to Electrode / nm"
                    # Reverse x-axis.
                    xlim = (xmax, xmin)
                else:
                    xlabel = r"z / nm"
                    xlim = (0, np.max(Sims.boxes_z))
                ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)

                legend_title = (
                    r"%.2f$" % Sims.surfqs[0]
                    + r" $e$/nm$^2$"
                    + "\n"
                    + r"$n_{EO}$"
                )
                if plt_sec == "left":
                    if args.q == "q0":
                        legend_title = r"$\sigma_s = " + legend_title
                    else:
                        legend_title = r"$\sigma_s = +" + legend_title
                    legend_loc = "upper right"
                elif plt_sec == "right":
                    if args.q == "q0":
                        legend_title = r"$\sigma_s = " + legend_title
                    else:
                        legend_title = r"$\sigma_s = -" + legend_title
                    legend_loc = "upper left"
                else:
                    if args.q == "q0":
                        legend_title = r"$\sigma_s = " + legend_title
                    else:
                        legend_title = r"$\sigma_s = \pm" + legend_title
                    legend_loc = "upper center"
                legend = ax.legend(
                    title=legend_title,
                    ncol=1 + n_infiles // (6 + 1),
                    loc=legend_loc,
                    **mdtplt.LEGEND_KWARGS_XSMALL,
                )
                legend.get_title().set_multialignment("center")

                pdf.savefig()
                plt.close()

print("Done")
