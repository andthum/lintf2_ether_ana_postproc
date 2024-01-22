#!/usr/bin/env python3

"""
Calculate and plot the number of compounds in each bin for a single
simulation.
"""


# Standard libraries
import argparse

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

# First-party libraries
import lintf2_ether_ana_postproc as leap


def annotate_bins(ax, bins):
    """
    Annotate a :class:`matplotlib.axes.Axes` with bin numbers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The :class:`~matplotlib.axes.Axes` to annotate.
    bins : array_like
        The bin edges.
    """
    bins = np.asarray(bins)
    ax.text(
        x=np.mean(ax.get_xlim()),
        y=ax.get_ylim()[1] + 0.12 * np.diff(ax.get_ylim()),
        s="Bin Number",
        fontsize="small",
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    for bin_num, bin_mid in enumerate(bins[1:] - np.diff(bins) / 2, start=1):
        if bin_num % 2 == 0:
            y_pos = ax.get_ylim()[1]
        else:
            y_pos = ax.get_ylim()[1] + 0.06 * np.diff(ax.get_ylim())
        ax.text(
            x=bin_mid,
            y=y_pos,
            s="{:^d}".format(bin_num),
            fontsize="x-small",
            horizontalalignment="center",
            verticalalignment="bottom",
        )


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Calculate and plot the number of compounds in each bin for a single"
        " simulation."
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
    choices=("Li",),  # ("Li", "NBT", "OBT", "OE"),
    help="Compound.  Default: %(default)s",
)
args = parser.parse_args()

analysis = "discrete-z"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
tool = "mdt"  # Analysis software.
outfile_base = (
    args.settings
    + "_"
    + args.system
    + "_"
    + analysis
    + "_"
    + args.cmp
    + "_bin_population"
)
outfile_txt = outfile_base + ".txt.gz"
outfile_pdf = outfile_base + ".pdf"


print("Creating Simulation instance(s)...")
if "_gra_" in args.system:
    surfq = leap.simulation.get_surfq(args.system)
    path_key = "q%g" % surfq
else:
    surfq = None
    path_key = "bulk"
set_pat = "[0-9][0-9]_" + args.settings + "_" + args.system
Sim = leap.simulation.get_sim(args.system, set_pat, path_key)


print("Reading data...")
# Read discrete trajectory.
file_suffix = analysis + analysis_suffix + "_dtrj.npz"
infile_dtrj = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
dtrj = mdt.fh.load_dtrj(infile_dtrj)
states_dtrj, n_cmps_per_bin = np.unique(dtrj, return_counts=True)
n_cmps, n_frames = dtrj.shape
del dtrj

# Read bin file.
file_suffix = analysis + analysis_suffix + "_bins.txt.gz"
infile_bins = leap.simulation.get_ana_file(Sim, analysis, tool, file_suffix)
bins = np.loadtxt(infile_bins)  # Bin edges in Angstrom.
states = np.arange(len(bins) - 1)
if not np.all(np.isin(states_dtrj, states)):
    raise ValueError(
        "The state indices of the discrete trajectory do no match to the bins"
    )

# Set the number of compounds for states that are not contained in the
# discrete trajectory to zero.
for state in states:
    if state not in states_dtrj:
        n_cmps_per_bin = np.insert(n_cmps_per_bin, state, 0)
n_cmps_per_bin = n_cmps_per_bin / n_frames

# Calculate the compound density in each bin.
box_x, box_y, box_z = Sim.box[:3]  # Box dimensions in Angstrom.
bin_widths = np.diff(bins)
bin_volumes = bin_widths * box_x * box_y
cmp_dens_per_bin = n_cmps_per_bin / bin_volumes


print("Creating output file(s)...")
Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK
bins_dist_left = np.round(bins[:-1] - elctrd_thk, 6)
bins_dist_right = np.round(box_z - elctrd_thk - bins[1:], 6)
data = np.column_stack(
    [
        bins[:-1],
        bins[1:],
        bins_dist_left,
        bins_dist_right,
        bin_volumes,
        n_cmps_per_bin,
        cmp_dens_per_bin,
    ]
)
header = (
    "Bin population (number of compounds in each bin).\n"
    + "\n\n"
    + "System:   {:s}\n".format(args.system)
    + "Settings: {:s}\n".format(args.settings)
    + "Compound: {:s}\n".format(args.cmp)
    + "Bin file:            {:s}\n".format(infile_dtrj)
    + "Discrete trajectory: {:s}\n".format(infile_dtrj)
    + "Number of compounds: {:d}\n".format(n_cmps)
    + "Number of frames:    {:d}\n".format(n_frames)
    + "\n"
    + "box_x:           {:>16.9e}A\n".format(box_x)
    + "box_y:           {:>16.9e}A\n".format(box_y)
    + "box_z:           {:>16.9e}A\n".format(box_z)
    + "Lower electrode: {:>16.9e}A\n".format(elctrd_thk)
    + "Upper electrode: {:>16.9e}A\n".format(box_z - elctrd_thk)
    + "\n\n"
    + "The columns contain:\n"
    + "  1 Lower bin edges / A\n"
    + "  2 Upper bin edges / A\n"
    + "  3 Distance of the lower bin edges to the lower electrode / A\n"
    + "  4 Distance of the upper bin edges to the upper electrode / A\n"
    + "  5 Bin volume / A^3\n"
    + "  6 Number of compounds per bin\n"
    + "  7 Number density of the compound per bin / A^{-3}\n"
    + "\n"
    + "Column number:\n"
    + "{:>14d}".format(1)
)
for col_num in range(2, data.shape[1] + 1):
    header += " {:>16d}".format(col_num)
leap.io_handler.savetxt(outfile_txt, data, header=header)
print("Created {}".format(outfile_txt))


print("Creating plot(s)...")
box_z /= 10  # Angstrom -> nm.
elctrd_thk /= 10  # Angstrom -> nm.
bins /= 10  # Angstrom -> nm.
bin_mids = bins[1:] - np.diff(bins) / 2
bin_volumes /= 1e3  # Angstrom^3 -> nm^3.
cmp_dens_per_bin *= 1e3  # 1/Angstrom^3 -> 1/nm^3.

xlabel = r"$z$ / nm"
xlim = (0, box_z)
ydata = (bin_volumes, n_cmps_per_bin, cmp_dens_per_bin)
ylabels = (
    r"Bin Volume / nm$^3$",
    (r"$N_{%s}$ per Bin" % leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]),
    (
        r"Density $\rho_{%s}$ per Bin / nm$^{-3}$"
        % leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
    ),
)

if surfq is None:
    legend_title = ""
else:
    legend_title = r"$\sigma_s = \pm %.2f$ $e$/nm$^2$" % surfq + "\n"
legend_title = (
    legend_title
    + r"$r = %.4f$" % Sim.Li_O_ratio
    + "\n"
    + r"$n_{EO} = %d$" % Sim.O_per_chain
)
legend_locs = ("lower center", "upper center", "upper center")

mdt.fh.backup(outfile_pdf)
with PdfPages(outfile_pdf) as pdf:
    for dat_ix, ydat in enumerate(ydata):
        fig, ax = plt.subplots(clear=True)
        if surfq is not None:
            leap.plot.elctrds(
                ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
            )
        if surfq is None:
            ax.plot(bin_mids, ydat, marker="o")
        else:
            # Remove first and last data points, because these lie
            # behind the electrodes.
            ax.plot(bin_mids[1:-1], ydat[1:-1], marker="o")
        ax.set(xlabel=xlabel, ylabel=ylabels[dat_ix], xlim=xlim)
        ylim = ax.get_ylim()
        if ylim[0] < 0:
            ax.set_ylim(0, ylim[1])
        leap.plot.bins(ax, bins)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        annotate_bins(ax, bins)
        legend = ax.legend(
            title=legend_title,
            loc=legend_locs[dat_ix],
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()
        plt.close()

print("Created {}".format(outfile_pdf))
print("Done")
