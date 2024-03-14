#!/usr/bin/env python3

"""
Plot the x-, y- and z-component of the mean displacement, the mean
squared displacement and the displacement variance as function of the
initial particle position at a constant diffusion time for a single
simulation.
"""


# Standard libraries
import argparse
import os

# Third-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the x-, y- and z-component of the mean displacement and the"
        " displacement variance as function of the initial particle position"
        " at a constant diffusion time for a single simulation."
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
    choices=("Li", "NTf2", "ether", "NBT", "OBT", "OE"),
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--time",
    type=float,
    required=False,
    default=1000,
    help=(
        "Diffusion time in ps for which to plot the displacements as function"
        " of the initial particle position.  If no data are present at the"
        " given diffusion time, the next nearest diffusion time for which data"
        " are present is used.  Default: %(default)s"
    ),
)
args = parser.parse_args()

analysis = "msd_layer"  # Analysis name.
analysis_suffix = "_" + args.cmp  # Analysis name specification.
ana_path = os.path.join(analysis, analysis + analysis_suffix)
tool = "mdt"  # Analysis software.
outfile = (
    args.settings
    + "_"
    + args.system
    + "_"
    + args.cmp
    + "_displvar_cross_section_xyz_"
    + "%.0fps" % args.time
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
args.time /= 1e3  # ps -> ns.
dimensions = ("x", "y", "z")
md_at_const_time = [None for dim in dimensions]
msd_at_const_time = [None for dim in dimensions]
# True MSD, not displacement variance
msd_at_const_time_true = [None for dim in dimensions]
for dim_ix, dim in enumerate(dimensions):
    (
        times_dim,
        bins_dim,
        md_data,
        msd_data_true,
    ) = leap.simulation.read_displvar_single(
        Sim, args.cmp, dim, displvar=False
    )
    # Calculate displacement variance.
    msd_data = msd_data_true - md_data**2
    if dim_ix == 0:
        times, bins = times_dim, bins_dim
    else:
        if bins_dim.shape != bins.shape:
            raise ValueError(
                "The input files do not contain the same number of bins"
            )
        if not np.allclose(bins_dim, bins, atol=0):
            raise ValueError(
                "The bin edges are not the same in all input files"
            )
        if times_dim.shape != times.shape:
            raise ValueError(
                "The input files do not contain the same number of lag times"
            )
        if not np.allclose(times_dim, times, atol=0):
            raise ValueError(
                "The lag times are not the same in all input files"
            )
    time, tix = mdt.nph.find_nearest(times, args.time, return_index=True)
    md_at_const_time[dim_ix] = md_data[tix]
    msd_at_const_time[dim_ix] = msd_data[tix]
    msd_at_const_time_true[dim_ix] = msd_data_true[tix]
bin_mids = bins[1:] - np.diff(bins) / 2
del times, times_dim, bins_dim, md_data, msd_data, msd_data_true


print("Creating plot(s)...")
Elctrd = leap.simulation.Electrode()
elctrd_thk = Elctrd.ELCTRD_THK / 10  # A -> nm
box_z = Sim.box[2] / 10  # Angstrom -> nm.

xlabel = r"Initial Position $z(t_0)$ / nm"
xlim = (0, box_z)

if args.cmp in ("NBT", "OBT", "OE"):
    legend_title = (
        "$" + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp] + "$" + ", "
    )
else:
    legend_title = leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp] + ", "
legend_title += r"$\Delta t = %.2f$ ns" % time + "\n"
if surfq is not None:
    legend_title += r"$\sigma_s = \pm %.2f$ $e$/nm$^2$" % surfq + "\n"
legend_title += (
    r"$n_{EO} = %d$, " % Sim.O_per_chain + r"$r = %.4f$" % Sim.Li_O_ratio
)
n_legend_cols = 3

markers = ("s", "D", "o")
if len(markers) != len(dimensions):
    raise ValueError(
        "`len(markers)` ({}) != `len(dimensions)`"
        " ({})".format(len(markers), len(dimensions))
    )

height_ratios = (0.2, 1)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot mean displacement vs initial position.
    ylabel = r"$\langle \Delta\mathbf{r} \rangle$ / nm"
    fig, axs = plt.subplots(
        clear=True,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.set_figheight(fig.get_figheight() * sum(height_ratios))
    ax_profile, ax = axs
    leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
    ax.axhline(y=0, color="black")
    if surfq is not None:
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
    md_min, md_max = 0, 0
    for dim_ix, dim in enumerate(dimensions):
        ax.plot(
            bin_mids,
            md_at_const_time[dim_ix],
            label="$" + dim + "$",
            marker=markers[dim_ix],
        )
        md_min = min(md_min, np.nanmin(md_at_const_time[dim_ix]))
        md_max = max(md_max, np.nanmax(md_at_const_time[dim_ix]))
    leap.plot.bins(ax, bins=bins)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if md_max >= abs(md_min):
        legend_loc = "upper center"
    else:
        legend_loc = "lower center"
    legend = ax.legend(
        title=legend_title,
        loc=legend_loc,
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot true MSD vs initial position.
    ylabel = r"$\langle \Delta\mathbf{r}^2 \rangle$ / nm$^2$"
    fig, axs = plt.subplots(
        clear=True,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.set_figheight(fig.get_figheight() * sum(height_ratios))
    ax_profile, ax = axs
    leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
    if surfq is not None:
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
    for dim_ix, dim in enumerate(dimensions):
        ax.plot(
            bin_mids,
            msd_at_const_time_true[dim_ix],
            label="$" + dim + "$",
            marker=markers[dim_ix],
        )
    leap.plot.bins(ax, bins=bins)
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend = ax.legend(
        title=legend_title,
        loc="lower center",
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Plot displacement variance vs initial position.
    ylabel = r"Var$[\Delta\mathbf{r}]$ / nm$^2$"
    fig, axs = plt.subplots(
        clear=True,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.set_figheight(fig.get_figheight() * sum(height_ratios))
    ax_profile, ax = axs
    leap.plot.profile(ax_profile, Sim=Sim, free_en=True)
    if surfq is not None:
        leap.plot.elctrds(
            ax, offset_left=elctrd_thk, offset_right=box_z - elctrd_thk
        )
    for dim_ix, dim in enumerate(dimensions):
        ax.plot(
            bin_mids,
            msd_at_const_time[dim_ix],
            label="$" + dim + "$",
            marker=markers[dim_ix],
        )
    leap.plot.bins(ax, bins=bins)
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend = ax.legend(
        title=legend_title,
        loc="lower center",
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
