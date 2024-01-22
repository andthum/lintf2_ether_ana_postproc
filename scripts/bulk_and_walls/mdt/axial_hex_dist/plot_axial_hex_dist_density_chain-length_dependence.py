#!/usr/bin/env python3


"""
Plot density profiles along the first- or second-nearest neighbor axes
of a hexagonal lattice for different chain lengths.
"""


# Standard libraries
import argparse
import glob
import os
import re

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# First-party libraries
import lintf2_ether_ana_postproc as leap


def get_slab(fname, prefix):
    """Get the position of the analyzed slab from the file name."""
    if not os.path.isfile(fname):
        raise FileNotFoundError("No such file: '{}'".format(fname))
    fname = os.path.basename(fname)  # Remove path to the file.
    fname = os.path.splitext(fname)[0]  # Remove (first) file extension.
    if not fname.startswith(prefix):
        raise ValueError(
            "The file name '{}' does not start with '{}'".format(fname, prefix)
        )
    slab = fname[len(prefix) :]  # Remove `prefix`.
    slab = re.sub("[^0-9|.|-]", "", slab)  # Remove non-numeric characters.
    slab = slab.strip(".")  # Remove leading and trailing periods.
    slab = slab.split("-")  # Split at hyphens.
    if len(slab) != 2:
        raise ValueError("Invalid slab: {}".format(slab))
    slab = [float(slab) for slab in slab]
    slab_start, slab_stop = min(slab), max(slab)
    return slab_start, slab_stop


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot density profiles along the first- or second-nearest neighbor"
        " axes of a hexagonal lattice for different chain lengths."
    )
)
parser.add_argument(
    "--surfq",
    type=str,
    required=False,
    choices=("q1",),  # "q0", "q0.25", "q0.5", "q0.75"),
    default="q1",
    # Other layers than the 1st and 2nd Li-layer at negative electrodes
    # with a surface charge of q = -1 e/nm^2 (might) require clustering
    # of the slabs.
    help="Surface charge in e/nm^2.",
)
parser.add_argument(
    "--cmp",
    type=str,
    required=False,
    choices=("Li",),  # "NBT", "OBT", "OE"),
    default="Li",
    # Other layers than the 1st and 2nd Li-layer at negative electrodes
    # with a surface charge of q = -1 e/nm^2 (might) require clustering
    # of the slabs.
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--axis",
    type=int,
    required=False,
    default=1,
    choices=(1, 2),
    help=(
        "Whether to use the first- or second-nearest neighbor axis.  Default:"
        " %(default)s"
    ),
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "axial_hex_dist"  # Analysis name.
analysis_axis = "_%dnn" % args.axis
analysis_suffix = "_" + args.cmp
analysis_tot = analysis + analysis_axis + analysis_suffix
ana_path = os.path.join(analysis, analysis + analysis_axis, analysis_tot)
tool = "mdt"  # Analysis software.
outfile = (
    settings
    + "_lintf2_peoN_20-1_gra_"
    + args.surfq
    + "_sc80_"
    + analysis_tot
    + "_density.pdf"
)


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_gra_" + args.surfq + "_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, args.surfq, sort_key="O_per_chain"
)


print("Reading data...")
# Get input files that contain the average compound density in each bin.
analysis_bin_pop = "discrete-z"
file_suffix_bin_pop = analysis_bin_pop + "_Li_bin_population.txt.gz"
infiles_bin_pop = leap.simulation.get_ana_files(
    Sims, analysis_bin_pop, tool, file_suffix_bin_pop
)

# Get the input files that contain the density along the hexagonal axes.
file_extension = ".txt.gz"
file_suffix_pattern = analysis_tot + "*.txt.gz"
infiles = [None for Sim in Sims.sims]
slab_widths = np.full(Sims.n_sims, np.nan, dtype=np.float64)
slab_dens = np.full_like(slab_widths, np.nan)
for sim_ix, path in enumerate(Sims.paths_ana):
    fname_pattern = Sims.fnames_ana_base[sim_ix] + file_suffix_pattern
    fpath_pattern = os.path.join(path, tool, ana_path, fname_pattern)
    files = glob.glob(fpath_pattern)

    # Get the file that contains the data for the slab/bin that is
    # closest to the negative electrode.
    file_prefix = Sims.fnames_ana_base[sim_ix] + analysis_tot
    slab_starts = np.full(len(files), np.nan, dtype=np.float64)
    slab_stops = np.full_like(slab_starts, np.nan)
    for f_ix, file in enumerate(files):
        slab_starts[f_ix], slab_stops[f_ix] = get_slab(file, file_prefix)
    ix_max = np.argmax(slab_starts)
    infiles[sim_ix] = files[ix_max]
    slab_widths[sim_ix] = slab_stops[ix_max] - slab_starts[ix_max]

    # Get the average compound density in the slab/bin.
    tol = 0.02
    bin_starts, bin_stops, bin_dens = np.loadtxt(
        infiles_bin_pop[sim_ix], usecols=(0, 1, 6), unpack=True
    )
    bin_starts, bin_stops = np.round(bin_starts, 2), np.round(bin_stops, 2)
    bin_ix_start = np.flatnonzero(
        np.isclose(bin_starts, slab_starts[ix_max], rtol=0, atol=tol)
    )
    bin_ix_stop = np.flatnonzero(
        np.isclose(bin_stops, slab_stops[ix_max], rtol=0, atol=tol)
    )
    if len(bin_ix_start) != 1:
        raise ValueError(
            "`len(bin_ix_start)` ({}) != 1".format(len(bin_ix_start))
        )
    if len(bin_ix_stop) != 1:
        raise ValueError(
            "`len(bin_ix_stop)` ({}) != 1".format(len(bin_ix_stop))
        )
    bin_ix_start, bin_ix_stop = bin_ix_start[0], bin_ix_stop[0]
    if bin_ix_stop != bin_ix_start:
        raise ValueError(
            "`bin_ix_stop` ({}) != `bin_ix_start`"
            " ({})".format(bin_ix_stop, bin_ix_start)
        )
    slab_dens[sim_ix] = bin_dens[bin_ix_start]  # 1/Angstrom^3.
del files, slab_starts, slab_stops
slab_widths /= 10  # Angstrom -> nm.

# Read data.
xdata = [None for Sim in Sims.sims]
ydata = [None for Sim in Sims.sims]
for sim_ix, infile in enumerate(infiles):
    data = np.loadtxt(infile, usecols=(0, 1, 2))
    xdata[sim_ix] = data[:, 0] / 10  # Angstrom -> nm.
    ydata[sim_ix] = np.nanmean(data[:, 1:], axis=1)  # 1/Angstrom^3.
    ydata[sim_ix] /= slab_dens[sim_ix]
del data


print("Reading data and creating plot(s)...")
xlabel = r"$r$ / nm"
ylabel = (
    r"Density $\rho_{"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
    + r"}(r) / \rho_{"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
    + r"}^{layer}$"
)
xlim = (0, 6 * 0.142)
ylim = (0.4, 4.2)

legend_title = r"$r = %.2f$" % Sims.Li_O_ratios[0] + "\n" + r"$n_{EO}$"
legend_loc = "upper center"
n_legend_cols = Sims.n_sims // 2

cmap = plt.get_cmap()
c_vals = np.arange(Sims.n_sims)
c_norm = max(1, Sims.n_sims - 1)
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Plot slab widths.
    fig, ax = plt.subplots(clear=True)
    ax.plot(Sims.O_per_chain, slab_widths, marker="o")
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set(
        xlabel=r"Ether Oxygens per Chain $n_{EO}$",
        ylabel="Layer Width / nm",
        xlim=(1, 200),
    )
    legend = ax.legend(title=legend_title.split("\n")[0])
    legend.get_title().set_multialignment("center")
    pdf.savefig(fig)
    plt.close(fig)

    # Plot densities.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        if Sim.O_per_chain == 3:
            linestyle = "dashed"
        else:
            linestyle = None
        if Sim.O_per_chain == 4:
            color = "tab:red"
            ax.plot([], [])  # Increment color cycle.
        elif Sim.O_per_chain == 5:
            color = "tab:orange"
            ax.plot([], [])  # Increment color cycle.
        elif Sim.O_per_chain == 8:
            color = "tab:brown"
            ax.plot([], [])  # Increment color cycle.
        else:
            color = None
        ax.plot(
            xdata[sim_ix],
            ydata[sim_ix],
            label=r"$%d$" % Sims.O_per_chain[sim_ix],
            color=color,
            linestyle=linestyle,
            alpha=leap.plot.ALPHA,
        )
    ax.set(xlabel=xlabel, ylabel=ylabel, ylim=ylim)
    legend = ax.legend(
        title=legend_title,
        loc=legend_loc,
        ncol=n_legend_cols,
        **mdtplt.LEGEND_KWARGS_XSMALL,
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig(fig)

    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    pdf.savefig(fig)
    plt.close(fig)

print("Created {}".format(outfile))
print("Done")
