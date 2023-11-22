#!/usr/bin/env python3

"""
Plot the radial distribution function (RDF) and the potential of mean
force (PMF) for two given compounds as function of the PEO chain length.
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
    if xlim_diff > 0.5 and xlim_diff <= 1.3:
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    elif xlim_diff > 1.3 and xlim_diff <= 2.5:
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    elif xlim_diff > 2.5 and xlim_diff <= 5:
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
    """
    ylim = np.asarray(ax.get_ylim())
    ylim_diff = ylim[-1] - ylim[0]
    yticks = np.asarray(ax.get_yticks())
    yticks_valid = (yticks >= ylim[0]) & (yticks <= ylim[-1])
    yticks = yticks[yticks_valid]
    if ylim_diff >= 10 and ylim_diff < 20:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if np.all(yticks >= 0) and np.all(yticks < 10) and ylim_diff > 2:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot the radial distribution function (RDF) and the potential of mean"
        " force (PMF) for two given compounds as function of the PEO chain"
        " length."
    )
)
parser.add_argument(
    "--cmp",
    type=str,
    required=True,
    choices=(
        # <settings>_<system>_rdf_Li.xvg.gz
        "Li-Li",
        "Li-NBT",
        "Li-OBT",
        "Li-OE",
        # <settings>_<system>_rdf_Li-com.xvg.gz
        "Li-NTf2",
        "Li-ether",
        # <settings>_<system>_rdf_NBT.xvg.gz
        "NBT-NBT",
        "NBT-OE",
        # <settings>_<system>_rdf_NTf2-com.xvg.gz
        "NTf2-NTf2",
        "NTf2-ether",
        # <settings>_<system>_rdf_OE.xvg.gz
        "OE-OE",
        # <settings>_<system>_rdf_ether-com.xvg.gz
        "ether-ether",
    ),
    help=(
        "Compounds for which to plot the RDF and PMF.  Default: %(default)s"
    ),
)
args = parser.parse_args()
cmp1, cmp2 = args.cmp.split("-")

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "rdf"  # Analysis name.
analysis_suffix = "_" + args.cmp.split("-")[0]  # Analysis name specification.
if args.cmp in (
    "Li-NTf2",
    "Li-ether",
    "NTf2-NTf2",
    "NTf2-ether",
    "ether-ether",
):
    analysis_suffix += "-com"
tool = "gmx"  # Analysis software.
outfile = (  # Output file name.
    settings + "_lintf2_peoN_20-1_sc80_" + analysis + "_" + args.cmp + ".pdf"
)

# Columns to read from the input file(s).
cols = (0,)  # Distance [nm].
# <settings>_<system>_rdf_Li.xvg.gz
if args.cmp == "Li-Li":
    cols += (1,)
    xmax_rdf, ymax_rdf = 2.8, 1.8
    xmax_cnrdf, ymax_cnrdf = 1.8, 11
    xmax_pmf, ylim_pmf = xmax_rdf, (-0.6, 1.1)
elif args.cmp == "Li-NBT":
    cols += (2,)
    xmax_rdf, ymax_rdf = 1.6, 2.9
    xmax_cnrdf, ymax_cnrdf = 0.7, 0.48
    xmax_pmf, ylim_pmf = xmax_rdf, (-3.5, 4.5)
elif args.cmp == "Li-OBT":
    cols += (3,)
    xmax_rdf, ymax_rdf = 1.6, 7
    xmax_cnrdf, ymax_cnrdf = 0.7, 0.48
    xmax_pmf, ylim_pmf = xmax_rdf, (-3.5, 4.5)
elif args.cmp == "Li-OE":
    cols += (4,)
    xmax_rdf, ymax_rdf = 1.6, 31
    xmax_cnrdf, ymax_cnrdf = 0.7, 7.5
    xmax_pmf, ylim_pmf = xmax_rdf, (-3.5, 4.5)
# <settings>_<system>_rdf_Li-com.xvg.gz
elif args.cmp == "Li-NTf2":
    cols += (1,)
    xmax_rdf, ymax_rdf = 1.6, 2.9
    xmax_cnrdf, ymax_cnrdf = 0.7, 0.48
    xmax_pmf, ylim_pmf = xmax_rdf, (-3.5, 4.5)
elif args.cmp == "Li-ether":
    cols += (2,)
    xmax_rdf, ymax_rdf = 1.6, 9.5
    xmax_cnrdf, ymax_cnrdf = 0.7, 3.2
    xmax_pmf, ylim_pmf = xmax_rdf, (-3.5, 4.5)
# <settings>_<system>_rdf_NBT.xvg.gz
elif args.cmp == "NBT-NBT":
    cols += (1,)
    xmax_rdf, ymax_rdf = 2.8, 1.8
    xmax_cnrdf, ymax_cnrdf = 1.8, 11
    xmax_pmf, ylim_pmf = xmax_rdf, (-0.6, 1.1)
elif args.cmp == "NBT-OE":
    cols += (2,)
    xmax_rdf, ymax_rdf = 2.8, 1.8
    xmax_cnrdf, ymax_cnrdf = 1.8, 11
    xmax_pmf, ylim_pmf = xmax_rdf, (-0.6, 1.1)
# <settings>_<system>_rdf_NTf2-com.xvg.gz
elif args.cmp == "NTf2-NTf2":
    cols += (1,)
    xmax_rdf, ymax_rdf = 2.8, 1.8
    xmax_cnrdf, ymax_cnrdf = 1.8, 11
    xmax_pmf, ylim_pmf = xmax_rdf, (-0.6, 1.1)
elif args.cmp == "NTf2-ether":
    cols += (2,)
    xmax_rdf, ymax_rdf = 2.8, 1.8
    xmax_cnrdf, ymax_cnrdf = 1.8, 11
    xmax_pmf, ylim_pmf = xmax_rdf, (-0.6, 1.1)
# <settings>_<system>_rdf_OE.xvg.gz
elif args.cmp == "OE-OE":
    cols += (1,)
    xmax_rdf, ymax_rdf = 1.6, 2.8
    xmax_cnrdf, ymax_cnrdf = 0.7, 7.5
    xmax_pmf, ylim_pmf = xmax_rdf, (-3.5, 4.5)
# <settings>_<system>_rdf_ether-com.xvg.gz
elif args.cmp == "ether-ether":
    cols += (1,)
    xmax_rdf, ymax_rdf = 1.6, 1.8
    xmax_cnrdf, ymax_cnrdf = 1.8, 11
    xmax_pmf, ylim_pmf = xmax_rdf, (-0.6, 1.1)
else:
    raise ValueError("Unknown --cmp: {}".format(args.cmp))

# Start of the region where the RDF is one in nm.
bulk_start = 2.5


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="O_per_chain"
)


print("Reading data...")
# RDFs.
file_suffix = analysis + analysis_suffix + ".xvg.gz"
infiles_rdf = leap.simulation.get_ana_files(Sims, analysis, tool, file_suffix)
n_infiles = len(infiles_rdf)

rdf_xdata = [None for file in infiles_rdf]
rdf_ydata = [None for file in infiles_rdf]
pmf_ydata = [None for file in infiles_rdf]
for i, infile in enumerate(infiles_rdf):
    rdf_xdata[i], rdf_ydata[i] = np.loadtxt(
        infile, comments=["#", "@"], usecols=cols, unpack=True
    )
    # Skip last data point because it's an artifact.
    rdf_xdata[i] = rdf_xdata[i][:-1]
    rdf_ydata[i] = rdf_ydata[i][:-1]
    pmf_ydata[i] = leap.misc.rdf2free_energy(
        x=rdf_xdata[i], rdf=rdf_ydata[i], bulk_start=bulk_start
    )
    rdf_xdata[i] = rdf_xdata[i].astype(np.float32)
    rdf_ydata[i] = rdf_ydata[i].astype(np.float32)
    pmf_ydata[i] = pmf_ydata[i].astype(np.float32)

# Cumulative number RDFs (integrated RDFs -> coordination numbers).
file_suffix = "cn" + analysis + analysis_suffix + ".xvg.gz"
infiles_cnrdf = leap.simulation.get_ana_files(
    Sims, analysis, tool, file_suffix
)
if len(infiles_cnrdf) != n_infiles:
    raise ValueError(
        "`len(infiles_cnrdf)` ({}) != `n_infiles`"
        " ({})".format(len(infiles_cnrdf), n_infiles)
    )

cnrdf_xdata = [None for file in infiles_rdf]
cnrdf_ydata = [None for file in infiles_rdf]
for i, infile in enumerate(infiles_cnrdf):
    cnrdf_xdata[i], cnrdf_ydata[i] = np.loadtxt(
        infile, comments=["#", "@"], usecols=cols, unpack=True
    )
    # Skip last data point because it's an artifact.
    cnrdf_xdata[i] = cnrdf_xdata[i][:-1]
    cnrdf_ydata[i] = cnrdf_ydata[i][:-1]
    cnrdf_xdata[i] = cnrdf_xdata[i].astype(np.float32)
    cnrdf_ydata[i] = cnrdf_ydata[i].astype(np.float32)


print("Creating plots...")
legend_title = (
    r"$"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp1]
    + "-"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[cmp2]
    + r"$"
    + "\n"
    + r"$r = %.2f$" % Sims.Li_O_ratios[0]
    + "\n"
    + r"$n_{EO}$"
)
n_legend_cols = 1 + n_infiles // (4 + 1)

cmap = plt.get_cmap()
c_vals = np.arange(n_infiles)
c_norm = n_infiles - 1
c_vals_normed = c_vals / c_norm
colors = cmap(c_vals_normed)

# Highlight by color

# Radial distribution function
# Chain length 2, 5:
# Li-OBT, Li-NBT, Li-TFSI
# Li-Li
# NBT-NBT, TFSI-TFSI
# Chain length 3:
# Li-OBT, Li-NBT, Li-TFSI
# No:
# Li-OE, Li-PEO
# NBT-OE, TFSI-PEO
# OE-OE, PEO-PEO

# Coordination number
# Chain length 2, 5:
# Li-OE, Li-PEO
# Li-OBT, Li-NBT, Li-TFSI
# Chain length 3:
# Li-OE, Li-PEO
# Li-OBT, Li-NBT, Li-TFSI
# No:
# Li-Li
# NBT-NBT, TFSI-TFSI
# NBT-OE, TFSI-PEO
# OE-OE, PEO-PEO

# Free energy
# Chain length 2, 5:
# Li-OE, Li-PEO
# Li-OBT, Li-NBT, Li-TFSI
# Li-Li
# NBT-NBT, TFSI-TFSI
# Chain length 3:
# Li-OE, Li-PEO
# Li-OBT, Li-NBT, Li-TFSI
# No:
# NBT-OE, TFSI-PEO
# OE-OE, PEO-PEO

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # RDF.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        if Sim.O_per_chain == 2:
            color = "tab:red"
            ax.plot([], [])  # Increment color cycle.
        elif Sim.O_per_chain == 3:
            color = "tab:brown"
            ax.plot([], [])  # Increment color cycle.
        elif Sim.O_per_chain == 5:
            color = "tab:orange"
            ax.plot([], [])  # Increment color cycle.
        else:
            color = None
        ax.plot(
            rdf_xdata[sim_ix],
            rdf_ydata[sim_ix],
            label=r"$%d$" % Sim.O_per_chain,
            color=color,
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Distance / nm",
        ylabel="RDF",
        xlim=(0, xmax_rdf),
        ylim=(0, ymax_rdf),
    )
    equalize_xticks(ax)
    equalize_yticks(ax)
    legend = ax.legend(
        title=legend_title, ncol=n_legend_cols, **mdtplt.LEGEND_KWARGS_XSMALL
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # Cumulative Number RDF.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        if Sim.O_per_chain == 2:
            color = "tab:red"
            ax.plot([], [])  # Increment color cycle.
        elif Sim.O_per_chain == 3:
            color = "tab:brown"
            ax.plot([], [])  # Increment color cycle.
        elif Sim.O_per_chain == 5:
            color = "tab:orange"
            ax.plot([], [])  # Increment color cycle.
        else:
            color = None
        ax.plot(
            cnrdf_xdata[sim_ix],
            cnrdf_ydata[sim_ix],
            label=r"$%d$" % Sim.O_per_chain,
            color=color,
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Distance / nm",
        ylabel="Coordination Number",
        xlim=(0, xmax_cnrdf),
        ylim=(0, ymax_cnrdf),
    )
    equalize_xticks(ax)
    equalize_yticks(ax)
    legend = ax.legend(
        title=legend_title, ncol=n_legend_cols, **mdtplt.LEGEND_KWARGS_XSMALL
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

    # PMF.
    fig, ax = plt.subplots(clear=True)
    ax.set_prop_cycle(color=colors)
    for sim_ix, Sim in enumerate(Sims.sims):
        if Sim.O_per_chain == 2:
            color = "tab:red"
            ax.plot([], [])  # Increment color cycle.
        elif Sim.O_per_chain == 3:
            color = "tab:brown"
            ax.plot([], [])  # Increment color cycle.
        elif Sim.O_per_chain == 5:
            color = "tab:orange"
            ax.plot([], [])  # Increment color cycle.
        else:
            color = None
        ax.plot(
            rdf_xdata[sim_ix],
            pmf_ydata[sim_ix],
            label=r"$%d$" % Sim.O_per_chain,
            color=color,
            alpha=leap.plot.ALPHA,
        )
    ax.set(
        xlabel="Distance / nm",
        ylabel=r"Free Energy / $k_B T$",
        xlim=(0, xmax_pmf),
        ylim=ylim_pmf,
    )
    equalize_xticks(ax)
    equalize_yticks(ax)
    legend = ax.legend(
        title=legend_title, ncol=n_legend_cols, **mdtplt.LEGEND_KWARGS_XSMALL
    )
    legend.get_title().set_multialignment("center")
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
