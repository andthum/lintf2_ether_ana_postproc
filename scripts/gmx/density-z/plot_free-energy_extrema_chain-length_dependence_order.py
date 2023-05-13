#!/usr/bin/env python3


"""
Plot the positions of the extrema of the free-energy profile of a given
compound as function of the PEO chain length.

The extrema are plotted in the order of their distance to the
electrodes.
"""


# Standard libraries
import argparse

# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# First-party libraries
import lintf2_ether_ana_postproc as leap


def legend_title(surfq_sign):
    r"""
    Create a legend title string.

    Parameters
    ----------
    surfq_sign : {"+", "-", r"\pm"}
        The sign of the surface charge.

    Returns
    -------
    title : str
        The legend title.

    Notes
    -----
    This function relies on global variables!
    """
    if surfq_sign not in ("+", "-", r"\pm"):
        raise ValueError("Unknown `surfq_sign`: '{}'".format(surfq_sign))
    return (
        r"$\sigma_s = "
        + surfq_sign
        + r" %.2f$ $e$/nm$^2$, " % Sims.surfqs[0]
        + r"$r = %.2f$" % Sims.Li_O_ratios[0]
        + "\n"
        + r"$F_{"
        + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
        + r"}$ "
        + args.peak_type.capitalize()
    )


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
        "Plot the positions of the extrema of the free-energy profile of a"
        " given compound as function of the PEO chain length."
    )
)
parser.add_argument(
    "--surfq",
    type=str,
    required=True,
    choices=("q0", "q0.25", "q0.5", "q0.75", "q1"),
    help="Surface charge in e/nm^2.",
)
parser.add_argument(
    "--cmp",
    type=str,
    required=False,
    default="Li",
    choices=("Li", "NBT", "OBT", "OE"),
    help="Compound.  Default: %(default)s",
)
parser.add_argument(
    "--peak-type",
    type=str,
    required=False,
    default="minima",
    choices=("minima", "maxima"),
    help="The peak/extremum type.  Default: %(default)s",
)
parser.add_argument(
    "--prob-thresh",
    type=float,
    required=False,
    default=0.5,
    help=(
        "Only consider peaks whose prominence is at least such high that only"
        " 100*PROB_THRESH percent of the particles have a higher 1-dimensional"
        " kinetic energy.  Default: %(default)s"
    ),
)
parser.add_argument(
    "--n-peaks",
    type=lambda val: mdt.fh.str2none_or_type(val, dtype=int),
    required=False,
    default=None,
    help=(
        "The maximum number of peaks to consider.  'None' means all peaks."
        "  Default: %(default)s"
    ),
)
parser.add_argument(
    "--common-ylim",
    required=False,
    default=False,
    action="store_true",
    help="Use common y limits for all plots.",
)
args = parser.parse_args()
if args.prob_thresh < 0 or args.prob_thresh > 1:
    raise ValueError(
        "--prob-thresh ({}) must be between 0 and 1".format(args.prob_thresh)
    )
if args.n_peaks is not None and args.n_peaks <= 0:
    raise ValueError(
        "--n-peaks ({}) must be a positive integer".format(args.n_peaks)
    )

settings = "pr_nvt423_nh"  # Simulation settings.
outfile = (  # Output file name.
    settings
    + "_lintf2_peoN_20-1_gra_"
    + args.surfq
    + "_sc80_free-energy_"
    + args.peak_type
    + "_"
    + args.cmp
    + "_order_pthresh_%.2f" % args.prob_thresh
)
if args.common_ylim:
    outfile += "_common_ylim.pdf"
else:
    outfile += ".pdf"

pkp_col = 1  # Column that contains the peak positions in nm.
cols = (  # Columns to read from the input file(s).
    pkp_col,  # Peak positions [nm]
    2,  # Peak heights [kT]
    3,  # Peak prominences [kT]
    18,  # Peak width at 50 % of the peak prominence [nm]
    25,  # Peak width at 100 % of the peak prominence (peak base) [nm]
)
pkp_col_ix = cols.index(pkp_col)
ylabels = (
    "Distance to Electrode / nm",
    r"Free Energy $F_{"
    + leap.plot.ATOM_TYPE2DISPLAY_NAME[args.cmp]
    + r"}(z_{"
    + args.peak_type[:3]
    + r"})$ / $k_B T$",
    "Prominence / $kT$",
    r"Width at $50$ % Prom. / nm",
    r"Width at $100$ % Prom. / nm",
)
if len(ylabels) != len(cols):
    raise ValueError(
        "`len(ylabels)` ({}) != `len(cols)`"
        " ({})`".format(len(ylabels), len(cols))
    )


print("Creating Simulation instance(s)...")
sys_pat = "lintf2_[gp]*[0-9]*_20-1_gra_" + args.surfq + "_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims = leap.simulation.get_sims(
    sys_pat, set_pat, args.surfq, sort_key="O_per_chain"
)


print("Reading data...")
prom_min = leap.misc.e_kin(args.prob_thresh)
ydata, n_pks_max = leap.simulation.read_free_energy_extrema(
    Sims, args.cmp, args.peak_type, cols, prom_min=prom_min
)
xdata = Sims.O_per_chain


print("Creating plot(s)...")
Elctrd = leap.simulation.Electrode()
bulk_start = Elctrd.BULK_START / 10  # A -> nm

n_pks_plot = np.max(n_pks_max)
if args.n_peaks is not None:
    n_pks_plot = min(n_pks_plot, args.n_peaks)
if n_pks_plot <= 0:
    raise ValueError("`n_pks_plot` ({}) <= 0".format(n_pks_plot))

legend_title_suffix = " Numbers"
n_legend_handles_comb = sum(min(n_pks, n_pks_plot) for n_pks in n_pks_max)
n_legend_cols_comb = min(3, 1 + n_legend_handles_comb // (2 + 1))
legend_locs_sep = tuple("best" for col in cols)
legend_locs_comb = tuple("best" for col in cols)
if len(legend_locs_sep) != len(cols):
    raise ValueError(
        "`len(legend_locs_sep)` ({}) != `len(cols)`"
        " ({})".format(len(legend_locs_sep), len(cols))
    )
if len(legend_locs_comb) != len(cols):
    raise ValueError(
        "`len(legend_locs_comb)` ({}) != `len(cols)`"
        " ({})".format(len(legend_locs_comb), len(cols))
    )

xlabel = r"Ether Oxygens per Chain $n_{EO}$"
xlim = (1, 200)
if args.common_ylim:
    if args.cmp == "Li":
        ylims = [
            (0, 3.6),  # Peak positions [nm] (--prob-thresh 1)
            (None, None),  # Peak heights [kT]
            (0, 11),  # Peak prominences [kT]
            (0, 0.65),  # Peak width at  50 % prominence [nm]
            (0, 2.8),  # Peak width at 100 % prominence [nm]
        ]
        if args.peak_type == "minima":
            # ylims = [
            #     (0.1, 3.1),  # Peak positions [nm]
            #     (-6, 4.5),  # Peak heights [kT]
            #     (0, 11),  # Peak prominences [kT]
            #     (0.02, 0.62),  # Peak width at  50 % prominence [nm]
            #     (0, 2.8),  # Peak width at 100 % prominence [nm]
            # ]
            ylims[1] = (-6, 4.5)  # Peak heights [kT]
        elif args.peak_type == "maxima":
            # ylims = [
            #     (0.2, 3.2),  # Peak positions [nm]
            #     (-0.5, 7.5),  # Peak heights [kT]
            #     (0, 8.25),  # Peak prominences [kT]
            #     (0.04, 0.6),  # Peak width at  50 % prominence [nm]
            #     (0.05, 2),  # Peak width at 100 % prominence [nm]
            # ]
            ylims[1] = (-0.5, 7.5)  # Peak heights [kT]
        else:
            raise ValueError("Unknown --peak-type ({})".format(args.peak_type))
    else:
        raise NotImplementedError(
            "Common ylim not implemented for compounds other than Li"
        )
else:
    ylims = tuple((None, None) for col in cols)
if len(ylims) != len(cols):
    raise ValueError(
        "`len(ylims)` ({}) != `len(cols)` ({})".format(len(ylims), len(cols))
    )

pk_pos_types = ("left", "right")  # Peak at left or right electrode.
linestyles_comb = ("solid", "dashed")
if len(linestyles_comb) != len(pk_pos_types):
    raise ValueError(
        "`len(linestyles_comb)` ({}) != `len(pk_pos_types)`"
        " ({})".format(len(linestyles_comb), len(pk_pos_types))
    )
markers = ("<", ">")
if len(markers) != len(pk_pos_types):
    raise ValueError(
        "`len(markers)` ({}) != `len(pk_pos_types)`"
        " ({})".format(len(markers), len(pk_pos_types))
    )

cmap = plt.get_cmap()
c_vals_sep = np.arange(n_pks_plot)
c_norm_sep = n_pks_plot - 1
c_vals_sep_normed = c_vals_sep / c_norm_sep
colors_sep = cmap(c_vals_sep_normed)

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for col_ix, yd_col in enumerate(ydata):
        # Peaks at left and right electrode combined in one plot.
        fig_comb, ax_comb = plt.subplots(clear=True)

        for pkt_ix, pkp_type in enumerate(pk_pos_types):
            yd_pkt = yd_col[pkt_ix]

            # Peaks at left and right electrode in separate plots.
            fig_sep, ax_sep = plt.subplots(clear=True)
            ax_sep.set_prop_cycle(color=colors_sep)

            c_vals_comb = c_vals_sep + 0.5 * pkt_ix
            c_norm_comb = c_norm_sep + 0.5 * (len(pk_pos_types) - 1)
            c_vals_comb_normed = c_vals_comb / c_norm_comb
            colors_comb = cmap(c_vals_comb_normed)
            ax_comb.set_prop_cycle(color=colors_comb)

            for pk_num in range(min(n_pks_max[pkt_ix], n_pks_plot)):
                y = np.full(len(yd_pkt), np.nan, dtype=np.float64)
                for sim_ix, yd_sim in enumerate(yd_pkt):
                    try:
                        # Try if peak exists in this simulation.
                        yd_sim[pk_num]
                    except IndexError:
                        # Peak does not exist in this simulation.
                        pass
                    else:
                        y[sim_ix] = yd_sim[pk_num]
                invalid = np.isnan(y)
                invalid_ix = np.flatnonzero(invalid)
                if np.all(invalid):
                    raise ValueError(
                        "No peak at the {} electrode in any simulation.  This"
                        " should not have happened".format(pkp_type)
                    )

                if pkp_type == "left":
                    label_sep = r"$%d$" % (pk_num + 1)
                    label_comb = r"$+" + label_sep[1:]
                elif pkp_type == "right":
                    label_sep = r"$%d$" % (pk_num + 1)
                    label_comb = r"$-" + label_sep[1:]
                else:
                    raise ValueError(
                        "Unknown `pkp_type`: '{}'".format(pkp_type)
                    )
                ax_sep.plot(
                    np.delete(xdata, invalid_ix),
                    np.delete(y, invalid_ix),
                    linestyle="solid",
                    marker=markers[pkt_ix],
                    label=label_sep,
                )
                ax_comb.plot(
                    np.delete(xdata, invalid_ix),
                    np.delete(y, invalid_ix),
                    linestyle=linestyles_comb[pkt_ix],
                    marker=markers[pkt_ix],
                    label=label_comb,
                )

            if pkp_type == "left":
                legend_title_sep = legend_title(r"+")
            elif pkp_type == "right":
                legend_title_sep = legend_title(r"-")
            else:
                raise ValueError("Unknown `pkp_type`: '{}'".format(pkp_type))
            ax_sep.set_xscale("log", base=10, subs=np.arange(2, 10))
            ax_sep.set(
                xlabel=xlabel,
                ylabel=ylabels[col_ix],
                xlim=xlim,
                ylim=ylims[col_ix],
            )
            equalize_yticks(ax_sep)
            legend_sep = ax_sep.legend(
                title=legend_title_sep + legend_title_suffix,
                ncol=2,
                loc=legend_locs_sep[col_ix],
                **mdtplt.LEGEND_KWARGS_XSMALL,
            )
            legend_sep.get_title().set_multialignment("center")
            pdf.savefig(fig_sep)
            plt.close(fig_sep)

        ax_comb.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax_comb.set(
            xlabel=xlabel,
            ylabel=ylabels[col_ix],
            xlim=xlim,
            ylim=ylims[col_ix],
        )
        equalize_yticks(ax_comb)
        legend_comb = ax_comb.legend(
            title=legend_title(r"\pm") + legend_title_suffix,
            ncol=n_legend_cols_comb,
            loc=legend_locs_comb[col_ix],
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend_comb.get_title().set_multialignment("center")
        pdf.savefig(fig_comb)
        plt.close(fig_comb)

print("Created {}".format(outfile))
print("Done")
