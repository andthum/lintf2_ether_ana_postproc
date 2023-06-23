#!/usr/bin/env python3


# MIT License


r"""
Plot a selected energy term from a Gromacs .edr file versus time for a
given 'transfer' simulation.

The energy terms are first averaged over all 'transfer' simulations with
the same start time.  Afterwards the mean of means and the standard
deviation of the mean of means is calculated.

To create all relevant plots run

.. code-block:: bash

    for sol in g1 g4 peo63; do
        for set in re_nvt423_ld pr_nvt423_vr; do
            for obs in "Potential" "Kinetic En." "Total Energy" "Temperature" "Pressure" "Constr. rmsd" "T-electrodes" "T-electrolyte"; do
                ~/git/github/lintf2_ether_ana_postproc/.venv/bin/python3 \
                    ~/git/github/lintf2_ether_ana_postproc/scripts/transfer_Li/gmx/energy/plot_energy.py \
                        --sol "${sol}" \
                        --settings "${set}" \
                        --observable "${obs}" ||
                        exit
            done
        done
    done
"""  # noqa: E501, W505


# Standard libraries
import argparse
import gzip
import os
import shutil
import sys
import uuid
from datetime import datetime

# Third-party libraries
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np
import pyedr
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# First-party libraries
import lintf2_ether_ana_postproc as leap


def read_edr_gz(
    fname, observable="Potential", print_obs=False, begin=0, end=-1, every=1
):
    """
    Read a gzipped Gromacs .edr file.

    Parameters
    ----------
    fname : str
        The file name.
    observable : str, optional
        The energy term to read from the .edr file.
    print_obs : bool, optional
        If ``True``, print all energy terms contained in the .edr file.
    begin : int, optional
        First frame to use from the .edr file.  Frame numbering starts
        at zero.
    end : int, optional
        Last frame to use from the .edr file (exclusive).
    every : int, optional
        Use every n-th frame from the .edr file.

    Returns
    -------
    data : dict
        A Dictionary that holds the times and the selected observable.
    units : dict
        A dictionary containing the time unit and the unit of the
        selected observable.
    """
    # Decompress the gzipped .edr file.
    timestamp = datetime.now()
    file_decompressed = (
        fname[:-7]
        + "_uuid_"
        + str(uuid.uuid4())
        + "_date_"
        + str(timestamp.strftime("%Y-%m-%d_%H-%M-%S"))
        + ".edr"
    )
    with gzip.open(fname, "rb") as file_in:
        with open(file_decompressed, "wb") as file_out:
            shutil.copyfileobj(file_in, file_out)

    # Read decompressed file.
    data = pyedr.edr_to_dict(file_decompressed, verbose=True)
    units = pyedr.get_unit_dictionary(file_decompressed)

    # Remove decompressed file.
    os.remove(file_decompressed)

    if print_obs:
        print(
            "The following energy terms are contained in the input file"
            " '{}':".format(fname)
        )
        for key in data.keys():
            print(key)
        print("The selected observable is: '{}'".format(observable))

    # Get desired observable.
    data = {"Time": data["Time"], observable: data[observable]}
    units = {"Time": units["Time"], observable: units[observable]}

    # Get desired frames.
    n_frames_tot = len(data["Time"])
    begin, end, every, n_frames = mdt.check.frame_slicing(
        start=begin, stop=end, step=every, n_frames_tot=n_frames_tot
    )
    for key, value in data.items():
        data[key] = value[begin:end:every]

    return data, units


# Input parameters.
parser = argparse.ArgumentParser(
    description=(
        "Plot a selected energy term from a Gromacs .edr file versus time for"
        " a given 'transfer' simulation."
    )
)
parser.add_argument(
    "--sol",
    type=str,
    required=True,
    choices=("g1", "g4", "peo63"),
    help="Solvent name.",
)
parser.add_argument(
    "--settings",
    type=str,
    required=True,
    choices=("re_nvt423_ld", "pr_nvt423_vr"),
    help="Simulation settings.",
)
parser.add_argument(
    "--observable",
    type=str,
    required=False,
    default="Potential",
    help=(
        "The energy term to read from the .edr file.  Default: %(default)s."
    ),
)
parser.add_argument(
    "--print-obs",
    required=False,
    default=False,
    action="store_true",
    help="Only print all energy terms contained in the .edr file and exit.",
)
parser.add_argument(
    "--begin",
    type=int,
    required=False,
    default=0,
    help=(
        "First frame to use from the .edr file.  Frame numbering starts"
        " at zero.  Default: %(default)s."
    ),
)
parser.add_argument(
    "--end",
    type=int,
    required=False,
    default=-1,
    help=(
        "Last frame to use from the .edr file (exclusive).  Default:"
        " %(default)s."
    ),
)
parser.add_argument(
    "--every",
    type=int,
    required=False,
    default=1,
    help="Use every n-th frame from the .edr file.  Default: %(default)s.",
)
args = parser.parse_args()
if args.settings == "re_nvt423_ld":
    time_unit = "ps"
    time_conv = 1
elif args.settings == "pr_nvt423_vr":
    time_unit = "ns"
    time_conv = 1e-3  # ps -> ns.
else:
    raise ValueError("Unknown --settings '{}'".format(args.settings))

system = "lintf2_" + args.sol + "_20-1_gra_q1_sc80"
analysis = "energy"  # Analysis name.
tool = "gmx"  # Analysis software.
outfile = (
    args.settings
    + "_"
    + system
    + "_"
    + "transfer_Li_"
    + analysis
    + "_"
    + args.observable.replace(" ", "_").replace(".", "")
    + ".pdf"
)


print("Creating Simulation instance(s)...")
sims_dct_sys = leap.transfer.get_sims(system, args.settings)
if args.print_obs:
    sims_dct_start = tuple(sims_dct_sys.values())[0]
    Sims = tuple(sims_dct_start.values())[0]
    Sim = Sims.sims[0]
    infile = Sim.settings + "_out_" + Sim.system + ".edr.gz"
    infile = os.path.join(Sim.path, infile)
    read_edr_gz(
        infile,
        args.observable,
        args.print_obs,
        args.begin,
        args.end,
        args.every,
    )
    print("Done")
    sys.exit(0)


print("Reading data and creating plot(s)...")
alpha_sims = 0.25
color_starts = "tab:red"

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    for sys_name, sims_dct_start in sims_dct_sys.items():
        print("System: {:s}".format(sys_name))

        n_start_times = len(sims_dct_start)
        cmap = plt.get_cmap()
        c_vals = np.arange(n_start_times)
        c_norm = n_start_times - 1
        c_vals_normed = c_vals / c_norm
        colors = cmap(c_vals_normed)

        fig, ax = plt.subplots(clear=True)
        ax.set_prop_cycle(color=colors)

        n_sims_tot = 0
        for start_ix, (start_time, Sims) in enumerate(sims_dct_start.items()):
            print("Start time: {:d} ns".format(start_time))

            for sim_ix, Sim in enumerate(Sims.sims):
                infile = Sim.settings + "_out_" + Sim.system + ".edr.gz"
                infile = os.path.join(Sim.path, infile)
                data_dct_sim, units_dct_sim = read_edr_gz(
                    infile,
                    args.observable,
                    args.print_obs,
                    args.begin,
                    args.end,
                    args.every,
                )
                time_unit_sim = units_dct_sim["Time"]
                data_unit_sim = units_dct_sim[args.observable]
                time_sim = data_dct_sim["Time"] * time_conv
                data_sim = data_dct_sim[args.observable]
                if time_unit_sim != "ps":
                    raise ValueError(
                        "Expected 'ps' as time unit but got '{}'.  Current"
                        " simulation: '{}'".format(time_unit_sim, Sim.path)
                    )

                # Average over all simulations with the same start time.
                if sim_ix == 0:
                    time_unit_sims = time_unit_sim
                    data_unit_sims = data_unit_sim
                    time_sims = time_sim
                    data_mom1_sims = data_sim  # First moment (mean).
                    data_mom2_sims = data_sim**2  # Second moment.
                    continue
                if time_unit_sim != time_unit_sims:
                    raise ValueError(
                        "The time unit of the current simulation ('{}')"
                        " differs from that of the previous simulation(s)"
                        " ('{}').  Current simulation:"
                        " '{}'".format(time_unit_sim, time_unit_sims, Sim.path)
                    )
                if data_unit_sim != data_unit_sims:
                    raise ValueError(
                        "The data unit of the current simulation ('{}')"
                        " differs from that of the previous simulation(s)"
                        " ('{}').  Current simulation:"
                        " '{}'".format(data_unit_sim, data_unit_sims, Sim.path)
                    )
                if len(time_sim) != len(time_sims):
                    raise ValueError(
                        "The number of frames in the current simulation ('{}')"
                        " differs from that in the previous simulation(s)"
                        " ('{}').  Current simulation:"
                        " '{}'".format(len(time_sim), len(time_sims), Sim.path)
                    )
                if not np.allclose(time_sim, time_sims, rtol=0):
                    raise ValueError(
                        "The times in the current simulation differ from that"
                        " in the previous simulation(s).  Current simulation:"
                        " '{}'".format(Sim.path)
                    )
                data_mom1_sims += data_sim
                data_mom2_sims += data_sim**2
            data_mom1_sims /= Sims.n_sims
            data_mom2_sims /= Sims.n_sims
            data_sd_sims = np.sqrt(data_mom2_sims - data_mom1_sims**2)
            data_sd_sims = data_sd_sims.astype(np.float32)

            if len(time_sims) > 1000:
                rasterized = True
            else:
                rasterized = False
            ax.fill_between(
                time_sims,
                data_mom1_sims - data_sd_sims,
                data_mom1_sims + data_sd_sims,
                linewidth=0,
                alpha=alpha_sims,
                rasterized=rasterized,
            )
            del data_sd_sims
            ax.plot(
                time_sims,
                data_mom1_sims,
                label=start_time,
                alpha=alpha_sims,
                rasterized=rasterized,
            )

            # Average over all start times.
            n_sims_tot += Sims.n_sims
            if start_ix == 0:
                time_unit_starts = time_unit_sims
                data_unit_starts = data_unit_sims
                time_starts = time_sims
                data_mom1_starts = Sims.n_sims * data_mom1_sims
                data_mom2_starts = Sims.n_sims * data_mom1_sims**2
                continue
            if time_unit_sims != time_unit_starts:
                raise ValueError(
                    "The time unit for the current start time ('{}') differs"
                    " from that for the previous start time(s) ('{}')."
                    "  Current start time:"
                    " '{}'".format(
                        time_unit_sims, time_unit_starts, start_time
                    )
                )
            if data_unit_sims != data_unit_starts:
                raise ValueError(
                    "The data unit for the current start time ('{}') differs"
                    " from that for the previous start time(s) ('{}')."
                    "  Current start time:"
                    " '{}'".format(
                        data_unit_sims, data_unit_starts, start_time
                    )
                )
            if len(time_sims) != len(time_starts):
                raise ValueError(
                    "The number of frames for the current start time ('{}')"
                    " differs from that for the previous start time(s) ('{}')."
                    "  Current start time:"
                    " '{}'".format(
                        len(time_sims), len(time_starts), start_time
                    )
                )
            if not np.allclose(time_sims, time_starts, rtol=0):
                raise ValueError(
                    "The times for the current start time differ from that for"
                    " the previous start time(s).  Current star time:"
                    " '{}'".format(start_time)
                )
            data_mom1_starts += Sims.n_sims * data_mom1_sims
            data_mom2_starts += Sims.n_sims * data_mom1_sims**2
        data_mom1_starts /= n_sims_tot
        data_mom2_starts /= n_sims_tot
        data_sd_starts = np.sqrt(data_mom2_starts - data_mom1_starts**2)
        data_sd_starts = data_sd_starts.astype(np.float32)
        data_mom1_starts = data_mom1_starts.astype(np.float32)
        data_mom2_starts = data_mom2_starts.astype(np.float32)

        ax.fill_between(
            time_starts,
            data_mom1_starts - data_sd_starts,
            data_mom1_starts + data_sd_starts,
            linewidth=0,
            color=color_starts,
            alpha=leap.plot.ALPHA,
            rasterized=rasterized,
        )
        del data_sd_starts
        ax.plot(
            time_starts,
            data_mom1_starts,
            label="Mean",
            color=color_starts,
            alpha=leap.plot.ALPHA,
            rasterized=rasterized,
        )
        ax.set(
            xlabel="Time / " + time_unit,
            ylabel=args.observable + " / " + data_unit_starts,
            xlim=(time_starts[0], time_starts[-1]),
        )
        legend_title = (
            r"$\sigma_s = \pm %.2f$" % Sims.surfqs[0]
            + r" $e$/nm$^2$"
            + "\n"
            + r"$n_{EO} = %d$, " % Sims.O_per_chain[0]
            + r"$r = %.2f$" % Sims.Li_O_ratios[0]
            + "\n"
            + r"$t_0$ / ns"
        )
        legend = ax.legend(
            title=legend_title,
            ncol=2,
            loc="upper right",
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        legend.get_title().set_multialignment("center")
        pdf.savefig()

        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlim=(time_starts[1], time_starts[-1]))
        pdf.savefig()
        plt.close()

print("Created {}".format(outfile))
print("Done")
