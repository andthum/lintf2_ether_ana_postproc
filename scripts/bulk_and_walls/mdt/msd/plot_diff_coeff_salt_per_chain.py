#!/usr/bin/env python3


"""
Plot the center-of-mass diffusion coefficients of Li, TFSI and PEO as
function of the number of lithium ions per PEO chain.
"""


# Standard libraries
import argparse

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
    description=(
        "Plot the center-of-mass diffusion coefficients of Li, TFSI and PEO as"
        " function of the number of lithium ions per PEO chain."
    ),
)
args = parser.parse_args()

settings = "pr_nvt423_nh"  # Simulation settings.
analysis = "msd"  # Analysis name.
tool = "mdt"  # Analysis software.
outfile = (  # Output file name.
    settings + "_lintf2_peoN_r_sc80_" + analysis + "_diff_coeff_tot.pdf"
)


print("Creating Simulation instance(s)...")
# Chain length n_EO = varying.
# Salt concentration r = 1/20 = 0.05.
sys_pat = "lintf2_[gp]*[0-9]*_20-1_sc80"
set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
Sims_peoN_20_1 = leap.simulation.get_sims(
    sys_pat, set_pat, path_key="bulk", sort_key="O_per_chain"
)

# Chain length n_EO = 2, 5 or 64.
# Salt concentration r = varying.
solvents = ("g1", "g4", "peo63")
Sims_sol_r_lst = [None for sol in solvents]
for sol_ix, sol in enumerate(solvents):
    sys_pat = "lintf2_" + sol + "_[0-9]*-[0-9]*_sc80"
    set_pat = "[0-9][0-9]_" + settings + "_" + sys_pat
    Sims = leap.simulation.get_sims(
        sys_pat, set_pat, path_key="bulk", sort_key="Li_O_ratio"
    )
    Sims_sol_r_lst[sol_ix] = Sims


Sims_lst = [Sims_peoN_20_1] + Sims_sol_r_lst
salt_per_chain = [None for Sims in Sims_lst]
for sims_ix, Sims in enumerate(Sims_lst):
    salt_per_chain[sims_ix] = (
        Sims.res_nums["cation"] / Sims.res_nums["solvent"]
    )


print("Reading data...")
compounds = ("ether", "NTf2", "Li")
diff_coeffs = [[[] for cmp in compounds] for Sims in Sims_lst]
diff_coeffs_sd = [[[] for cmp in compounds] for Sims in Sims_lst]
for sims_ix, Sims in enumerate(Sims_lst):
    for cmp_ix, cmp in enumerate(compounds):
        file_suffix = analysis + "_" + cmp + "_tot_diff_coeff.txt.gz"
        infiles = leap.simulation.get_ana_files(
            Sims, analysis, tool, file_suffix
        )
        for infile in infiles:
            d_coeff, d_coeff_sd = np.loadtxt(infile, usecols=(0, 1))
            diff_coeffs[sims_ix][cmp_ix].append(d_coeff)
            diff_coeffs_sd[sims_ix][cmp_ix].append(d_coeff_sd)


print("Creating plot(s)...")
xlabel = r"Salt per Chain $s = r \cdot n_{EO}$"
ylabel = r"Diff. Coeff. / nm$^2$ ns$^{-1}$"

labels_sim = [r"$r = %.2f$" % Sims_lst[0].Li_O_ratios[0]]
labels_sim += [r"$n_{EO} = %d$" % Sims.O_per_chain[0] for Sims in Sims_lst[1:]]
markers_sim = ("o", "^", "v", ">")

labels_cmp = ("PEO", "TFSI", "Li")
markers_cmp = ("o", "s", "^")
colors_cmp = ("tab:blue", "tab:orange", "tab:green")

mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    # Diffusion coefficients vs chain length, all compounds in one plot.
    fig, ax = plt.subplots(clear=True)
    for sims_ix, _Sims in enumerate(Sims_lst):
        ax.plot(
            [],
            [],
            color="black",
            marker=markers_sim[sims_ix],
            label=labels_sim[sims_ix],
        )
    for cmp_ix, _cmp in enumerate(compounds):
        ax.plot([], [], color=colors_cmp[cmp_ix], label=labels_cmp[cmp_ix])
    for sims_ix, _Sims in enumerate(Sims_lst):
        for cmp_ix, _cmp in enumerate(compounds):
            ax.errorbar(
                salt_per_chain[sims_ix],
                diff_coeffs[sims_ix][cmp_ix],
                yerr=None,  # diff_coeffs_sd[sims_ix][cmp_ix], (SD < symbols).
                marker=markers_sim[sims_ix],
                color=colors_cmp[cmp_ix],
                alpha=leap.plot.ALPHA,
            )
    ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="lower left", ncol=2, **mdtplt.LEGEND_KWARGS_XSMALL)
    pdf.savefig()
    plt.close()

    # Diffusion coefficients vs chain length, one plot for each compound
    for cmp_ix, _cmp in enumerate(compounds):
        fig, ax = plt.subplots(clear=True)
        for sims_ix, _Sims in enumerate(Sims_lst):
            ax.errorbar(
                salt_per_chain[sims_ix],
                diff_coeffs[sims_ix][cmp_ix],
                yerr=None,  # diff_coeffs_sd[sims_ix][cmp_ix], (SD < symbols).
                label=labels_sim[sims_ix],
                marker=markers_sim[sims_ix],
                alpha=leap.plot.ALPHA,
            )
        ax.set_xscale("log", base=10, subs=np.arange(2, 10))
        ax.set_yscale("log", base=10, subs=np.arange(2, 10))
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(
            title=labels_cmp[cmp_ix],
            loc="lower left",
            **mdtplt.LEGEND_KWARGS_XSMALL,
        )
        pdf.savefig()
        plt.close()

print("Created {}".format(outfile))
print("Done")
