#!/usr/bin/env python3


"""
Identify the diffusive regime of the mean squared displacement and
extract the self-diffusion coefficient by fitting a straight line to it.
"""


# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def log_line(x, slope=1, intercept=1e-2):
    """Straight line in log scale."""
    return intercept * x**slope


outfile = "test.pdf"
infile = "pr_nvt423_nh_lintf2_peo63_20-1_sc80_msd_Li.txt.gz"
times, msd = np.loadtxt(infile, usecols=(0, 1), unpack=True)
msd_grad = np.gradient(msd, times)


# mdt.fh.backup(outfile)
with PdfPages(outfile) as pdf:
    fix, ax = plt.subplots(clear=True)
    ax.plot(times, msd)
    # ax.plot(times, log_line(times), color="black", linestyle="dashed")
    # ax.set_xscale("log", base=10, subs=np.arange(2, 10))
    # ax.set_yscale("log", base=10, subs=np.arange(2, 10))
    pdf.savefig()
    plt.close()

    fix, ax = plt.subplots(clear=True)
    ax.plot(times, msd_grad, rasterized=True)
    pdf.savefig()
    plt.close()

    fix, ax = plt.subplots(clear=True)
    ax.plot(msd_grad, msd_grad, rasterized=True)
    pdf.savefig()
    plt.close()

print("Created {}".format(outfile))
print("Done")
