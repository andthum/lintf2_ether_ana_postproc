#!/usr/bin/env python3


"""
Plot the Maxwell-Boltzmann distribution of the kinetic energy in
1-dimensional space.
"""


# Third-party libraries
import matplotlib.pyplot as plt
import mdtools as mdt
import mdtools.plot as mdtplt  # Load MDTools plot style  # noqa: F401
import numpy as np  # Load MDTools plot style  # noqa: F401
from scipy import constants

# First-party libraries
import lintf2_ether_ana_postproc as leap


# Input parameters.
outfile = "kinetic_energy_distribution.pdf"
ndim = 1  # Number of spatial dimensions.
temp = 1 / constants.k  # To get the distribution in units of kT.
ekin = np.linspace(0, 2.5, 1000)  # Kinetic energy values.


# Create plot(s).
fig, ax = plt.subplots(clear=True)
ax.plot(ekin, mdt.stats.ekin_dist(ekin, temp=temp, d=ndim), color="C0")
ax.vlines(
    x=leap.misc.e_kin(0.5),
    ymin=0,
    ymax=mdt.stats.ekin_dist(leap.misc.e_kin(0.5), temp=temp, d=ndim),
    color="C1",
    label=r"$c(0.5) = %.4f$ $k_B T$" % leap.misc.e_kin(0.5),
)
ax.vlines(
    x=1 / 2,
    ymin=0,
    ymax=mdt.stats.ekin_dist(1 / 2, temp=temp, d=ndim),
    color="C2",
    label=r"$\langle E_{kin} \rangle = 1/2$ $k_B T$",
)
ax.vlines(
    x=3 / 4,
    ymin=0,
    ymax=mdt.stats.ekin_dist(3 / 4, temp=temp, d=ndim),
    color="C3",
    label=r"$\langle E_{kin}^2 \rangle = 3/4$ $(k_B T)^2$",
)
ax.set(
    xlabel=r"$E_{kin}$ / $k_B T$",
    ylabel=r"$k_B T$ $\rho(E_{kin})$",
    xlim=(0, ekin[-1]),
    ylim=(0, ekin[-1]),
)
legend = ax.legend(loc="upper right")
mdt.fh.backup(outfile)
plt.savefig(outfile)
plt.close()
