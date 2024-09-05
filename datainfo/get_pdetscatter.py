import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.lines import Line2D
import matplotlib.colors

import os
import json
from scipy.interpolate import griddata, RegularGridInterpolator

rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=20
rcParams["ytick.labelsize"]=20
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=18
rcParams["axes.labelsize"]=18
rcParams["axes.grid"] = True
rcParams["grid.color"] = 'black'
#rcParams["grid.linewidth"] = 2.
rcParams["grid.alpha"] = 0.5

data = np.loadtxt('withcorrectionon_time_SNR8.0LISApopIII_4years_Msrc_z_q_Xi1_Xi2_pdet_MFpdet.txt').T
mtotall = data[0]
zall = data[1]
#print(np.max(zall))
Mall = mtotall
pdetall = data[-1]
print(np.max(pdetall), np.min(pdetall))
print("total samples", len(pdetall))
indices = np.argwhere(pdetall > 1e-4).flatten()
Mall = Mall[indices]
zall = zall[indices]
pdetall = pdetall[indices]

print(np.max(pdetall), np.min(pdetall))
print("total samples", len(pdetall))

levels = np.logspace(-4, 0, num=10)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(Mall, zall, c=pdetall, cmap='plasma', norm=LogNorm())#(vmin=1e-4, vmax=1))
plt.colorbar(scatter).set_label(label=r'$p_\mathrm{det}$', size=25)
plt.xlabel(r'$M_\mathrm{source} [M_\odot]$', fontsize=22)
plt.ylabel(r'$\mathrm{redshift}$', fontsize=22)
plt.semilogx()
#plt.ylim(ymax=35.0)
plt.grid(True)
plt.show()
