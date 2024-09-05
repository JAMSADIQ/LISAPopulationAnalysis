import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib.patches
from matplotlib.patches import Rectangle
import glob
import deepdish as dd

rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=18
rcParams["ytick.labelsize"]=18
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=18
rcParams["axes.labelsize"]=18
rcParams["axes.grid"] = True
rcParams["grid.color"] = 'grey'
rcParams["grid.linewidth"] = 1.
#rcParams["grid.linestyle"] = ':'
rcParams["grid.alpha"] = 0.6




#d =np.loadtxt("popIII_4years_observed_events_true_Msrc_z_snr.txt").T
#d = np.loadtxt("popIII_with_lensing_4years_observed_events_true_Mz_z_snr.txt").T
#Msrc = d[0]
d = np.loadtxt("popIII_with_lensing_4years_observed_events_true_Mz_z_snr.txt").T
Mz = d[0] 
z = d[1]
snr = d[2]
Msrc = Mz/(1.0 + z)


plt.figure(figsize=(8, 6))
plt.scatter(Msrc, z, c=snr, cmap='plasma', norm=LogNorm())  # Using 'viridis' colormap with LogNorm
plt.colorbar(label=r'$\mathrm{SNR}$')  # Adding colorbar to show the values of f
plt.xlabel(r'$M_\mathrm{source} [M_\odot]$', fontsize=20)
plt.ylabel(r'$\mathrm{redshift}$', fontsize=20)
plt.title('SNR on true parameters', fontsize=20)
plt.grid(True)
plt.semilogx()
plt.tight_layout()
#plt.savefig("posterior_data_plots/SNR_on_true_params_Msrc_z_scatter_popIII_4years.png")
plt.savefig("Lensed_SNR_on_true_params_Msrc_z_scatter_popIII_4years.png")
plt.show()
