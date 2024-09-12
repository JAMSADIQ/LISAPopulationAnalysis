import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib import cm
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.lines import Line2D
import matplotlib.colors

import os
import json

rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=20
rcParams["ytick.labelsize"]=20
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=18
rcParams["axes.labelsize"]=24
rcParams["axes.grid"] = True
rcParams["grid.color"] = 'black'
#rcParams["grid.linewidth"] = 2.
rcParams["grid.alpha"] = 0.4



##########new data 
data = np.loadtxt('alldata.dat').T
mchirp_z = data[1]
z = data[2]
mchirp = mchirp_z/(1.0 + z)
average_snr = data[3]

#scatter plot 
boundaries = [0, 8, 30, 100, np.max(average_snr) + 1]
print(boundaries)
cmap = plt.get_cmap('viridis', len(boundaries) - 1)
norm = BoundaryNorm(boundaries, cmap.N, clip=True)
# Create the scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(mchirp, z, s=100, c=average_snr, cmap=cmap, norm=norm)
cbar = plt.colorbar(scatter, boundaries=boundaries, ticks=boundaries, extend='both')
#scatter = plt.scatter(mchirp, z, c=average_snr, cmap='viridis', norm=LogNorm())
#plt.colorbar(scatter, label='Average SNR')
# Add custom ticks to the colorbar
cbar.set_ticks(boundaries)
#cbar.set_ticklabels(['0', '< 8', '8-30', '30-100', '> 100'])
plt.xlabel(r'$\mathcal{M}$')
plt.ylabel(r'redshift $z$')
plt.semilogx()
plt.title('q=1.001 Mchirp vs z with Average SNR as color', fontsize=16)
plt.show()

from scipy.interpolate import griddata
M_unique = np.unique(np.log10(mchirp))
Mv  = mchirp * (1 + 1.0001) ** 1.2 / 1.0001 ** 0.6
z_unique = np.unique(z)

Mclean = np.log10(np.logspace(2, 9.9, 200))
zclean = np.linspace(0.5, 20, 200)
M_grid, z_grid = np.meshgrid(Mclean, zclean) #np.meshgrid(M_unique, z_unique)
nonlogM_grid, z_grid = np.meshgrid(10**Mclean, zclean) #np.meshgrid(M_unique, z_unique)

# Interpolate f(M, z) onto the grid
#f_grid = griddata((np.log10(mchirp), z), average_snr, (M_grid, z_grid), method='cubic')
f_grid = griddata((np.log10(Mv), z), average_snr, (M_grid, z_grid), method='cubic')
import matplotlib.colors
#levels = [0, 8, 30, 100, 300, 1000, 3000]#, np.max(average_snr) + 1]
levels = [10, 20, 50, 200, 500, 1000, 3000]#, np.max(average_snr) + 1]
norm = matplotlib.colors.BoundaryNorm(levels, len(levels))
colors = list(plt.cm.Blues(np.linspace(0, 1,len(levels)-1)))
#colors[-1] = "red"
cmap = matplotlib.colors.ListedColormap(colors,"", len(colors))
# Label levels with specially formatted floats

# Create the contour plot

plt.figure(figsize=(10, 6))
contourl = plt.contour(nonlogM_grid, z_grid, f_grid, colors='white', levels=levels, norm=norm)
contour = plt.contourf(nonlogM_grid, z_grid, f_grid, levels=levels, cmap=cmap, norm=norm)
cbar = plt.colorbar(contour, ticks=levels)#, boundaries=boundaries, ticks=boundaries, extend='both')
plt.clabel(contourl, inline=False, fontsize=18, colors='k', use_clabeltext=True)
cbar.set_label('average SNR')
cbar.set_ticks(levels)
# plotting data on top ot this with color of pdet
pdetdata = np.loadtxt('/home/jsadiq/Research/E_awKDE/CatalogLISA/lensedPopIII/json-params/samples/new_save_pdet_with_time_to_merger_randomize/withcorrectionon_time_SNR8.0LISApopIII_4years_Msrc_z_q_Xi1_Xi2_pdet_MFpdet.txt').T
Mall = pdetdata[0]
zall = pdetdata[1]
pdetall = pdetdata[-1]
scatter = plt.plot(Mall, zall, 'r+', label='popIII PE data')#, c=pdetall, cmap='plasma', norm=LogNorm())#(vmin=1e-4, vmax=1))
#plt.colorbar(scatter, location='bottom')
#plt.colorbar(scatter).set_label(label=r'$p_\mathrm{det}$', size=25)
plt.semilogx()
#cbar.set_ticklabels(['0', '< 8', '8-30', '30-100', '> 100'])
#plt.xlabel(r'$\mathcal{M}_\mathrm{source} [M_\odot]$')
plt.xlabel(r'$M_\mathrm{source} [M_\odot]$')
plt.ylabel(r'redshift $z$')
plt.ylim(ymin=-0.001,ymax=20)
plt.legend()
plt.title('q=1.0001, Average SNR as color', fontsize=16)
plt.tight_layout()
plt.savefig("LisaconstantSNRcontour_with_popIIIdata.png")
plt.show()




quit()

x_unique = np.unique(mchirp)
#x_unique = np.unique(mchirp_unred)
y_unique = np.unique(z)
print(len(y_unique), len(x_unique))
print(len(average_snr))
# Reshape farr into a 2D array
farr_reshaped = average_snr.reshape(len(x_unique), len(y_unique))
import matplotlib.colors
levels = [0, 8, 30, 100, 300, 1000, 3000, np.max(average_snr) + 1]
norm = matplotlib.colors.BoundaryNorm(levels, len(levels))
colors = list(plt.cm.cividis(np.linspace(0, 1,len(levels)-1)))
#colors[-1] = "red"
cmap = matplotlib.colors.ListedColormap(colors,"", len(colors))
# Label levels with specially formatted floats

contourl = plt.contour(np.log10(x_unique), y_unique, farr_reshaped.T, colors='white', levels=levels, norm=norm) 
contour = plt.contourf(np.log10(x_unique), y_unique, farr_reshaped.T, levels=levels, cmap=cmap, norm=norm) 
cbar = plt.colorbar(contour, ticks=levels)#, boundaries=boundaries, ticks=boundaries, extend='both')
plt.clabel(contourl, inline=False, fontsize=18, colors='white', use_clabeltext=True)
cbar.set_label('average SNR')
cbar.set_ticks(levels)
#cbar.set_ticklabels(['0', '< 8', '8-30', '30-100', '> 100'])
plt.xlabel(r'$\mathrm{log}_{10} \mathcal{M}$')
plt.ylabel(r'redshift $z$')
plt.title('q=1.001 Mchirp vs z with Average SNR as color', fontsize=16)
plt.show()



boundaries = [0, 8, 30, 100, np.max(average_snr) + 1]
print(boundaries)
cmap = plt.get_cmap('viridis', len(boundaries) - 1)
norm = BoundaryNorm(boundaries, cmap.N, clip=True)
# Create the scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(mchirp_unred, z, s=100, c=average_snr, cmap=cmap, norm=norm)
cbar = plt.colorbar(scatter, boundaries=boundaries, ticks=boundaries, extend='both')
#scatter = plt.scatter(mchirp, z, c=average_snr, cmap='viridis', norm=LogNorm())
#plt.colorbar(scatter, label='Average SNR')
# Add custom ticks to the colorbar
cbar.set_ticks(boundaries)
#cbar.set_ticklabels(['0', '< 8', '8-30', '30-100', '> 100'])
plt.xlabel(r'$\mathcal{M}$')
plt.ylabel(r'redshift $z$')
plt.semilogx()
plt.title('q=1.001 Mchirp vs z with Average SNR as color', fontsize=16)
plt.show()

