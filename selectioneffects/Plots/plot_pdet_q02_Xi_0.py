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
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from scipy.interpolate import RegularGridInterpolator

rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=15
rcParams["ytick.labelsize"]=15
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=16
rcParams["axes.labelsize"]=18
rcParams["axes.grid"] = True
rcParams["grid.color"] = 'black'
#rcParams["grid.linewidth"] = 2.
rcParams["grid.alpha"] = 0.4

from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from scipy.interpolate import RegularGridInterpolator

# Load the data
data = np.loadtxt('datapdet_threshold_snr10.txt').T
data2 = np.loadtxt('datapdet_threshold_snr8.txt').T
M1, z1, pdet1 = np.log10(data[0]), data[1], data[2]
M2, z2, pdet2 = np.log10(data2[0]), data2[1], data2[2]

# Set values of pdet >= 0.9 to 0.9 for both datasets
pdet1[pdet1 >= 0.99] = 1.0
pdet2[pdet2 >= 0.99] = 1.0

# Prepare unique values for M and z for both datasets
M1_unique = np.unique(M1)
z1_unique = np.unique(z1)
M2_unique = np.unique(M2)
z2_unique = np.unique(z2)

# Reshape pdet to match the grid for both datasets
data_values1 = pdet1.reshape((len(M1_unique), len(z1_unique)))
data_values2 = pdet2.reshape((len(M2_unique), len(z2_unique)))

# Create the interpolators for both datasets
interp1 = RegularGridInterpolator((M1_unique, z1_unique), data_values1, bounds_error=False, fill_value=0.0)
interp2 = RegularGridInterpolator((M2_unique, z2_unique), data_values2, bounds_error=False, fill_value=0.0)

# Create a meshgrid for plotting
grid_x1, grid_y1 = np.meshgrid(M1_unique, z1_unique)
grid_z1 = interp1((grid_x1, grid_y1))

grid_x2, grid_y2 = np.meshgrid(M2_unique, z2_unique)
grid_z2 = interp2((grid_x2, grid_y2))

# Create custom colormaps
cmap_red_yellow = LinearSegmentedColormap.from_list('red_yellow', ['red', 'yellow'])
cmap_blue_yellow = LinearSegmentedColormap.from_list('blue_yellow', ['blue', 'yellow'])

# Define levels for the normalization
levels = np.concatenate(([1e-5], np.logspace(-4, 1, 10)))


# Plotting the contour plot with logarithmic normalization
plt.figure(figsize=(12, 10))

norm2 = BoundaryNorm(levels, ncolors=cmap_blue_yellow.N)
contour_lines2 = plt.contour(grid_x2, grid_y2, grid_z2, levels=[1e-4, 1e-3, 1e-2, 1e-1, 0.9, 0.95, 1.0], colors='black', linewidths=0.5)
contour2 = plt.contourf(grid_x2, grid_y2, grid_z2, levels=levels, cmap=cmap_blue_yellow, norm=norm2, alpha=0.99)
# First dataset contourf plot
norm1 = BoundaryNorm(levels, ncolors=cmap_red_yellow.N)
contour1 = plt.contourf(grid_x1, grid_y1, grid_z1, levels=levels, cmap=cmap_red_yellow, norm=norm1, alpha=0.99)
# Add contour lines
contour_lines1 = plt.contour(grid_x1, grid_y1, grid_z1, levels=[1e-4, 1e-3, 1e-2, 1e-1, 0.9, 0.95, 1.0], colors='purple', linewidths=0.5)
plt.clabel(contour_lines1, fmt='%1.0e', colors='black')

# Second dataset contourf plot
#norm2 = BoundaryNorm(levels, ncolors=cmap_blue_yellow.N)
#contour2 = plt.contourf(grid_x2, grid_y2, grid_z2, levels=levels, cmap=cmap_blue_yellow, norm=norm2, alpha=0.6)
# Add contour lines
#contour_lines2 = plt.contour(grid_x2, grid_y2, grid_z2, levels=[1e-4, 1e-3, 1e-2, 1e-1, 0.9], colors='black', linewidths=0.5)
#plt.clabel(contour_lines2, fmt='%1.0e', colors='black')

# Add a colorbar with fixed ticks
cbar = plt.colorbar(contour1, ticks=[1e-4, 1e-3, 1e-2, 1e-1, 0.9])
cbar.set_label(r'$p_{det}$')

plt.text(2.9, 10, r'$SNR=10$', fontsize=20, color='red', rotation='horizontal')#, transform=plt.gca().transAxes)
plt.text(3.0, 15, r'$SNR=8$', fontsize=20, color='blue', rotation='horizontal')#, transform=plt.gca().transAxes)
#plt.text(3, 14, 'SNR8', horizontalalignment='center', verticalalignment='center', fontsize=14, color='blue', transform=plt.gca().transAxes, alpha=0.6)
# Set labels and title
plt.xlabel(r'$Log10(M_z)$')
plt.ylabel(r'$Redshift (z)$')
plt.title(r'$p_{det} \,\,with \, \,  q=0.2, \vec{\chi} = 0$', fontsize=18)
plt.savefig('Pdet_fixedq_spinszero.png')
# Show plot
plt.show()

