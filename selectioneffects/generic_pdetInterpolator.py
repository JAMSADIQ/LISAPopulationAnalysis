import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Sample data (replace this with your actual data)
data = np.loadtxt("new100_dataMz_z_Pdet.txt").T
Mz = data[0]
z_redshift = data[1]
pdet = data[2]


parameters1D_1 = data[0] #np.array([1, 2, 3, 4, 5])
parameters1D_2 = data[1] #np.array([10, 20, 30, 40, 50])
function_values = data[2] #np.array([10, 20, 30, 40, 50])

# Create a 2D mesh
mesh1D_1, mesh1D_2 = np.meshgrid(parameters1D_1, parameters1D_2)

# Reshape the mesh to 1D arrays
mesh1D_1_flat = mesh1D_1.flatten()
mesh1D_2_flat = mesh1D_2.flatten()
function_values_flat = function_values.flatten()

# Define a range of values on the 2D mesh
new_parameters1D_1 = np.logspace(2.5, 8, 100)
new_parameters1D_2 = np.logspace(-2, np.log10(20), 100)

# Create a 2D mesh for the new parameters
new_mesh1D_1, new_mesh1D_2 = np.meshgrid(new_parameters1D_1, new_parameters1D_2)

# Perform interpolation using griddata
new_function_values = griddata((mesh1D_1_flat, mesh1D_2_flat), function_values_flat, (new_mesh1D_1, new_mesh1D_2), method='linear', fill_value=0.0)

# Create a contour plot
plt.contourf(new_mesh1D_1, new_mesh1D_2, new_function_values, cmap='viridis')
plt.colorbar(label='Function Values')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.title('Contour Plot of Interpolated Function Values')
plt.show()

