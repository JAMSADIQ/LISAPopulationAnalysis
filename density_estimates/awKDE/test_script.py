import numpy as np
import matplotlib.pyplot as plt
import utils_awkde as u_awkde
import numpy as np
from scipy.optimize import minimize

# Test script
def test_kde_awkde():
    # Test 1: One-dimensional data
    x_1d = np.array([2, 3, 4, 5, 6, 7])
    x_grid_1d = np.linspace(0, 10, 20)

    y_1d = u_awkde.kde_awkde(x_1d, x_grid_1d)
    assert y_1d.shape == x_grid_1d.shape, "Test 1 Failed: Shape mismatch for one-dimensional data."

    # Test 2: Two-dimensional data
    x_2d = np.array([[2], [3], [4], [5], [6], [7]])
    x_grid_2d = np.linspace(0, 10, 100)[:, np.newaxis]

    y_2d = u_awkde.kde_awkde(x_2d, x_grid_2d)
    assert y_2d.shape == (100,), "Test 2 Failed: Shape mismatch for two-dimensional data."

    # Test 3: ret_kde=True
    kde_obj, y_with_kde = u_awkde.kde_awkde(x_1d, x_grid_1d, ret_kde=True)
    assert hasattr(kde_obj, 'fit'), "Test 3 Failed: KDE object does not have 'fit' method."
    assert y_with_kde.shape == x_grid_1d.shape, "Test 3 Failed: Shape mismatch when ret_kde=True."

    print("All tests passed.")

############# Some 1 D simple example with default bw and alpha #########

# Generate one-dimensional Gaussian data
np.random.seed(0)
x_1d = np.random.normal(loc=0, scale=1, size=100)

# Define the grid for evaluation
x_grid_1d = np.linspace(-5, 5, 1000)

# Apply KDE
y_1d = u_awkde.kde_awkde(x_1d, x_grid_1d)

# Plot the results
plt.figure(figsize=(8, 6))
plt.hist(x_1d, bins=30, density=True, alpha=0.5, label='Histogram of samples')
plt.plot(x_grid_1d, y_1d, label='KDE', color='red')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('KDE for One-Dimensional Gaussian Data')
plt.legend()
plt.show()
################## Simple 2D case ############################
# Generate two-dimensional Gaussian data
mean = [0, 0]
cov = [[1, 0], [0, 1]]  # Diagonal covariance matrix
x_2d = np.random.multivariate_normal(mean, cov, 100)

# Define the grid for evaluation
x_grid_2d_x = np.linspace(-5, 5, 100)
x_grid_2d_y = np.linspace(-5, 5, 100)
x_grid_2d = np.array(np.meshgrid(x_grid_2d_x, x_grid_2d_y)).T.reshape(-1, 2)

################## using  for loop for fixed choices of bw, alpha
# get optimized bw and alpha using loocv
#bandwidth_options = ['scott', 'silverman', 0.1, 0.5, 1.0]
#alpha_options = [0.0, 0.1, 0.3, 0.5, 1.0]
#optbw, optalp, _ = u_awkde.optimize_parameters(x_2d, bandwidth_options, alpha_options, method='loo_cv', fom_plot_name='showfom')

############### use scipy minimize, nelder-mead, many options to explore here
def objective(params, sample, method='loocv', n_splits=5):
    bwchoice, alpha = params
    print("method", method)
    print("bwchoice, alpha", bwchoice, alpha)
    try:
        if method == 'loocv':
            fom = u_awkde.loocv_awkde(sample, bwchoice, alpha)
        else:
            fom = u_awkde.kfold_cv_awkde(sample, bwchoice, alpha, n_splits)
    except ValueError as e:
        print(f"Encountered an error with params (alpha: {alpha}, bwchoice: {bwchoice}): {e}")
        fom = np.inf  # Assign a very high cost to invalid parameters
    return -fom  # We want to maximize the FoM, hence minimize its negative

initial_params = [0.5, 0.1]
#vounds has min tuple and max tuple for each params
bounds = [(0.01, 0.0), (1.0, 1.0)]
result = minimize(objective, initial_params, args=(x_2d, 'kfold', 4), method='Nelder-Mead')#, options={'xatol': 1e-7, 'disp': True})
print(result)
# Extract optimal alpha and bandwidth
optbw, optalp = result.x
print(f"Optimal alpha: {optalp}")
print(f"Optimal bandwidth choice: {optbw}")

# Apply KDE
y_2d = u_awkde.kde_awkde(x_2d, x_grid_2d, global_bandwidth=optbw, alpha=optalp)

# Reshape the output for plotting
y_2d = y_2d.reshape(100, 100)

# Plot the results
plt.figure(figsize=(8, 6))
plt.contourf(x_grid_2d_x, x_grid_2d_y, y_2d, levels=20, cmap='viridis')
plt.scatter(x_2d[:, 0], x_2d[:, 1], alpha=0.5, edgecolor='k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('KDE for Two-Dimensional Gaussian Data')
plt.colorbar(label='Density')
plt.show()

# Run the test script with pytest
#test_kde_awkde()
