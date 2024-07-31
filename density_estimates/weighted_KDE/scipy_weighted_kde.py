import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Step 1: Generate a 2-dimensional Gaussian distribution
np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]  # diagonal covariance
n_samples = 500

# Generate samples
data = np.random.multivariate_normal(mean, cov, n_samples)

# Step 2: Create a pattern (e.g., higher density in a specific region)
# Let's create a denser region around (1,1)
pattern_center = [1, 1]
pattern_cov = [[0.1, 0], [0, 0.1]]
n_pattern_samples = 100

# Generate pattern samples
pattern_data = np.random.multivariate_normal(pattern_center, pattern_cov, n_pattern_samples)
data_with_pattern = np.vstack([data, pattern_data]).T
print(data_with_pattern.shape)
# Step 3: Assign weights to the samples based on the pattern
weights = np.ones(n_samples + n_pattern_samples)
weights[-n_pattern_samples:] = 10  # Increase weight for pattern samples



# Step 4: Perform KDE with the weighted/unweighted samples
kde_weighted = stats.gaussian_kde(data_with_pattern, bw_method=0.2 , weights=weights)
kde_unweighted = stats.gaussian_kde(data_with_pattern, bw_method=0.2 )

# Create grid for evaluation
x_grid = np.linspace(-3, 3, 100)
y_grid = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_grid, y_grid)
xy_sample = np.vstack([X.ravel(), Y.ravel()])
# Evaluate the KDEs
density_weighted = np.reshape(kde_weighted(xy_sample).T, X.shape)
density_unweighted = np.reshape(kde_unweighted(xy_sample).T, X.shape)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)

# KDE with weights
axes[0].scatter(data_with_pattern[:, 0], data_with_pattern[:, 1], c='blue', s=2, label='Data Points')
axes[0].scatter(pattern_data[:, 0], pattern_data[:, 1], c='red', s=10, label='Pattern Points')
axes[0].contour(X, Y, density_weighted, levels=20, cmap='viridis')
axes[0].set_title('KDE with Weights')
axes[0].set_xlabel('X-axis')
axes[0].set_ylabel('Y-axis')
axes[0].legend()

# KDE without weights
axes[1].scatter(data_with_pattern[:, 0], data_with_pattern[:, 1], c='blue', s=2, label='Data Points')
axes[1].scatter(pattern_data[:, 0], pattern_data[:, 1], c='red', s=10, label='Pattern Points')
axes[1].contour(X, Y, density_unweighted, levels=20, cmap='viridis')
axes[1].set_title('KDE without Weights')
axes[1].set_xlabel('X-axis')
axes[1].legend()

plt.suptitle('2D Gaussian Distribution KDE Comparison')
plt.show()
