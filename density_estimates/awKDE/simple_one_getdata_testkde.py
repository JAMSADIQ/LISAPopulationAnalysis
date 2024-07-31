import numpy as np
import matplotlib.pyplot as plt
import utils_awkde as u_awkde

def generate_power_law_data(size, exponent=2.0, xmin=1, xmax=100):
    """Generate data following a truncated power law distribution."""
    r = np.random.uniform(0, 1, size)
    power_law_data = ((xmax ** (exponent - 1) - xmin ** (exponent - 1)) * r + xmin ** (exponent - 1)) ** (1 / (exponent - 1))
    return power_law_data

def generate_gaussian_peaks_data(means, sigmas, xmin=2, xmax=100, n_samples_per_peak=500):
    """
    Generate data for multiple Gaussian peaks.

    Parameters:
    - means: List of means for the Gaussian peaks.
    - sigmas: List of standard deviations (sigmas) for the Gaussian peaks.
    - xmin: Minimum x value.
    - xmax: Maximum x value.
    - n_samples_per_peak: Number of samples per Gaussian peak.

    Returns:
    - data: Array of generated data points.
    """
    if len(means) != len(sigmas):
        raise ValueError("The length of means and sigmas must be the same.")

    data = []
    for mu, sigma in zip(means, sigmas):
        peak_data = np.random.normal(loc=mu, scale=sigma, size=n_samples_per_peak)
        data.extend(peak_data)

    # Clip data to the specified range
    data = np.clip(data, xmin, xmax)

    return np.array(data)



def generate_uniform_data(size, low=2, high=100):
    """Generate data following a uniform distribution."""
    uniform_data = np.random.uniform(low, high, size)
    return uniform_data

# Generate power law data
power_law_data = generate_power_law_data(size=500, exponent=-2., xmin=2, xmax=100)
# Add Gaussian peaks to power law data
gaussian_peaks_data = generate_gaussian_peaks_data([20, 50], [5, 12], xmin=2, xmax=100, n_samples_per_peak=250)#250 in each peak so total 500 samples 

data_power_law_with_gaussian_peaks = np.concatenate((power_law_data, gaussian_peaks_data))

# Generate uniform data
uniform_data = generate_uniform_data(size=3000, low=2, high=100)

# Function to plot KDE
def plot_kde(data, label, ax):
    print(label, min(data), max(data))
    x_range = np.linspace(min(data), max(data), 1000)
    kde_vals = u_awkde.kde_awkde(data, x_range, global_bandwidth='scott', alpha=0.0) #bw silverman, alpha =0.5
    ax.plot(x_range, kde_vals, label=label)
    ax.set_title(label)
    ax.set_xlabel("x")
    ax.set_ylabel("pdf(x)")

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# KDE for power law with Gaussian peaks
#plot_kde(data_power_law_with_gaussian_peaks, "Power Law with Gaussian Peaks", axes[0])
plot_kde(gaussian_peaks_data, "Power Law with Gaussian Peaks", axes[0])

# KDE for uniform distribution
plot_kde(uniform_data, "Uniform Distribution", axes[1])

plt.tight_layout()
plt.show()

