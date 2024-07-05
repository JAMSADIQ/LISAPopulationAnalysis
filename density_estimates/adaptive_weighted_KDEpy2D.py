#This script will use KDEpy 2D for weighted KDE and also try to implement
from KDEpy.TreeKDE import TreeKDE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def adaptive_weighted_kde(train_data, eval_data, alpha=0.0, bw=0.5, weights=None):
    """
    Use KDEpy to get weighted 
    and adaptive kde 
    we want both in 2D and 1D cases
    """
    # get kde on trian data with fixed global bandwidth
    pilot_kde = TreeKDE(bw=bw).fit(train_data)
    pilot_values = pilot_kde.evaluate(train_data)
    from scipy.stats import gmean
    g = gmean(pilot_values)
    loc_bw_factor = (pilot_values / g) **alpha
    bw_arr = bw/loc_bw_factor #check wang and wang paper
    if weights is not None:
        estimate = TreeKDE(bw=bw_arr).fit(train_data, weights)
    else: 
        estimate = TreeKDE(bw=bw_arr).fit(train_data)
    return estimate.evaluate(eval_data)



#tests this on simple OneD and twoD datacase
data = np.random.randn(2**6)
dataeval = np.linspace(np.min(data), np.max(data), 100)
kdevals = adaptive_weighted_kde(data, dataeval, alpha=0.0, bw=0.5, weights=None)
adkdevals = adaptive_weighted_kde(data, dataeval, alpha=1.0, bw=0.5, weights=None)
halfadkdevals = adaptive_weighted_kde(data, dataeval, alpha=0.5, bw=0.5, weights=data**2)
plt.plot(dataeval,  kdevals)
plt.plot(dataeval,  adkdevals, label='adaptivefull')
plt.plot(dataeval,  halfadkdevals, label='adaptive_weighted Squaredata')
plt.legend()
plt.show()

#Lets Try 2-d data

# Create 2D data of shape (obs, dims)
rndgen = np.random.RandomState(seed=3575)  # ESEL
# Gaussian
mean = 3.
sigma = .25
# a^2 * x * exp(-a * x)
a = 100.
n_samples = 1000
logE_sam = rndgen.normal(mean, sigma, size=n_samples)
# From pythia8: home.thep.lu.se/~torbjorn/doxygen/Basics_8h_source.html
u1, u2 = rndgen.uniform(size=(2, n_samples))
sigma_sam = -np.log(u1 * u2) / a
# Shape must be (n_points, n_features)
sample = np.vstack((logE_sam, sigma_sam)).T
# Evaluate at dense grid
minx, maxx = np.amin(sample[:, 0]), np.amax(sample[:, 0])
miny, maxy = np.amin(sample[:, 1]), np.amax(sample[:, 1])

x = np.linspace(minx, maxx, 100)
y = np.linspace(miny, maxy, 100)

XX, YY = np.meshgrid(x, y)
grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T

kdevals = adaptive_weighted_kde(sample, grid_pts, alpha=0.0, bw=0.1, weights=None) 
kdevals = kdevals.reshape(XX.shape)

# These are bin edges for pcolormesh, len = len(x) + 1
dx2, dy2 = (x[1] - x[0]) / 2., (y[1] - y[0]) / 2.
bx = np.concatenate((x - dx2, [x[-1] + dx2]))
by = np.concatenate((y - dy2, [y[-1] + dy2]))

color_of_points = '#351322'
plt.figure()
plt.pcolormesh(bx, by,kdevals, cmap="Blues", norm=LogNorm(), shading="flat")
plt.scatter(logE_sam, sigma_sam, marker=".", color=color_of_points,
            edgecolor="none", s=30)
plt.title("KDE log PDF + original sample")
plt.show()
