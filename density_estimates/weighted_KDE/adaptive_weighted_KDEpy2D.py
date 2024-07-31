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




def standardize_data(train_data, eval_data):
    """
    get standardize data (divide data by its standard deviation)
    use this for fit and evaluate kde
    for better results?
    """
    #dvide the data by std
    stds = np.std(train_data, axis=0)  # record the stds
    std_train_data = np.zeros_like(train_data)
    for dim, t_data in enumerate(train_data.T):
        std_train_data[:, dim] = t_data/np.std(t_data)#train_data[:, dim]/np.std(train_data[:, dim]) 
     
    std_eval_data = np.zeros_like(eval_data)
    for dim, data in enumerate(eval_data.T):
        std_eval_data[:, dim] = eval_data[:, dim]/stds[dim]
    return std_train_data, std_eval_data




#tests this on simple OneD and twoD datacase
#data = np.random.randn(2**6)
#dataeval = np.linspace(np.min(data), np.max(data), 100)
#standardize data
#data, dataeval = standardize_data(data, dataeval)
#kdevals = adaptive_weighted_kde(data, dataeval, alpha=0.0, bw=0.5, weights=None)
#adkdevals = adaptive_weighted_kde(data, dataeval, alpha=1.0, bw=0.5, weights=None)
#halfadkdevals = adaptive_weighted_kde(data, dataeval, alpha=0.5, bw=0.5, weights=data**2)
#plt.plot(dataeval,  kdevals, label='non-adaptive')
#plt.plot(dataeval,  adkdevals, label='adaptivefull')
#plt.plot(dataeval,  halfadkdevals, label='adaptive_weighted Squaredata')
#plt.legend()
#plt.show()


allPEdata = np.loadtxt('../data_files/Lense_4_years_events_randomsamples_extracted_Mz_z_pdet_mfpdet.txt').T
#allPEdata = np.loadtxt('/home/jsadiq/Research/E_awKDE/Github_repo/LISAPopulationAnalysis/density_estimates/data_files/Lense_4_years_events_randomsamples_medians_Mz_z.txt').T
all_Mz = allPEdata[0]
all_z = allPEdata[1]
all_pdet = allPEdata[-1]
data = np.vstack((np.log10(all_Mz), all_z)).T
print(len(all_z))
Mz_eval = np.logspace(2.5, 8, 200)
log10Mz_eval = np.log10(Mz_eval)
z_eval = np.logspace(-1, np.log10(20), 200)
log10Mz_mesh, z_mesh = np.meshgrid(log10Mz_eval, z_eval)
grid_pts = np.array(list(map(np.ravel, [log10Mz_mesh, z_mesh]))).T
#standardize data
stddata, stddataeval = standardize_data(data, grid_pts)
adkdevals = adaptive_weighted_kde(data, grid_pts, alpha=1.0, bw=0.2, weights=1.0/all_pdet)
adkdevals = adkdevals.reshape(z_mesh.shape)
stdadkdevals = adaptive_weighted_kde(stddata, stddataeval, alpha=1.0, bw=0.2, weights=1.0/all_pdet)
stdadkdevals = stdadkdevals.reshape(z_mesh.shape)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# Plot pcolormesh with logarithmic normalization on the left subplot
pcol = ax1.pcolormesh(log10Mz_mesh, z_mesh, adkdevals, norm=LogNorm(), shading='auto')
fig.colorbar(pcol, ax=ax1, label='unweighted kde')
ax1.contour(log10Mz_mesh, z_mesh, adkdevals, norm=LogNorm(), colors='black')
ax1.set_title('without standardization')
ax1.set_xlabel('Log10[Mz]', fontsize=16)
ax1.set_ylabel('z', fontsize=16)

pcol2 = ax2.pcolormesh(log10Mz_mesh, z_mesh, stdadkdevals, norm=LogNorm(), shading='auto')
fig.colorbar(pcol2, ax=ax2, label='unweighted kde')
ax2.set_title('with standardization')
ax2.contour(log10Mz_mesh, z_mesh, stdadkdevals, norm=LogNorm(), colors='black')
ax2.set_xlabel('Log10[Mz]', fontsize=16)
ax2.set_ylabel('z', fontsize=16)
# Display the plots
plt.tight_layout()
plt.show()


quit()
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
