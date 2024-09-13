# This will make kde of intrinsic data with opt data find before
# and also use average of iterative KDEs and plot them
import utils_awkde as u_awkde
import utils_plot as u_plot
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
# get intrinsic data and get kde after smoothing the dat
data =np.loadtxt('/home/jsadiq/Research/E_awKDE/CatalogLISA/lensedPopIII/combined_intrinsicdata100years_Mz_z_withPlanck_cosmology.dat').T
# get relvant iterative data
frateh5 = h5.File('/home/jsadiq/Research/E_awKDE/CatalogLISA/lensedPopIII/json-params/samples/save_pdet/output_nonadaptiveWeightedKDEIterativereweight1000iterations.hdf5', 'r')


intMz = data[0]
intz = data[1]

origsample = np.vstack((np.log10(intMz) , intz)).T
################# for experimenting with intrinsic kde #####
# check scatter addition in z
logzp1 = np.log(1+intz)
val = 0.043#*3.0
scatter = logzp1 + np.random.normal(0, val, size=logzp1.shape)
bins_s=100
plt.hist(scatter, bins=bins_s, density=False, histtype='step', color='r' ,label='with Gauss-scatter {0:.4f}'.format(val))
plt.hist(logzp1, bins=bins_s, density=False, histtype='step', color='b', label='no scatter')
plt.title("bins={0}".format(bins_s))
plt.xlabel("log[1+z]", fontsize=15)
plt.legend()
plt.tight_layout()
plt.show()

retranfscatter =  np.exp(scatter) - 1
sample = np.vstack((np.log10(intMz) , retranfscatter)).T
# get kde using 
XXx = frateh5['LogMz'][:]
YYy = frateh5['z'][:]
kdevalslist = []
discard=100
for i in range(1000):
    current_kdeval  = frateh5['kde_iter{0:04}'.format(i)][:]
    kdevalslist.append(current_kdeval)
iterbwlist = frateh5['bandwidths'][:]
#iteralplist = frateh5['alphas'][:]
frateh5.close()

Mz_eval = np.logspace(2, 10, 200)[:, np.newaxis]
Mz_grid = np.logspace(2, 10, 200)
z_eval = np.logspace(-1, np.log10(20), 200)[:, np.newaxis]
z_grid = np.logspace(-1, np.log10(20), 200)

##### We will use adaptive or weighted KDE code here for 1/pdet for weights rather awkde
XX, YY = np.meshgrid(np.log10(Mz_grid), z_grid)
grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T
optbw, optalp = 0.1, 0.11
print("opt bw, alp = ", optbw, optalp)

current_kde, ZZ = u_awkde.kde_awkde(sample, grid_pts, global_bandwidth=optbw, alpha=optalp, ret_kde=True)
kdeval2D_intrinsic = ZZ.reshape(XX.shape)
TheoryMtot, Theory_z = intMz, retranfscatter
average_list =  np.percentile(kdevalslist[:], 50, axis=0)
p5 = np.percentile(kdevalslist[:], 5, axis=0)
p95 =  np.percentile(kdevalslist[:], 95, axis=0)

u_plot.ThreePlots(XXx, YYy, average_list, p95, p5, TheoryMtot, Theory_z, iternumber=1001, plot_name='combined_all')
u_plot.compare_twodimensionalKDEPlot(XXx, YYy, kdeval2D_intrinsic, average_list, kdeval2D_intrinsic)
