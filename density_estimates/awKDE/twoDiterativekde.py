import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import json
import h5py as h5
import scipy
from scipy.interpolate import RegularGridInterpolator
import sys

import utils_awkde as u_awkde
import utils_plot as u_plot


#careful parsers 
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--datafilename',  default='combine_popIIIyear92_events.hdf',help='h5 file containing N samples for m1for all gw bbh event')
### For KDE in log parameter we need to add --logkde 
parser.add_argument('--logkde', action='store_true',help='if True make KDE in log params but results will be in onlog')
bwchoices= np.logspace(-2, 0, 15).tolist() 
parser.add_argument('--bw-grid', default= bwchoices, nargs='+', help='grid of choices of global bandwidth')
alphachoices = np.linspace(0., 1.0, 11).tolist()
parser.add_argument('--alpha-grid', nargs="+", default=alphachoices, type=float, help='grid of choices of sensitivity parameter alpha for local bandwidth')

# limits on KDE evulation: 
parser.add_argument('--reweightmethod', default='bufferkdevals', help='Only for gaussian sample shift method: we can reweight samples via buffered kdevals(bufferkdevals) or buffered kdeobjects (bufferkdeobject)', type=str)
parser.add_argument('--reweight-sample-option', default='reweight', help='choose either "noreweight" or "reweight" if reweight use fpop prob to get reweight sample (one sample for no bootstrap or no or multiple samples for poisson)', type=str)
parser.add_argument('--bootstrap-option', default='poisson', help='choose either "poisson" or "nopoisson" if None it will reweight based on fpop prob with single reweight sample for eaxh event', type=str)

#### buffer iteratio
parser.add_argument('--buffer-start', default=5, type=int, help='start of buffer in reweighting.')
parser.add_argument('--buffer-interval', default=100, type=int, help='interval of buffer that choose how many previous iteration resulkts we use in next iteration for reweighting.')

#plots and saving data
parser.add_argument('--pathplot', default='./', help='public_html path for plots', type=str)
parser.add_argument('--pathtag', default='re-weight-bootstrap_', help='public_html path for plots', type=str)
parser.add_argument('--output-filename', default='output_data_', help='write a proper name of output hdf files based on analysis', type=str)
opts = parser.parse_args()


#############for ln paramereter we need these
def prior_factor_function(samples):
    """ 
    LVC use uniform-priors for masses 
    in linear sclae. so
    reweighting need a constant factor

   note that in the reweighting function if we use input masses in log
   form so when we need to factor
   we need non-log mass  so we take exp
    """
    Mz_vals, z_vals = samples[:, 0], samples[:, 1]
    if opts.logkde:
        #it is assumed that samples are log10(Mz)
        factor = 1.0/10**(Mz_vals)
    else:
        factor = np.ones_like(Mz_vals)
    return factor


###### reweighting  methods ####################################
def get_random_sample(original_samples, bootstrap='poisson'):
    """without reweighting"""
    rng = np.random.default_rng()
    if bootstrap =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1))
    else:
        reweighted_sample = rng.choice(original_samples)
    return reweighted_sample


def get_reweighted_sample(original_samples, pdetvals, fpop_kde, bootstrap='poisson', prior_factor=prior_factor_function):
    """
    inputs 
    original_samples: list of mean of each event samples 
    fpop_kde: kde_object [GaussianKDE(opt alpha, opt bw and Cov=True)]
    kwargs:
    bootstrap: [poisson or nopoisson] from araparser option
    prior: for ln parameter in we need to handle non uniform prior
    return: reweighted_sample 
    one or array with poisson choice
    if reweight option is used
    using kde_object compute probabilty on original samples and 
    we compute rate using Vt as samples 
    and apply 
    use in np.random.choice  on kwarg 
    """
    # need to test this
    fkde_samples = fpop_kde.predict(original_samples)*1.0/pdetvals
    frate_atsample = fkde_samples * prior_factor(original_samples) 
    fpop_at_samples = frate_atsample/frate_atsample.sum() # normalize
    rng = np.random.default_rng()
    if bootstrap =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1), p=fpop_at_samples)
    else:
        reweighted_sample = rng.choice(original_samples, p=fpop_at_samples)

    return reweighted_sample


def median_bufferkdelist_reweighted_samples(original_samples, original_pdet, Log10_Mz_val, z_val, kdelist, bootstrap_choice='poisson', prior_factor=prior_factor_function):
    """
    using previous 100 ietrations pdf estimates (kde or rate)
    interpolate all of them to get values of KDE on
    original samples(means of PE samples)
    take the average of KDE values[at those samples]
    and normalize them and use them  as probablity in
    reweighting samples
    -------------
    inputs
    sample : original samples or mean of PE samples
    x_grid_kde : x-grid onto which kdes are computes
    kdelist : previous 100 kdes in iterations [buffer]
    bootstrap_choice :by default poisson or use from opts.bootstrap option
    prior_factor: for log10Mz parameter to take into account non uniform prior
    return:
    reweighted sample , pdet [none, one or array of values] based on poisson
    dsitrubution with mean 1.
    """
    interpkdeval_list = []
    for kde_values in kdelist:
        interp = RegularGridInterpolator((Log10_Mz_val, z_val), kde_values, bounds_error=False, fill_value=0.0)
        kde_interp_vals = interp(original_samples)
        #apply prior factor
        weighted_kde_interp_vals = kde_interp_vals*prior_factor(original_samples)*(1.0/original_pdet)
        interpkdeval_list.append(weighted_kde_interp_vals)
    mediankdevals = np.percentile(interpkdeval_list, 50, axis=0)
    norm_mediankdevals = mediankdevals/sum(mediankdevals)
    rng = np.random.default_rng()
    if bootstrap_choice =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1), p=norm_mediankdevals)
    else:
        reweighted_sample = rng.choice(original_samples, p=norm_mediankdevals)
    return reweighted_sample


def normdata(dataV):
    normalized_data = (dataV - np.min(dataV)) / (np.max(dataV) - np.min(dataV))
    return normalized_data

############################################################################

hdf_file = h5.File(opts.datafilename, 'r')
sampleslists_Mz = []
sampleslists_z = []
sampleslists_pdet = []
medianlist_Mz = []
medianlist_z = []
medianlist_pdet = []
#print(hdf_file.keys())
plt.figure()
for event_name in hdf_file.keys():
    data = hdf_file[event_name][...].T
    #print(data)
    data_Mz = data[0]
    data_z = data[1]
    data_pdet = data[2]
    # remove below if there is no issue with z >20
    indices = np.argwhere(data_z <= 20.0).flatten()
    data_z = data_z[indices]
    dataMz = data_Mz[indices]
    datapdet = data_pdet[indices]
    #plt.scatter(np.log10(dataMz), data_z, marker='+')
    plt.scatter(np.log10(data[0]), data[1], marker='+')
    sampleslists_Mz.append(dataMz)
    sampleslists_pdet.append(datapdet)
    sampleslists_z.append(data_z)
    medianlist_Mz.append(np.median(dataMz))
    medianlist_z.append(np.median(data_z))
    medianlist_pdet.append(np.median(datapdet))
plt.xlabel("Log10[M]")
plt.ylabel("z")
plt.title("100 samples per event")
plt.savefig("colored_samples_perevent_year92.png")
plt.show()
hdf_file.close()
median_arr_Mz = np.array(medianlist_Mz)
median_arr_z = np.array(medianlist_z)
median_arr_pdet = np.array(medianlist_pdet)
#sampleslists = np.vstack((np.array(sampleslists_Mz), np.array(sampleslists_z))).T

flat_Mz_all = np.concatenate(sampleslists_Mz).flatten()#np.array(sampleslists_Mz).flatten()
flat_z_all = np.concatenate(sampleslists_z).flatten()
flat_pdet_all = np.concatenate(sampleslists_pdet).flatten()

Mz_grid = np.logspace(2, 10, 200)
z_grid = np.logspace(-1, np.log10(20), 200)

##### We will use Weighted KDE code here for 1/pdet for weights rather awkde
XX, YY = np.meshgrid(np.log10(Mz_grid), z_grid)
grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T
sample = np.vstack((np.log10(median_arr_Mz) ,median_arr_z)).T
all_samples = np.vstack((np.log10(flat_Mz_all), flat_z_all)).T
print("total samples", len(flat_pdet_all), len(flat_z_all), all_samples.shape)

############### Code is expensive we got opt vals on median and save them below
alphagrid = opts.alpha_grid #
bwgrid = opts.bw_grid
##First median samples KDE
current_kde, errorkdeval, errorbBW, erroraALP = u_awkde.kde_twoD_with_do_optimize(sample, grid_pts, bwgrid, alphagrid, ret_kde=True, optimize_method='loocv')
ZZ = errorkdeval.reshape(XX.shape)
#priorfactor

u_plot.new2DKDE(XX, YY,  ZZ, np.log10(median_arr_Mz), median_arr_z, median_arr_pdet, iterN=0, title='KDEmedian', saveplot=True)
############## All data KDE #############################
ZZall = u_awkde.kde_awkde(all_samples, grid_pts, global_bandwidth=errorbBW, alpha=erroraALP, ret_kde=False)
ZZall = ZZall.reshape(XX.shape)

u_plot.new2DKDE(XX, YY,  ZZall, np.log10(flat_Mz_all), flat_z_all, flat_pdet_all, iterN=1, saveplot=True, title='KDEall')

########## ASTROPHYSICAL catalog DATA
data = np.loadtxt("intrinsicdata.dat").T
Theory_z = data[1]
TheoryMtot = data[0]
u_plot.ThreePlots(XX, YY, ZZall, TheoryMtot, Theory_z, plot_name='AllSamples')


#### Iterative weighted-KDE
iterbwlist = []
iteralplist = []
iterbwlist = []
iteralplist = []

frateh5 = h5.File(opts.outputfilename+'awkde_method_Data2DIterativeCase.hdf5', 'w')
dsetxx = frateh5.create_dataset('LogMz', data=XX)
dsetyy = frateh5.create_dataset('z', data=YY)

discard = 100#opts.buffer_start   # how many iterations to discard default =5
Nbuffer = 100#opts.buffer_interval #100 buffer [how many (x-numbers of ) previous iteration to use in reweighting with average of those past (x-numbers of ) iteration results
kdevalslist = []


for i in range(1000+discard):
    print("i - ", i)
    rwsamples = []
    for sampleMz, sample_z,  samples_pdet in zip(sampleslists_Mz, sampleslists_z, sampleslists_pdet):
        samples= np.vstack((np.log10(sampleMz), sample_z)).T
        if i < discard + Nbuffer :
            rwsample= get_reweighted_sample(samples, samples_pdet, current_kde, bootstrap=opts.bootstrap_option)
        else: 
            rwsample= median_bufferkdelist_reweighted_samples(samples, samples_pdet, np.log10(Mz_grid), z_grid, kdevalslist[-100:], bootstrap_choice=opts.bootstrap_option)
        rwsamples.append(rwsample)
    rwsamples = np.concatenate(rwsamples)
    print("iter", i, "  totalsamples = ", len(rwsamples))
    current_kde, current_kdeval, shiftedbw, shiftedalp =  u_awkde.kde_twoD_with_do_optimize(rwsamples, grid_pts, bwgrid, alphagrid, ret_kde=True, optimize_method='loocv')
    u_awkde.get_Ndimkde(np.array(rwsamples), grid_pts, alphagrid, bwgrid, ret_kde=True)
    current_kdeval = current_kdeval.reshape(XX.shape)
    kdevalslist.append(current_kdeval)
    iterbwlist.append(shiftedbw)
    iteralplist.append(shiftedalp)
    frateh5.create_dataset('kde_iter{0:04}'.format(i), data=current_kdeval)
    if  i > 0.0 and i %100 == 0:
        u_plot.ThreePlots(XX, YY, np.percentile(kdevalslist[-100:], 50, axis=0),  TheoryMtot, Theory_z ,iternumber=i, plot_name='AwavergeKDE2D')
        u_plot.histogram_datalist(iterbwlist[-100:], dataname='bw', pathplot='./', Iternumber=i)
        u_plot.histogram_datalist(iteralplist[-100:], dataname='alpha', pathplot='./', Iternumber=i)
    print(i, "step done ")
frateh5.create_dataset('bandwidths', data=iterbwlist)
frateh5.create_dataset('alphas', data=iteralplist)

frateh5.close()
# Calculate the average of all KDE after discard
average_list = np.percentile(kdevalslist[100:], 50, axis=0) 
u_plot.ThreePlots(XX, YY, average_list,  TheoryMtot, Theory_z, iternumber=1001, plot_name='Awcombined_all')


#alpha bw plots
u_plot.bandwidth_correlation(iterbwlist, number_corr=discard, error=0.02,  pathplot=opts.pathplot)
u_plot.bandwidth_correlation(iteralplist, number_corr=discard,  error=0.02, param='alpha',pathplot=opts.pathplot)
u_plot.bandwidth_correlation(iterbwlist, number_corr=discard, error=0.0, pathplot=opts.pathplot)
u_plot.bandwidth_correlation(iteralplist, number_corr=discard, error=0.0, param='alpha', pathplot=opts.pathplot)

