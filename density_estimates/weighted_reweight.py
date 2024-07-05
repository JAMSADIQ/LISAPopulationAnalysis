import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import h5py as h5
import sys
from KDEpy.TreeKDE import TreeKDE
import operator
import util_plots as u_plot
import scipy
from scipy.interpolate import RegularGridInterpolator


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--datafilename',default='Observed_200Random_PEsamples_Mz_DL_Pdet_z.hdf5' , help='h5 or txt file containing data for median and sigma for m1')
parser.add_argument('--type-data', choices=['gw_pe_samples', 'mock_data'], help='mock data for some power law with gaussian peak or gwtc  pe samples data. h5 files for two containing data for median and sigma for m1')
parser.add_argument('--fpopchoice', default='kde', help='choice of fpop to be rate or kde', type=str)
bwchoices= np.logspace(-1.5, 0, 15).tolist() #['scott', 'silverman']+ np.logspace(-1.5, 0, 15).tolist() # not ssure if this is good
parser.add_argument('--bw-grid', default= bwchoices, nargs='+', help='grid of choices of global bandwidth')
alphachoices = np.linspace(0.1, 1.0, 10).tolist()
#[0.1, 0.2, 0.3,0.4, 0.5, 0.7, 0.75, 0.8, 0.85, 0.87, 0.9, 0.95, 1.0]
parser.add_argument('--alpha-grid', nargs="+", default=alphachoices, type=float, help='grid of choices of sensitivity parameter alpha for local bandwidth')

parser.add_argument('--reweightmethod', default='bufferkdevals', help='Only for gaussian sample shift method: we can reweight samples via buffered kdevals(bufferkdevals) or buffered kdeobjects (bufferkdeobject)', type=str)
parser.add_argument('--reweight-sample-option', default='reweight', help='choose either "noreweight" or "reweight" if reweight use fpop prob to get reweight sample (one sample for no bootstrap or no or multiple samples for poisson)', type=str)
parser.add_argument('--bootstrap-option', default='poisson', help='choose either "poisson" or "nopoisson" if None it will reweight based on fpop prob with single reweight sample for eaxh event', type=str)
parser.add_argument('--pathplot', default='./', help='directory path for plots', type=str)
opts = parser.parse_args()

############################ adaptive-weighted-KDEpy ###################################
def adaptive_weighted_kde(train_data, eval_data, alpha=0.0, bw=0.5, weights=None, returnKDE=False):
    """
    Use KDEpy to get weighted 
    and adaptive kde 
    we want both in 2D and 1D cases
    return prepared_kde and kde_evaluated at values
    """
    # get kde on trian data with fixed global bandwidth
    pilot_kde = TreeKDE(bw=bw).fit(train_data)
    pilot_values = pilot_kde.evaluate(train_data)
    from scipy.stats import gmean
    g = gmean(pilot_values)
    loc_bw_factor = (pilot_values / g)**alpha
    bw_arr = bw/loc_bw_factor #check wang and wang paper
    if weights is not None:
         estimate = TreeKDE(bw=bw_arr).fit(train_data, weights)
    else:
        estimate = TreeKDE(bw=bw_arr).fit(train_data)
    if returnKDE==True:
        return estimate, estimate.evaluate(eval_data) 
    return estimate.evaluate(eval_data)

def leave_one_out_cross_validation(sample, bw, alpha):
    """
    use log of Likelihood as fom for loocv on samples
    to choose best bw and smoothing/local bw factor
    """
    fom = 0.
    #print(sample.shape[0])
    for i in range(sample.shape[0]):
        # for oneD case
        if sample.shape[0] == sample.shape[-1]:
            leave_one_sample, miss_sample = np.delete(sample, i, axis=0), np.array([[sample[i]]])
        else:
            leave_one_sample, miss_sample = np.delete(sample, i, axis=0), np.array([sample[i]])
        #print(miss_sample, miss_sample.shape)
        #quit()
        y = adaptive_weighted_kde(leave_one_sample,  miss_sample, alpha=alpha, bw=bw, weights=None)
        fom += np.log(y)
    return fom

def get_optimized_bw_alpha_using_loocv(sample, bwgrid, alphagrid):
    """
    using fom from loocv find optimized bw and alpha
    """
    FOM= {}
    for gbw in bwgrid:
        for alphaval in alphagrid:
            FOM[(gbw, alphaval)] = leave_one_out_cross_validation(sample, gbw, alphaval)
    optval = max(FOM.items(), key=operator.itemgetter(1))[0]
    optbw, optalpha  = optval[0], optval[1]
    maxFOM = max(FOM)
    return  FOM, optbw, optalpha, maxFOM

def get_opt_params_and_kde(samplevalues, x_gridvalues, alphagrid, bwgrid):
    """
    inputs: samplevalues, x_gridvalues, alphagrid, bwgrid
        make sure order of alpha and bwds
    return: kdeval, optbw, optalpha
      make sure of order of outputs
    """
    FOMdict, optbw, optalpha, maxFOM = get_optimized_bw_alpha_using_loocv(samplevalues,
    alphagrid, bwgrid)
    kdeval = adaptive_weighted_kde(samplevalues, x_gridvalues, alpha=optalpha, bw=optbw, weights=None)
    print("bw, alp = ", optbw, optalpha)
    return kdeval, optbw, optalpha

#normalize data
def normdata(dataV):
    normalized_data = (dataV - np.min(dataV)) / (np.max(dataV) - np.min(dataV))
    return normalized_data


################## Iterative Reweighting #######################
###### Gaussian samples for each event and reweight them with fpop prob
def prior_factor_function(samples):
    """
    KDE is computed on Log10[Mz] and z 
    but Mz is uniform in prior for PE samples
    samples is vstacked so we need Mz samples
    to be non-log and z will remain the same
    """
    log10Mz_vals, z_vals = samples[:, 0], samples[:, 1]
    ln_Mz = log10Mz_vals * np.log(10)
    factor = 1.0/(np.exp(ln_Mz))
    return factor

def get_random_sample(original_samples, bootstrap='poisson'):
    """
    return random sample without reweighting with or without
    Poisson distribution
    ----------
    original_samples : samples of an event
    kwargs :
        bootstrap : poisson
                    by default 
                choose nopoisson or None for non poisson distibution
                in that case only one sample will be returned
    ----------
    return poisson distributed random (0, 1 or more) samples from given
    samples of an event 
    """
    rng = np.random.default_rng()
    if bootstrap =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1))
    else:
        reweighted_sample = rng.choice(original_samples)
    return reweighted_sample


def get_reweighted_sample(original_samples, original_pdet, fpop_kde, sample_option='reweight', bootstrap='poisson',  prior_factor=prior_factor_function):
    """
    reweight given samples for an event using fpop (pdf of population/rate) 
    from all events sample as probabilty for each sample
    -------------
    original_samples : as name suggests, the samples of an event
    fpop_kde : previous pdf [kde/rate] estimate from all events samples
    sample_option : we always reweight of this is not needed
    inputs 
    original_samples: list of mean of each event samples 
    fpop_kde: kde_object [GaussianKDE(opt alpha, opt bw and Cov=True)]
    kwargs:
    bootstrap: [poisson or nopoisson] from argparser option
    prior_factor: for log10Mz parameter to take into account non uniform prior
    ------------
    return : reweighted_sample and reweighted_pdet
    of events whose all  samples are given
    as original_samples
    either  none, one or array of  samples due to poisson distribution
    """
    fpop_at_samples = fpop_kde.evaluate(original_samples)
    #apply prior factor
    fpop_at_samples *=  prior_factor(original_samples)
    fpop = fpop_at_samples/fpop_at_samples.sum()
    if bootstrap =='poisson':
        #find mostly likely Indices as we need it for pdet
        likely_indices = np.random.choice(np.arange(len(original_samples)), np.random.poisson(1), p=fpop)
    else:
        likely_indices = np.random.choice(np.arange(len(original_samples)), p=fpop)

    reweighted_sample = original_samples[likely_indices]
    reweighted_pdet = original_pdet[likely_indices]
    return reweighted_sample, reweighted_pdet


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
        kde_interp_vals  *= prior_factor(original_samples)
        interpkdeval_list.append(kde_interp_vals)
    mediankdevals = np.percentile(interpkdeval_list, 50, axis=0)
    fpop = mediankdevals/sum(mediankdevals)
    if bootstrap_choice =='poisson':
        #find mostly likely Indices as we need it for pdet
        likely_indices = np.random.choice(np.arange(len(original_samples)), np.random.poisson(1), p=fpop)
    else:
        likely_indices = np.random.choice(np.arange(len(original_samples)), p=fpop)

    reweighted_sample = original_samples[likely_indices]
    reweighted_pdet = original_pdet[likely_indices]
    return reweighted_sample, reweighted_pdet

###############################################################################

###Intrinsic Data
Intdata = np.loadtxt("Intrinsic_rate_data_Mz_KDEMz_z_KDEz.txt").T
IntM, IntKDEM, IntZ, IntKDEz = Intdata[0], Intdata[1], Intdata[2], Intdata[3]
IntKDEM = normdata(IntKDEM)
IntKDEz = normdata(IntKDEz)

###################################PE sample Data
### STEP I: get data [create mock dat or call hdf file for mock or gw data]
hdf_file = h5.File(opts.datafilename, 'r')
sampleslists_Mz = []
sampleslists_DL = []
sampleslists_z = []
sampleslists_pdet = []
medianlist_Mz = []
medianlist_DL = []
medianlist_z = []

for group_name in hdf_file.keys():
    #event number
    group = hdf_file[group_name]
    dataMz = group['M'][()]
    dataDL = group['DL'][()]
    dataPdet = group['pdet'][()]
    data_z = group['z'][()] 
    sampleslists_Mz.append(dataMz)
    sampleslists_DL.append(dataDL)
    sampleslists_pdet.append(dataPdet)
    sampleslists_z.append(data_z)
    medianlist_Mz.append(np.median(dataMz))
    medianlist_DL.append(np.median(dataDL))
    medianlist_z.append(np.median(data_z))

hdf_file.close()
median_arr_Mz = np.array(medianlist_Mz)
median_arr_DL = np.array(medianlist_DL)
median_arr_z = np.array(medianlist_z)

#sampleslists = np.vstack((np.array(sampleslists_Mz), np.array(sampleslists_z))).T
flat_Mz_all = np.concatenate(sampleslists_Mz)#np.array(sampleslists_Mz).flatten()
flat_z_all = np.concatenate(sampleslists_z) #.flatten()
flat_DL_all = np.concatenate(sampleslists_DL) #.flatten()
flat_pdet_all = np.concatenate(sampleslists_pdet) #.flatten()

#### Plot histogram and median awKDE to make sure it work
#u_plot.plot_hist_Mz_z(flat_Mz_all, median_arr_Mz, flat_z_all, median_arr_z)

###############################################################################
#Fix non-sense samples?? can remove this for reweight case
indices = np.argwhere(flat_z_all < 20.0)
flat_z_all  =  flat_z_all[indices]
flat_Mz_all  =  flat_Mz_all[indices]
flat_pdet_all  =  flat_pdet_all[indices]
print("shape = ", flat_z_all.shape, flat_pdet_all.shape, flat_Mz_all.shape)
################## Grid for KDE evalution
zgrid = np.logspace(-2, np.log10(20), 200)
Mzgrid = np.logspace(2.5, 8, 200)  #need in prior factor
LogMzgrid = np.log10(Mzgrid)
prior_factor_XX, prior_factor_YY = np.meshgrid(1.0/Mzgrid/np.log(10), np.ones_like(zgrid))# we need to factor


###########For OneD KDE we need this way
Mz_eval = np.logspace(2.5, 8, 200)[:, np.newaxis]
z_eval = np.logspace(-2, np.log10(20), 200)[:, np.newaxis]
DL_eval = np.logspace(np.log10(44), np.log10(230421), 300)
##### We will use Weighted KDE code here for 1/pdet for weights rather awkde
_1_over_XX, nl_1_over_YY = np.meshgrid(1.0/Mzgrid/np.log(10), np.ones_like(z_eval))# we need to factor
XX, YY = np.meshgrid(np.log10(Mz_eval), z_eval)
grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T
sample = np.vstack((np.log10(median_arr_Mz) ,median_arr_z)).T
all_samples = np.hstack((np.log10(flat_Mz_all), flat_z_all))
print(sample.shape, all_samples.shape)


############### Code is expensive we got opt vals on median and save them below
#alphagrid = opts.alpha_grid #[1.0]
#bwgrid = opts.bw_grid
#  OneD KDEs or Mz, z, DL on medians to check if it works
#kdeval_Mz, bw_Mz, alpha_Mz = get_opt_params_and_kde(np.log10(median_arr_Mz), np.log10(Mz_eval), alphagrid, bwgrid)
#kdeval_z, bw_z, alpha_z = get_opt_params_and_kde(median_arr_z, z_eval, alphagrid, bwgrid)
#kdeval_Mz = normdata(kdeval_Mz,)
### Saved Optimized params for Adaptive
bwMz, alpMz =  0.2, 0.6105402296585326
bwz, alpz =  0.8, 0.03162277660168379
bw2D, alp2D =  0.8, 1.0 #0.30000000000000004, 0.47705826961439296

# lets Try 2D case
#ZZ, bw2D, alp2D = get_opt_params_and_kde(sample, grid_pts, alphagrid, bwgrid)

kde_object, ZZ = adaptive_weighted_kde(sample, grid_pts, alpha=alp2D, bw=bw2D, weights=None, returnKDE=True)
ZZ = ZZ.reshape(XX.shape)


################# Check this  properly
##### Before Reweighting we Try with Prior Factor correction to see?
ZZ2 = ZZ*prior_factor_XX 

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
surface = ax1.plot_surface(XX, YY, ZZ, cmap='viridis', norm=LogNorm())
fig.colorbar(surface, label='Log Scale')
ax1.set_title('Mesh Plot')
# Plot as contour
ax2 = fig.add_subplot(122)
contour = ax2.contour(XX, YY, ZZ, cmap='viridis')
ax2.set_title('No-Prior Factor')
plt.savefig("Without_prior_factor_median2DKDE.png")
#plt.show()
plt.close()

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
#ax1.plot_surface(XX, YY, ZZ2, cmap='viridis')
surface = ax1.plot_surface(XX, YY, ZZ2, cmap='viridis', norm=LogNorm())
fig.colorbar(surface, label='Log Scale')
ax1.set_title('Mesh Plot')
# Plot as contour
ax2 = fig.add_subplot(122)
contour = ax2.contour(np.exp(XX)/np.log(10), YY, ZZ2, cmap='viridis')
ax2.set_title('with Prior_factor')
ax2.set_xscale('log')
plt.savefig("With_prior_factor_median2DKDE.png")
#plt.show()
plt.close()

#### Get 1D plot from2D data using scipy.integrate.simpson
from scipy.integrate import simpson
def  ThreePlots(XX, YY, ZZ,  IntM, IntKDEM, IntZ, IntKDEz, logKDE=True,iternumber=1):

#from scipy.integrate import simpson
# Integrate along the Mz-axis Now we compute 2D KDE using log10M not M
    kde_Mz = simpson(y=ZZ, x=YY, axis=0)
    # Integrate along the y-axis
    if logKDE==False:
        nonln_XX = np.exp(XX)/np.log(10) 
        kde_Z = simpson(y=ZZ, x=nonln_XX, axis=1)
    else:
        kde_Z = simpson(y=ZZ, x=XX, axis=1)

    kde_Mz = normdata(kde_Mz)
    kde_Z = normdata(kde_Z)
    
    # Plot the original function
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    #ax1 = fig.add_subplot(121, projection='3d')
    axs[1,0].contour(XX, YY, ZZ, cmap='viridis')
    axs[1,0].set_xlabel('Log10[Mz]')
    axs[1,0].set_ylabel('z')
    
    # Plot the integrated results
    #ax2 = fig.add_subplot(122)
    axs[0,0].plot(np.log10(Mz_eval), kde_Mz, label='obs')
    axs[0,0].plot(IntM, IntKDEM, 'k--', label='Intrinsic')
    axs[0,0].legend()
    #axs[0,0].set_xlabel('Log10[Mz]')
    axs[0,0].set_ylabel('p(Log10[Mz])')
    axs[0,0].set_title('Integrated Results')
    
    #ax3 = fig.add_subplot(123)
    #axs[1,1].plot(z_eval, kde_Mz, label='Integral along y')
    #rotate plot
    axs[1,1].plot(kde_Z, z_eval,label='obs')
    #axs[1,1].plot(IntZ, IntKDEz, 'k--', label='Intrinsic')
    axs[1,1].plot(IntKDEz, IntZ, 'k--', label='Intrinsic')
    #axs[1,1].set_ylabel('z')
    axs[1,1].set_xlabel('p(z)')
    axs[1,1].legend()
    axs[1,1].set_title('Integrated Results')
    axs[0, 1].axis('off')
    plt.tight_layout()
    
    #plt.savefig("IterativeKDE_plot_Iter{0}.png".format(iternumber))
    plt.show()
    return  0 
#ThreePlots(XX, YY, ZZ,  IntM, IntKDEM, IntZ, IntKDEz)
ThreePlots(XX, YY, ZZ2,  IntM, IntKDEM, IntZ, IntKDEz, logKDE=False)

weights = 1.0/flat_pdet_all 
#weights /= np.sum(weights)

print(len(weights), all_samples.shape)

############## Because we have 200*91 samples this below is too expensive
#kdeobject, kdevals = adaptive_weighted_kde(all_samples, grid_pts, alpha=alp2D, bw=bw2D, weights=weights, returnKDE=True)
## Make a plot to check
#kdevals  = kdevals.reshape(XX.shape)
#ThreePlots(XX, YY, kdevals,  IntM, IntKDEM, IntZ, IntKDEz)
#

#### Iterative weighted-KDE

#frateh5 = h5.File('saved_Data2DIterativeCase.hdf5', 'w')
#dsetxx = frateh5.create_dataset('LogMz', data=XX)
#dsetyy = frateh5.create_dataset('z', data=YY)

discard = 10
kdevalslist = []
for i in range(1000+discard):
    rwpdet = []
    rwsamples = []
    for samples_Mz, samples_z, samples_pdet  in zip(sampleslists_Mz, sampleslists_z, sampleslists_pdet):
        samples = np.vstack((np.log10(samples_Mz), samples_z)).T
        if i <= 100+discard:
            rwsample_k, rwpdet_k = get_reweighted_sample(samples, samples_pdet, kde_object, bootstrap='poisson', prior_factor=prior_factor_function)
        else:
            rwsample_k, rwpdet_k =  median_bufferkdelist_reweighted_samples(samples, samples_pdet, LogMzgrid, zgrid, kdevalslist[-3:], bootstrap_choice='poisson', prior_factor=prior_factor_function)
        rwsamples.append(rwsample_k)
        rwpdet.append(rwpdet_k)
    rwsamples = np.concatenate(rwsamples)
    print(len(rwsamples), rwsamples.shape)
    rwpdet = np.concatenate(rwpdet) 
    weights = 1.0/rwpdet
    kdeobject, kdevals = adaptive_weighted_kde(rwsamples, grid_pts, alpha=alp2D, bw=bw2D, weights=weights, returnKDE=True)
    kdevals = kdevals.reshape(XX.shape)
    kdevalslist.append(kdevals)
         
    if  i %50 == 0:
        ThreePlots(XX, YY, kdevals,  IntM, IntKDEM, IntZ, IntKDEz, iternumber=i)

    #frateh5.create_dataset('rate_iter{0:04}'.format(i), data=kdevals)
    print(i, "step done ")

#frateh5.close()
# Transpose the list of lists to get lists of corresponding elements
transposed_lists = list(map(list, zip(*kdevalslist)))
# Calculate the average of each corresponding element
average_list = [sum(values) / len(values) for values in transposed_lists]
ThreePlots(XX, YY, kdevals,  IntM, IntKDEM, IntZ, IntKDEz, iternumber=1010)
