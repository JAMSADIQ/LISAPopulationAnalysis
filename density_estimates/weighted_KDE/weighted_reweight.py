import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import utils_plot as u_plot
import argparse
import h5py as h5
import sys
from KDEpy.TreeKDE import TreeKDE
import operator
import scipy
from scipy.interpolate import RegularGridInterpolator


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--datafilename',default='../data_files/Lense_4_years_events_randomsamples_Mz_z_pdet_mfpdet.hdf' , help='h5 or txt file containing data for median and sigma for m1')
parser.add_argument('--type-data', choices=['gw_pe_samples', 'mock_data'], help='mock data for some power law with gaussian peak or gwtc  pe samples data. h5 files for two containing data for median and sigma for m1')
parser.add_argument('--fpopchoice', default='kde', help='choice of fpop to be rate or kde', type=str)
bwchoices= np.logspace(-1.5, 0, 15).tolist() #['scott', 'silverman']+ np.logspace(-1.5, 0, 15).tolist() # not ssure if this is good
parser.add_argument('--bw-grid', default= bwchoices, nargs='+', help='grid of choices of global bandwidth')
alphachoices = np.linspace(0.1, 1.0, 11).tolist()
#[0.1, 0.2, 0.3,0.4, 0.5, 0.7, 0.75, 0.8, 0.85, 0.87, 0.9, 0.95, 1.0]
parser.add_argument('--alpha-grid', nargs="+", default=alphachoices, type=float, help='grid of choices of sensitivity parameter alpha for local bandwidth')

parser.add_argument('--reweightmethod', default='bufferkdevals', help='Only for gaussian sample shift method: we can reweight samples via buffered kdevals(bufferkdevals) or buffered kdeobjects (bufferkdeobject)', type=str)
parser.add_argument('--reweight-sample-option', default='reweight', help='choose either "noreweight" or "reweight" if reweight use fpop prob to get reweight sample (one sample for no bootstrap or no or multiple samples for poisson)', type=str)
parser.add_argument('--bootstrap-option', default='poisson', help='choose either "poisson" or "nopoisson" if None it will reweight based on fpop prob with single reweight sample for eaxh event', type=str)
parser.add_argument('--pathplot', default='./', help='directory path for plots', type=str)
opts = parser.parse_args()

############################ adaptive-weighted-KDEpy ###################################
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


def adaptive_weighted_kde(train_data, eval_data, bw=0.5, alpha=0., weights=None, returnKDE=False, standardize=False):
    """
    Use KDEpy to get weighted 
    and adaptive kde 
    we want both in 2D and 1D cases
    return prepared_kde and kde_evaluated at values
    """
    if standardize == True:
        train_data, eval_data = standardize_data(train_data, eval_data)
    # get kde on trian data with fixed global bandwidth
    pilot_kde = TreeKDE(bw=bw).fit(train_data)
    pilot_values = pilot_kde.evaluate(train_data)
    from scipy.stats import gmean
    g = gmean(pilot_values)
    loc_bw_factor = (pilot_values / g)**alpha
    bw_arr = bw/loc_bw_factor #check wang and wang paper
    if weights is not None:
        #if np.sum(weights)!=1.0:
        #    weights /=weights.sum()
        estimate = TreeKDE(bw=bw_arr).fit(train_data, weights)
    else:
        estimate = TreeKDE(bw=bw_arr).fit(train_data)
    if returnKDE==True:
        return estimate, estimate.evaluate(eval_data) 
    return estimate.evaluate(eval_data)


def leave_one_out_cross_validation(sample, bw, alpha, weights=None, standardize=False):
    """
    use log of Likelihood as fom for loocv on samples
    to choose best bw and smoothing/local bw factor
    """
    fom = 0.
    for i in range(sample.shape[0]):
        # for oneD case
        #if sample.shape[0] == sample.shape[-1]:
        #    leave_one_sample, miss_sample = np.delete(sample, i, axis=0), np.array([[sample[i]]])
        #else:
        leave_one_sample, miss_sample = np.delete(sample, i, axis=0), np.array([sample[i]])
        y = adaptive_weighted_kde(leave_one_sample,  miss_sample, bw=bw, alpha=alpha, weights=weights, standardize=standardize)
        fom += np.log(y)
    return fom

from sklearn.model_selection import KFold
def k_fold_cross_validation(sample, bw, alpha, k=2, weights=None, standardize=False):
    """
    Evaluate the K-fold cross validated log likelihood for an awKDE with
    specific bandwidth and sensitivity (alpha) parameters
    """

    fomlist = []
    kfold = KFold(n_splits=k, shuffle=True, random_state=None)
    for train_index, test_index in kfold.split(sample):
        train, test = sample[train_index], sample[test_index]
        y = adaptive_weighted_kde(train, test,  alpha=alpha, bw=bw, weights=None, standardize=standardize)
        # Figure of merit : log likelihood for training samples
        fomlist.append(np.sum(np.log(y)))

    # Return the sum over all K sets of training samples
    return np.sum(fomlist)


def get_optimized_bw_alpha_using_cv(sample, bwgrid, alphagrid, cv_method='loocv',  k=2, weights=None, standardize=False):
    """
    using fom from loocv/ k-fold find optimized bw and alpha
    """
    FOM= {}
    for gbw in bwgrid:
        for alphaval in alphagrid:
            print("loop bw, alpha = ", gbw, alphaval)
            if cv_method == 'loocv':
                FOM[(gbw, alphaval)] = leave_one_out_cross_validation(sample, gbw, alphaval, weights=weights, standardize=standardize)
            else:
                print("k-fold cross validation with k={0}".format(k))
                FOM[(gbw, alphaval)] = k_fold_cross_validation(sample, gbw, alphaval, k=k, weights=weights, standardize=standardize)
    optval = max(FOM.items(), key=operator.itemgetter(1))[0]
    optbw, optalpha  = optval[0], optval[1]
    maxFOM = max(FOM)
    return  FOM, optbw, optalpha, maxFOM

def get_opt_params_and_kde(samplevalues, x_gridvalues, bwgrid ,alphagrid, cv_method='loocv', k=2, weights=None, standardization=False):
    """
    inputs: samplevalues, x_gridvalues, alphagrid, bwgrid
        make sure order of alpha and bwds
    return: kdeval, optbw, optalpha
      make sure of order of outputs
    """
    FOMdict, optbw, optalpha, maxFOM = get_optimized_bw_alpha_using_cv(samplevalues, bwgrid, alphagrid, cv_method=cv_method,  k=k, weights=None, standardize=False)
    kde_object, kdeval = adaptive_weighted_kde(samplevalues, x_gridvalues, alpha=optalpha, bw=optbw, weights=weights, standardize=standardization, returnKDE=True)
    print("bw, alp = ", optbw, optalpha)
    return kdeval, optbw, optalpha, kde_object

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
    # we need standardization here too on original samples
    original_samples, s_repeat = standardize_data(original_samples, original_samples,)
    fpop_at_samples = fpop_kde.evaluate(original_samples)
    #apply prior factor
    #fpop_at_samples *=  prior_factor(original_samples)
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
    #adding standardization here to see 
    original_samples, s_repeat = standardize_data(original_samples, original_samples,)
    mediankdevals = np.percentile(kdelist, 50, axis=0)
    interp = RegularGridInterpolator((Log10_Mz_val, z_val), mediankdevals.T, bounds_error=False, fill_value=0.0)
    kde_interp_vals = interp(original_samples)#*prior_factor(original_samples)
    fpop = kde_interp_vals/sum(kde_interp_vals)
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
data_intrinsic = np.loadtxt("../data_files/combined_intrinsicdata100years_Mz_z_withPlanck_cosmology.dat").T
TheoryMtot = data_intrinsic[0]
Theory_z = data_intrinsic[1]

##PE sample Data
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
    data_pdet = data[-1] #sigmacorrected pdet
    #indices = np.argwhere(data_z <= 20.0).flatten()
    data_z = data_z#[indices]
    dataMz = data_Mz#[indices]
    datapdet = data_pdet#[indices]
    plt.scatter(np.log10(dataMz), data_z, marker='+')
    sampleslists_Mz.append(dataMz)
    sampleslists_pdet.append(datapdet)
    sampleslists_z.append(data_z)
    medianlist_Mz.append(np.median(dataMz))
    medianlist_z.append(np.median(data_z))
    medianlist_pdet.append(np.median(datapdet))

hdf_file.close()
plt.xlabel("Log10[M]")
plt.ylabel("z")
plt.title("100 samples per event 4 years")
plt.savefig("colored_lensed_100samples_per_event_4years.png")
#plt.show()
plt.close()

##data into arrays
median_arr_Mz = np.array(medianlist_Mz)
median_arr_z = np.array(medianlist_z)
median_arr_pdet = np.array(medianlist_pdet)

flat_Mz_all = np.concatenate(sampleslists_Mz).flatten()#np.array(sampleslists_Mz).flatten()
flat_z_all = np.concatenate(sampleslists_z).flatten()
flat_pdet_all = np.concatenate(sampleslists_pdet).flatten()

#### grid points for evaluation
Mz_eval = np.logspace(2, 10, 200)[:, np.newaxis]
Mz_grid = np.logspace(2, 10, 200)
z_eval = np.logspace(-1, np.log10(20), 200)[:, np.newaxis]
z_grid = np.logspace(-1, np.log10(20), 200)

##### We will use Weighted KDE code here for 1/pdet for weights rather awkde
XX, YY = np.meshgrid(np.log10(Mz_grid), z_grid, indexing='xy')
grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T
prior_factor_XX, prior_factor_YY = np.meshgrid(1.0/Mz_grid/np.log(10), np.ones_like(z_grid))# we need to factor

##### We will use Weighted KDE code here for 1/pdet for weights rather awkde
_1_over_XX, nl_1_over_YY = np.meshgrid(1.0/Mz_grid/np.log(10), np.ones_like(z_grid))# we need to factor
med_sample = np.vstack((np.log10(median_arr_Mz), median_arr_z)).T
all_samples = np.hstack((np.log10(flat_Mz_all), flat_z_all))
print("data shapes", med_sample.shape, all_samples.shape)


############### Code is expensive we got opt vals on median and save them below
alphagrid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #opts.alpha_grid #[1.0]
bwgrid = np.logspace(-1.3, -0.3, 15).tolist()

#on medians to get first optimized results and use same if we want to intrinsic KDE (weights not needed for optimization)

ZZ, shiftedbw, shiftedalp, kde_object = get_opt_params_and_kde(med_sample, grid_pts, bwgrid, alphagrid, cv_method='2-fold', k=4, weights=1.0/median_arr_pdet, standardization=True)
ZZ = ZZ.reshape(XX.shape)
print("opt bw, alp = ", shiftedbw, shiftedalp)

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

u_plot.ThreePlots(XX, YY, ZZ2,  TheoryMtot, Theory_z, logKDE=True,  iternumber=0, plot_name='initial')

#quit()
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

discard = 100
kdevalslist = []
iterbwlist = [] 
iteralplist = [] 
frateh5 = h5.File('output_k4fold_adpative_weighted_kdepy_withstandardization.hdf5', 'w')
dsetxx = frateh5.create_dataset('LogMz', data=XX)
dsetyy = frateh5.create_dataset('z', data=YY)

for i in range(1000+discard):
    rwpdet = []
    rwsamples = []
    for samples_Mz, samples_z, samples_pdet  in zip(sampleslists_Mz, sampleslists_z, sampleslists_pdet):
        samples = np.vstack((np.log10(samples_Mz), samples_z)).T
        if i <= 100+discard:
            rwsample_k, rwpdet_k = get_reweighted_sample(samples, samples_pdet, kde_object, bootstrap='poisson', prior_factor=prior_factor_function)
        else:
            rwsample_k, rwpdet_k =  median_bufferkdelist_reweighted_samples(samples, samples_pdet, np.log10(Mz_grid), z_grid, kdevalslist[-10:], bootstrap_choice='poisson', prior_factor=prior_factor_function)
        rwsamples.append(rwsample_k)
        rwpdet.append(rwpdet_k)
    rwsamples = np.concatenate(rwsamples)
    print(len(rwsamples), rwsamples.shape)
    rwpdet = np.concatenate(rwpdet) 
    weights = 1.0/rwpdet
    #kdeobject, kdevals = adaptive_weighted_kde(rwsamples, grid_pts, alpha=alp2D, bw=bw2D, weights=weights, returnKDE=True)
    kdevals, shiftedbw, shiftedalp, kde_object = get_opt_params_and_kde(rwsamples, grid_pts, bwgrid, alphagrid, cv_method='4-fold', k=4, weights=1.0/rwpdet, standardization=True)

    kdevals = kdevals.reshape(XX.shape)
    kdevalslist.append(kdevals)
    iterbwlist.append(shiftedbw)
    iteralplist.append(shiftedalp)
    frateh5.create_dataset('kde_iter{0:04}'.format(i), data=kdevals)     
    if i > 0 and i %100 == 0:
        u_plot.ThreePlots(XX, YY, kdevals,  TheoryMtot, Theory_z, logKDE=True,  iternumber=i, plot_name='average_')
        u_plot.histogram_datalist(iterbwlist[-100:], dataname='bw', pathplot='bwhist', Iternumber=i)
        u_plot.histogram_datalist(iteralplist[-100:], dataname='alpha', pathplot='alphit', Iternumber=i)

    print(i, "step done ")
frateh5.create_dataset('bandwidths', data=iterbwlist)
frateh5.create_dataset('alphas', data=iteralplist)
frateh5.close()
average_list =  np.percentile(kdevalslist[100:], 50, axis=0)
u_plot.ThreePlots(XX, YY, average_list,  TheoryMtot, Theory_z, iternumber=1001, plot_name='percentile_combined_all')
# Transpose the list of lists to get lists of corresponding elements
transposed_lists = list(map(list, zip(*kdevalslist[100:])))
# Calculate the average of each corresponding element
average_list = [sum(values) / len(values) for values in transposed_lists]
u_plot.ThreePlots(XX, YY, kdevals, TheoryMtot, Theory_z, logKDE=True,  iternumber=1000, plot_name='combined_all')
