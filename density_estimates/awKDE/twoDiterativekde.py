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
from matplotlib import rcParams
rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--datafilename',  default='/home/jsadiq/Research/E_awKDE/CatalogLISA/lensedPopIII/json-params/samples/new_save_pdet_with_time_to_merger_randomize/SourceMasswithcorrection_time_SNRthreshold8.0combine_4years_lensed_events.hdf',help='h5 file containing N samples for m1for all gw bbh event')

### For KDE in log parameter we need to add --logkde 
parser.add_argument('--logkde', action='store_true',help='if True make KDE in log params but results will be in onlog')
bwchoices= np.logspace(-1.5, -0.5, 10).tolist() 

parser.add_argument('--bw-grid', default= bwchoices, nargs='+', help='grid of choices of global bandwidth')
alphachoices = [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]#np.linspace(0., 1.0, 11).tolist()
parser.add_argument('--alpha-grid', nargs="+", default=alphachoices, type=float, help='grid of choices of sensitivity parameter alpha for local bandwidth')

# limits on KDE evulation: 
parser.add_argument('--reweightmethod', default='bufferkdevals', help='Only for gaussian sample shift method: we can reweight samples via buffered kdevals(bufferkdevals) or buffered kdeobjects (bufferkdeobject)', type=str)
parser.add_argument('--reweight-sample-option', default='reweight', help='choose either "noreweight" or "reweight" if reweight use fpop prob to get reweight sample (one sample for no bootstrap or no or multiple samples for poisson)', type=str)
parser.add_argument('--bootstrap-option', default='poisson', help='choose either "poisson" or "nopoisson" if None it will reweight based on fpop prob with single reweight sample for eaxh event', type=str)
parser.add_argument('--logparam-prior', default=True, help='Prior factor in reweighting')
parser.add_argument('--buffer-start', default=100, type=int, help='start of buffer in reweighting.')
parser.add_argument('--buffer-interval', default=100, type=int, help='interval of buffer that choose how many previous iteration resulkts we use in next iteration for reweighting.')
parser.add_argument('--total-iterations', default=1000, type=int, help='number of  iteration in iterative reweighting.')

#plots and saving data
parser.add_argument('--pathplot', default='./', help='public_html path for plots', type=str)
parser.add_argument('--pathtag', default='re-weight-bootstrap_', help='public_html path for plots', type=str)
parser.add_argument('--output-filename', default='output_data_', help='write a proper name of output hdf files based on analysis', type=str)
opts = parser.parse_args()


#############for ln paramereter we need these
def prior_factor_function(samples, logkde=opts.logparam_prior):
    """
    Calculates the prior factor for reweighting samples in the context of
    LVC's uniform-prior assumption for masses in linear scale.

    Args:
        samples (numpy.ndarray): A 2D NumPy array containing the samples, where
            the first column represents the total masses (M) and the second column
            represents the redshifts (z).
        logkde (bool, optional): If True, indicates that the input masses are
            already in log10 form. Defaults to True.

    Returns:
        numpy.ndarray: A 1D NumPy array containing the prior factor for each
        sample.
    """
    M_vals, z_vals = samples[:, 0], samples[:, 1]
    if  logkde:
        #If the input masses are log10(Mz), calculate the factor accordingly
        factor = 1.0/10**(M_vals)
    else:
        factor = np.ones_like(M_vals)
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


def get_reweighted_sample(original_samples, pdetvals, fpop_kde, bootstrap='poisson', prior_factor=prior_factor_function, use_prior=False):
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
    if use_prior == True:
        frate_atsample = fkde_samples * prior_factor(original_samples) 
    else:
        frate_atsample = fkde_samples
    fpop_at_samples = frate_atsample/frate_atsample.sum() # normalize
    rng = np.random.default_rng()
    if bootstrap =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1), p=fpop_at_samples)
    else:
        reweighted_sample = rng.choice(original_samples, p=fpop_at_samples)

    return reweighted_sample


def median_bufferkdelist_reweighted_samples(original_samples, original_pdet, Log10_Mz_val, z_val, kdelist, bootstrap_choice='poisson', prior_factor=prior_factor_function, use_prior=False):
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
    # get median and than make an interpolator
    median_kde_values = np.percentile(kdelist, 50, axis=0)
    #apply .T to check
    interp = RegularGridInterpolator((Log10_Mz_val, z_val), median_kde_values.T, bounds_error=False, fill_value=0.0)
    kde_interp_vals = interp(original_samples)*(1.0/original_pdet)#*prior_factor(original_samples)
    if use_prior == True:
        kde_interp_vals *= prior_factor(original_samples)
    norm_mediankdevals = kde_interp_vals/sum(kde_interp_vals)
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
sampleslists_M = []
sampleslists_z = []
sampleslists_pdet = []
medianlist_M = []
medianlist_z = []
medianlist_pdet = []
#print(hdf_file.keys())
plt.figure()
#print(len(hdf_file.keys()))
#quit()
for event_name in hdf_file.keys():
    data = hdf_file[event_name][...].T
    #print(data)
    data_M = data[0]
    data_z = data[1]
    data_pdet = data[2]
    # remove below if there is no issue with z >20
    indices = np.argwhere(data_z <= 20).flatten()
    #print(event_name, data_M[indices], data_z[indices])
    
    #indices = np.argwhere(data_pdet >= 1e-3).flatten()
    data_z = data_z[indices]
    dataM = data_M[indices]
    datapdet = data_pdet[indices]
    #if np.max(data_z > 23.0) and np.max(np.log10(data_M) > 3):
        #plt.figure()
    plt.scatter(dataM, data_z, marker='+')
        #plt.title(event_name)
        #plt.show()
    sampleslists_M.append(dataM)
    sampleslists_pdet.append(datapdet)
    sampleslists_z.append(data_z)
    medianlist_M.append(np.median(dataM))
    medianlist_z.append(np.median(data_z))
    medianlist_pdet.append(np.median(datapdet))

plt.xlabel(r"$M_\mathrm{source}\, [M_\odot]$", fontsize=20)
plt.ylabel(r"$\mathrm{redshift}$", fontsize=20)
plt.semilogx()
plt.grid()
plt.title("100 samples per event")
plt.tight_layout()
plt.savefig(opts.pathplot+"colored_samples_perevent_year92.png")
plt.show()
hdf_file.close()
median_arr_M = np.array(medianlist_M)
median_arr_z = np.array(medianlist_z)
median_arr_pdet = np.array(medianlist_pdet)
#sampleslists = np.vstack((np.array(sampleslists_M), np.array(sampleslists_z))).T

flat_M_all = np.concatenate(sampleslists_M).flatten()#np.array(sampleslists_M).flatten()
flat_z_all = np.concatenate(sampleslists_z).flatten()
flat_pdet_all = np.concatenate(sampleslists_pdet).flatten()

M_grid = np.logspace(2., 9., 200)
z_grid = np.logspace(-1, np.log10(20), 200)

XX, YY = np.meshgrid(np.log10(M_grid), z_grid)
nonlnXX, YY = np.meshgrid(M_grid, z_grid)
grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T
sample = np.vstack((np.log10(median_arr_M) ,median_arr_z)).T
all_samples = np.vstack((np.log10(flat_M_all), flat_z_all)).T
print("total samples", len(flat_pdet_all), len(flat_z_all), all_samples.shape)

############### Code is expensive we got opt vals on median and save them below
alphagrid = opts.alpha_grid #
bwgrid = opts.bw_grid
##First median samples KDE
#optimize_method default is loo_cv
current_kde, errorkdeval, errorbBW, erroraALP = u_awkde.kde_twoD_with_do_optimize(sample, grid_pts, bwgrid, alphagrid, ret_kde=True, optimize_method='kfold_cv')
ZZ = errorkdeval.reshape(XX.shape)
#priorfactor

u_plot.new2DKDE(nonlnXX, YY,  ZZ, median_arr_M, median_arr_z, iterN=0, saveplot=True, title='medianKDE')
############## All data KDE #############################
ZZall = u_awkde.kde_awkde(all_samples, grid_pts, global_bandwidth=errorbBW, alpha=erroraALP, ret_kde=False)
ZZall = ZZall.reshape(XX.shape)

u_plot.new2DKDE(nonlnXX, YY,  ZZall, flat_M_all, flat_z_all, iterN=1, saveplot=True, title='KDEallsamples')

#quit()
########## ASTROPHYSICAL catalog DATA
#data = np.loadtxt("/home/jsadiq/Research/E_awKDE/CatalogLISA/lensedPopIII/combined_intrinsicdata100years_Mz_z_withPlanck_cosmology.dat").T
data = np.loadtxt("/home/jsadiq/Research/E_awKDE/CatalogLISA/lensedPopIII/Corrected_for_timeofObsrandom_detected_events_given_intrinsic_events_with_4year_observations_for_lensed_data_Mz_z_based_on_optSNR_with_MC_on_extrinsic_params_with_threshold_SNR8.dat").T
Theory_z = data[1]
TheoryMtot = data[0]
#We want source frame mass
TheoryMtot /= (1.0 +  Theory_z)
u_plot.ThreePlots(XX, YY, ZZall, ZZall, ZZall, nonlnXX, TheoryMtot, Theory_z, plot_name='AllSamples')
#quit()

#### Iterative weighted-KDE
iterbwlist = []
iteralplist = []
iterbwlist = []
iteralplist = []

frateh5 = h5.File(opts.output_filename+'awkde_method_Data2DIterativeCase.hdf5', 'w')
dsetxx = frateh5.create_dataset('M', data=nonlnXX)
dsetyy = frateh5.create_dataset('z', data=YY)

discard = int(opts.buffer_start)   # how many iterations to discard default =5
Nbuffer = int(opts.buffer_interval) #100 buffer [how many (x-numbers of ) previous iteration to use in reweighting with average of those past (x-numbers of ) iteration results
kdevalslist = []
TotalIterations = 100

for i in range(TotalIterations+discard):
    print("i - ", i)
    rwsamples = []
    for sampleM, sample_z,  samples_pdet in zip(sampleslists_M, sampleslists_z, sampleslists_pdet):
        samples= np.vstack((np.log10(sampleM), sample_z)).T
        if i < discard + Nbuffer :
            rwsample= get_reweighted_sample(samples, samples_pdet, current_kde, bootstrap=opts.bootstrap_option)
        else: 
            rwsample= median_bufferkdelist_reweighted_samples(samples, samples_pdet, np.log10(M_grid), z_grid, kdevalslist[-Nbuffer:], bootstrap_choice=opts.bootstrap_option)
        rwsamples.append(rwsample)
    rwsamples = np.concatenate(rwsamples)
    print("iter", i, "  totalsamples = ", len(rwsamples))
    current_kde, current_kdeval, shiftedbw, shiftedalp =  u_awkde.kde_twoD_with_do_optimize(rwsamples, grid_pts, bwgrid, alphagrid, ret_kde=True, optimize_method='kfold_cv')
    #u_awkde.get_Ndimkde(np.array(rwsamples), grid_pts, alphagrid, bwgrid, ret_kde=True)
    current_kdeval = current_kdeval.reshape(XX.shape)
    kdevalslist.append(current_kdeval)
    iterbwlist.append(shiftedbw)
    iteralplist.append(shiftedalp)
    frateh5.create_dataset('kde_iter{0:04}'.format(i), data=current_kdeval)
    if  i > 0.0 and i %Nbuffer == 0:
        medKDE = np.percentile(kdevalslist[-Nbuffer:], 50, axis=0)
        KDE5th = np.percentile(kdevalslist[-Nbuffer:], 5, axis=0)
        KDE95th = np.percentile(kdevalslist[-Nbuffer:], 95, axis=0)
        u_plot.ThreePlots(XX, YY, medKDE, KDE95th, KDE5th, nonlnXX, TheoryMtot, Theory_z ,iternumber=i, plot_name='AvergeKDE2D')
        u_plot.histogram_datalist(iterbwlist[-Nbuffer:], dataname='bw', pathplot='./', Iternumber=i)
        u_plot.histogram_datalist(iteralplist[-Nbuffer:], dataname='alpha', pathplot='./', Iternumber=i)
    print(i, "step done ")
frateh5.create_dataset('bandwidths', data=iterbwlist)
frateh5.create_dataset('alphas', data=iteralplist)

frateh5.close()
# Calculate the average of all KDE after discard
average_list = np.percentile(kdevalslist[discard:], 50, axis=0) 
pc5th_list = np.percentile(kdevalslist[discard:], 5, axis=0) 
pc95th_list = np.percentile(kdevalslist[discard:], 95, axis=0) 

u_plot.ThreePlots(XX, YY, average_list, pc95th_list, pc5th_list, nonlnXX, TheoryMtot, Theory_z, iternumber=1001, plot_name='Awcombined_all')


#alpha bw plots
u_plot.bandwidth_correlation(iterbwlist, number_corr=discard, error=0.02,  pathplot=opts.pathplot)
u_plot.bandwidth_correlation(iteralplist, number_corr=discard,  error=0.02, param='alpha',pathplot=opts.pathplot)
u_plot.bandwidth_correlation(iterbwlist, number_corr=discard, error=0.0, pathplot=opts.pathplot)
u_plot.bandwidth_correlation(iteralplist, number_corr=discard, error=0.0, param='alpha', pathplot=opts.pathplot)

