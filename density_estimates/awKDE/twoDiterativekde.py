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
bwchoices= np.logspace(-1.5, -0.5, 10).tolist() 
parser.add_argument('--bw-grid', default= bwchoices, nargs='+', help='grid of choices of global bandwidth')
alphachoices = [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]#np.linspace(0., 1.0, 11).tolist()
parser.add_argument('--alpha-grid', nargs="+", default=alphachoices, type=float, help='grid of choices of sensitivity parameter alpha for local bandwidth')
parser.add_argument('--crossvalidationmethod', default='loo_cv', type=str, help='leave one out cv or k-fold for best choice of alpha and bw default is loo_cv for k fold check number of folds')

# limits on KDE evulation: 
parser.add_argument('--reweightmethod', default='bufferkdevals', help='Only for gaussian sample shift method: we can reweight samples via buffered kdevals(bufferkdevals) or buffered kdeobjects (bufferkdeobject)', type=str)
parser.add_argument('--reweight-sample-option', default='reweight', help='choose either "noreweight" or "reweight" if reweight use fpop prob to get reweight sample (one sample for no bootstrap or no or multiple samples for poisson)', type=str)
parser.add_argument('--bootstrap-option', default='poisson', help='choose either "poisson" or "nopoisson" if None it will reweight based on fpop prob with single reweight sample for eaxh event', type=str)

parser.add_argument('--logparam-prior', default=True, help='Prior factor in reweighting')
parser.add_argument('--useprior', default=False, help='if we want to use non uniform prior factor effect. need some chenges in code')

parser.add_argument('--buffer-start', default=100, type=int, help='start of buffer in reweighting.')
parser.add_argument('--buffer-interval', default=100, type=int, help='interval of buffer that choose how many previous iteration resulkts we use in next iteration for reweighting.')
parser.add_argument('--total-iterations', default=1000, type=int, help='number of  iteration in iterative reweighting.')

#plots and saving data
parser.add_argument('--methodtag', default='withpdetandpriorfactor', help='mention if we are using pdet factor and prior in reweighting', type=str)
parser.add_argument('--pathplot', default='./', help='public_html path for plots', type=str)
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
    """
     Generates a random sample from the given original samples 
     using the specified bootstrap method.

    Args:
        original_samples (numpy.ndarray): The original samples from which to draw the random sample.
        bootstrap (str, optional): The bootstrap method to use. Can be 'poisson' or 'regular'. Defaults to 'poisson'.

    Returns:
        numpy.ndarray: A[/few/none for Poisson] random sample drawn from the original samples

    """
    rng = np.random.default_rng()
    if bootstrap =='poisson':
        # Sample with replacement using Poisson distribution
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1))
    else:
        # Sample with replacement using a uniform distribution
        reweighted_sample = rng.choice(original_samples)
    return reweighted_sample


def get_reweighted_sample(original_samples, pdetvals, fpop_kde, bootstrap='poisson', prior_factor=prior_factor_function, use_prior=False):
    """
    Generates a reweighted sample from the given original samples 
    based on the provided probability density estimation (PDE) 
    and optional prior factor.

    Args:
        original_samples (numpy.ndarray): The original samples from
           which to draw the reweighted sample.
        pdetvals (numpy.ndarray): pdet (selection effects) values 
           corresponding to the original samples.
        fpop_kde (GaussianKDE): The KDE object from previous iteration 
           use for reweight samples.
        bootstrap (str, optional): The bootstrap method to use.
           Can be 'poisson' or 'regular'. Defaults to 'poisson'.
        prior_factor (function, optional): A function that calculates 
           the prior factor for reweighting. Defaults to `prior_factor_function`.
        use_prior (bool, optional): Whether to use the prior factor for 
           reweighting. Defaults to False.

    Returns:
        numpy.ndarray: A reweighted sample drawn from the original samples.
    """
    #Issue can occus oif pdet <<1 1e-6 or small maybe 0
    fkde_samples = fpop_kde.predict(original_samples)
    fkde_with_pdet = fkde_samples * 1.0/pdetvals

    if use_prior == True:
        fpop_atsample = fkde_with_pdet * prior_factor(original_samples) 
    else:
        fpop_atsample = fkde_with_pdet

    fpop_norm = fpop_atsample/fpop_atsample.sum() # normalize

    rng = np.random.default_rng()
    if bootstrap =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1), p=fpop_norm)
    else:
        reweighted_sample = rng.choice(original_samples, p=fpop_norm)

    return reweighted_sample


def median_bufferkdelist_reweighted_samples(original_samples, original_pdet, Log10_Mz_val, z_val, kdelist, bootstrap_choice='poisson', prior_factor=prior_factor_function, use_prior=False):
    """
    Generates a reweighted sample using the median KDE from a list of previous N iterations.
    Note that medain with 50th percentile need transpose 
                np.percentile(kdelist, 50, axis=0).T
    Args:
        original_samples (numpy.ndarray): The original samples to reweight.
        original_pdet (numpy.ndarray): The original pdet values (selection effects).
        Log10_M_val (numpy.ndarray): The log10 M values used for the KDEs.
        z_val (numpy.ndarray): The z values used for the KDEs.
        kdelist (list): A list of previous KDE estimates for previous N iterations
        bootstrap_choice (str, optional): The bootstrap method to use ('poisson' or 'regular'). 
           Defaults to 'poisson'.
        prior_factor (function, optional): A function to calculate the prior factor. 
           Defaults to `prior_factor_function`.
        use_prior (bool, optional): Whether to use the prior factor for reweighting. 
            Defaults to False.

    Returns:
        numpy.ndarray: A reweighted sample.
    """
    # get median and than make an interpolator #must take transpose of median output otherwise incorrect
    median_kde_values = np.percentile(kdelist, 50, axis=0)
    interp = RegularGridInterpolator((Log10_Mz_val, z_val), median_kde_values.T, bounds_error=False, fill_value=0.0)
    kde_interp_vals = interp(original_samples)
    fkde_with_pdet  =  kde_interp_vals * (1.0/original_pdet)

    if use_prior == True:
        fpop_atsample = fkde_with_pdet * prior_factor(original_samples)
    else:
        fpop_atsample = fkde_with_pdet

    fpop_norm = fpop_atsample/fpop_atsample.sum() # normalize

    rng = np.random.default_rng()
    if bootstrap_choice =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1), p=fpop_norm)
    else:
        reweighted_sample = rng.choice(original_samples, p=fpop_norm)
    return reweighted_sample


def normdata(dataV):
    normalized_data = (dataV - np.min(dataV)) / (np.max(dataV) - np.min(dataV))
    return normalized_data

###################Specific HDF data #########
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
    # remove samples  with z >20
    #indices = np.argwhere(data_z <= 20).flatten()
    # remove samples with pdet < 1e-4
    #indices = np.argwhere(data_pdet >= 1e-3).flatten()
    data_z = data_z#[indices]
    dataM = data_M#[indices]
    datapdet = data_pdet#[indices]
    plt.scatter(dataM, data_z, marker='+')
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
plt.savefig(opts.pathplot+opts.methodtag+"colored_samples_perevent_year92.png")
plt.show()
hdf_file.close()


median_arr_M = np.array(medianlist_M)
median_arr_z = np.array(medianlist_z)
median_arr_pdet = np.array(medianlist_pdet)
flat_M_all = np.concatenate(sampleslists_M).flatten()#np.array(sampleslists_M).flatten()
flat_z_all = np.concatenate(sampleslists_z).flatten()
flat_pdet_all = np.concatenate(sampleslists_pdet).flatten()


#################Range of values for KDE evaluation ################
######### 1:  Range of at which to evluate KDE can be set in parser Use logpace points
M_grid = np.logspace(2., 9., 200)  #100-10^9
z_grid = np.logspace(-1, np.log10(20), 200) #0.1 to 20

XX, YY = np.meshgrid(np.log10(M_grid), z_grid)
nonlnXX, YY = np.meshgrid(M_grid, z_grid) #For plotting

grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T
median_samples = np.vstack((np.log10(median_arr_M) ,median_arr_z)).T
all_samples = np.vstack((np.log10(flat_M_all), flat_z_all)).T
print("total samples", len(flat_pdet_all), len(flat_z_all), all_samples.shape)

##### KDE parameters
alphagrid = opts.alpha_grid # using 0-1 with 0.1 spacing 11 choices
bwgrid = opts.bw_grid      # 15 choices from 0.001 to 0.5

##First median samples KDE use loo_cv for alpha/bw
current_kde, errorkdeval, errorbBW, erroraALP = u_awkde.kde_twoD_with_do_optimize(median_samples, grid_pts, bwgrid, alphagrid, ret_kde=True, optimize_method='loo_cv')
ZZ = errorkdeval.reshape(XX.shape)
u_plot.new2DKDE(nonlnXX, YY,  ZZ, median_arr_M, median_arr_z, iterN=0, saveplot=True, title='medianPEKDE', show_plot=True, pathplot=opts.pathplot)

u_plot.ThreePlots(XX, YY, ZZ, ZZ, ZZ, nonlnXX, TheoryMtot, Theory_z, iternumber=0, plot_name='medianPEKDE', make_errorbars=False, show_plot=True, pathplot=opts.pathplot)
############## All data KDE ##############################################################################
ZZall = u_awkde.kde_awkde(all_samples, grid_pts, global_bandwidth=errorbBW, alpha=erroraALP, ret_kde=False)
ZZall = ZZall.reshape(XX.shape)

u_plot.new2DKDE(nonlnXX, YY,  ZZall, flat_M_all, flat_z_all, iterN=0, saveplot=True, title='KDEallsamples', show_plot=True, pathplot=opts.pathplot)
u_plot.ThreePlots(XX, YY, ZZall, ZZall, ZZall, nonlnXX, TheoryMtot, Theory_z, iternumber=0, plot_name='allPEsamplesKDE', make_errorbars=False, show_plot=True, pathplot=opts.pathplot)
#quit()
########## ASTROPHYSICAL catalog DATA
#data = np.loadtxt("/home/jsadiq/Research/E_awKDE/CatalogLISA/lensedPopIII/combined_intrinsicdata100years_Mz_z_withPlanck_cosmology.dat").T
data = np.loadtxt("/home/jsadiq/Research/E_awKDE/CatalogLISA/lensedPopIII/Corrected_for_timeofObsrandom_detected_events_given_intrinsic_events_with_4year_observations_for_lensed_data_Mz_z_based_on_optSNR_with_MC_on_extrinsic_params_with_threshold_SNR8.dat").T
Theory_z = data[1]
TheoryMtot = data[0]
#We want source frame mass
TheoryMtot /= (1.0 +  Theory_z)

u_plot.ThreePlots(XX, YY, ZZall, ZZall, ZZall, nonlnXX, TheoryMtot, Theory_z, plot_name='AllSamples', pathplot=opts.pathplot)

#### Iterative weighted-KDE
iterbwlist = []
iteralplist = []
iterbwlist = []
iteralplist = []
frateh5 = h5.File(opts.output_filename+'awkde_method_'+opts.methodtag+'Data2DIterativeCase.hdf5', 'w')
dsetxx = frateh5.create_dataset('M', data=nonlnXX)
dsetyy = frateh5.create_dataset('z', data=YY)

discard = int(opts.buffer_start)   # how many iterations to discard default =100
Nbuffer = int(opts.buffer_interval) #100 buffer [how many (x-numbers of ) previous iteration to use in reweighting with average of those past (x-numbers of ) iteration results
kdevalslist = []
TotalIterations = int(opts.total_iterations)#1000

print("optimization of alpha and bw with ", opts.crossvalidationmethod)
for i in range(TotalIterations+discard):
    print("i - ", i)
    rwsamples = []
    for sampleM, sample_z,  samples_pdet in zip(sampleslists_M, sampleslists_z, sampleslists_pdet):
        samples= np.vstack((np.log10(sampleM), sample_z)).T
        if i < discard + Nbuffer :
            rwsample= get_reweighted_sample(samples, samples_pdet, current_kde, bootstrap=opts.bootstrap_option, use_prior=opts.useprior)
        else: 
            rwsample= median_bufferkdelist_reweighted_samples(samples, samples_pdet, np.log10(M_grid), z_grid, kdevalslist[-Nbuffer:], bootstrap_choice=opts.bootstrap_option, use_prior=opts.useprior)
        rwsamples.append(rwsample)
    rwsamples = np.concatenate(rwsamples)
    print("iter", i, "  totalsamples = ", len(rwsamples))
    print("optimization of alpha and bw with ")
    current_kde, current_kdeval, shiftedbw, shiftedalp =  u_awkde.kde_twoD_with_do_optimize(rwsamples, grid_pts, bwgrid, alphagrid, ret_kde=True, optimize_method=opts.crossvalidationmethod)
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
        u_plot.ThreePlots(XX, YY, medKDE, KDE95th, KDE5th, nonlnXX, TheoryMtot, Theory_z ,iternumber=i, plot_name='AvergeKDE2D', make_errorbars=True, show_plot=True, pathplot = opts.pathplot)
        u_plot.histogram_datalist(iterbwlist[-Nbuffer:], dataname='bw', pathplot=opts.pathplot, Iternumber=i)
        u_plot.histogram_datalist(iteralplist[-Nbuffer:], dataname='alpha', pathplot=opts.pathplot, Iternumber=i)
    print(i, "step done ")
frateh5.create_dataset('bandwidths', data=iterbwlist)
frateh5.create_dataset('alphas', data=iteralplist)
frateh5.close()
# Calculate the average of all KDE after discard
average_list = np.percentile(kdevalslist[discard:], 50, axis=0) 
pc5th_list = np.percentile(kdevalslist[discard:], 5, axis=0) 
pc95th_list = np.percentile(kdevalslist[discard:], 95, axis=0) 
u_plot.ThreePlots(XX, YY, average_list, pc95th_list, pc5th_list, nonlnXX, TheoryMtot, Theory_z, iternumber=1001, plot_name='combined1000iterations', make_errorbars=True, show_plot=True, pathplot=opts.pathplot)


#alpha bw plots
u_plot.bandwidth_correlation(iterbwlist, number_corr=discard, error=0.02,  pathplot=opts.pathplot)
u_plot.bandwidth_correlation(iteralplist, number_corr=discard,  error=0.02, param='alpha',pathplot=opts.pathplot)
u_plot.bandwidth_correlation(iterbwlist, number_corr=discard, error=0.0, pathplot=opts.pathplot)
u_plot.bandwidth_correlation(iteralplist, number_corr=discard, error=0.0, param='alpha', pathplot=opts.pathplot)

