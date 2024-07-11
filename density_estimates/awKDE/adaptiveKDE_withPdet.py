import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import h5py as h5
import scipy
from scipy.interpolate import RegularGridInterpolator


import utils_awkde as u_awkde
import utils_data as u_data
import utils_plot as u_plot


#careful parsers 
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--datafilename',  default='pop3_200randomPEsamples_output_file.hdf5',help='h5 file containing N samples for m1for all gw bbh event')
bwchoices= np.logspace(-1, 0, 10).tolist() 
parser.add_argument('--bw-grid', default= bwchoices, nargs='+', help='grid of choices of global bandwidth')
alphachoices = np.linspace(0.1, 1.0, 10).tolist()
parser.add_argument('--alpha-grid', nargs="+", default=alphachoices, type=float, help='grid of choices of sensitivity parameter alpha for local bandwidth')

# limits on KDE evulation: 
parser.add_argument('--reweightmethod', default='bufferkdevals', help='Only for gaussian sample shift method: we can reweight samples via buffered kdevals(bufferkdevals) or buffered kdeobjects (bufferkdeobject)', type=str)
parser.add_argument('--reweight-sample-option', default='reweight', help='choose either "noreweight" or "reweight" if reweight use fpop prob to get reweight sample (one sample for no bootstrap or no or multiple samples for poisson)', type=str)
parser.add_argument('--bootstrap-option', default='poisson', help='choose either "poisson" or "nopoisson" if None it will reweight based on fpop prob with single reweight sample for eaxh event', type=str)

#### buffer iteratio
parser.add_argument('--buffer-start', default=10, type=int, help='start of buffer in reweighting.')
parser.add_argument('--buffer-interval', default=100, type=int, help='interval of buffer that choose how many previous iteration resulkts we use in next iteration for reweighting.')

#plots and saving data
parser.add_argument('--pathplot', default='./', help='public_html path for plots', type=str)
parser.add_argument('--pathtag', default='re-weight-bootstrap_', help='public_html path for plots', type=str)
parser.add_argument('--output-filename', default='output_data_', help='write a proper name of output hdf files based on analysis', type=str)
opts = parser.parse_args()



###### Gaussian samples for each event and reweight them with fpop prob

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


###### reweighting 

def get_random_sample(original_samples, bootstrap='poisson'):
    """without reweighting"""
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
    fkde_samples = fpop_kde.predict(original_samples) #*1.0/vtvals
    pdetvals[pdetvals < 1e-3] = 0
    fkde_samples = division_a_by_b(fkde_samples, pdetvals) #here we divide by weights
    frate_atsample = fkde_samples * prior_factor(original_samples) 
    fpop_at_samples = frate_atsample/frate_atsample.sum() # normalize
    rng = np.random.default_rng()
    if bootstrap =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1), p=fpop_at_samples)
    else:
        reweighted_sample = rng.choice(original_samples, p=fpop_at_samples)

    return reweighted_sample


def median_bufferkdelist_reweighted_samples(sample, m1val, m2val , kdelist, bootstrap_choice='poisson', prior_factor=prior_factor_function):
    """
    added a prior factor to handle non uniform prior factor
    for ln parameter kde or rate
    inputs
    and based on what choice of bootstrap is given
    """
    interpkdeval_list = []
    for kde in kdelist:
        interp = RegularGridInterpolator((m1val, m2val), kde, bounds_error=False, fill_value=0.0)
        kde_interp_vals = interp(sample)
        kde_interp_vals  *= prior_factor(sample)

        interpkdeval_list.append(kde_interp_vals)
    mediankdevals = np.percentile(interpkdeval_list, 50, axis=0)
    norm_mediankdevals = mediankdevals/sum(mediankdevals)
    rng = np.random.default_rng()
    if bootstrap_choice =='poisson':
        reweighted_sample = rng.choice(sample, np.random.poisson(1), p=norm_mediankdevals)
    else:
        reweighted_sample = rng.choice(sample, p=norm_mediankdevals)
    return reweighted_sample
###############################
from scipy.integrate import simpson
def OneD_integrals(Mgrid, Zgrid, Rategrid, title='KDE', iterN=0, saveplot=False):
    """
    integrating Rate for Mz, z
    """
    #kde_Mz = simpson(y=Rategrid, x=Zgrid, axis=0)
    #kde_Z = simpson(y=Rategrid, x=Mgrid, axis=1)
    kde_Z = simpson(y=Rategrid, x=Mgrid, axis=0)
    kde_Mz = simpson(y=Rategrid, x=Zgrid, axis=1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(np.log10(Mz), normdata(kde_Mz), label='obs')
    axs[0].set_ylabel('p(Log10[Mz])')
    axs[1].set_xlabel('Log10[Mz]')
    axs[0].set_title('Integrated Results')

    axs[1].plot(z, normdata(kde_Z), label='obs')
    axs[1].set_ylabel('p(z)')
    axs[1].set_xlabel('z')
    axs[1].legend()
    axs[1].set_title('Integrated Results')
    plt.tight_layout()
    if saveplot ==True:
        plt.savefig("integrated_plot"+title+"iter_{0}.png".format(iterN))
    plt.close()
    return 0

def normdata(dataV):
    normalized_data = (dataV - np.min(dataV)) / (np.max(dataV) - np.min(dataV))
    return normalized_data


def division_a_by_b(a,b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

#######################################################################
#We either have mock data h5 files(or we generate it) or  gw pe-samples
f1 = h5.File(opts.datafilename, 'r')
grpMz = f1['data_Mz']
grp_z = f1['data_z']
sampleslists_Mz = []
sampleslists_z = []
medianlist_Mz = []
medianlist_z = []
eventlist = f1['event/event_list'][...]
for k in grpMz.keys():
    Mz_values = grpMz[k][...]
    z_values = grp_z[k][...]
    sampleslists_Mz.append(Mz_values)
    sampleslists_z.append(z_values)
    medianlist_Mz.append(np.percentile(Mz_values, 50)) 
    medianlist_z.append(np.percentile(z_values, 50)) 
f1.close()


#flat_Mz_all = np.concatenate(sampleslists_Mz) 
#flat_z_all = np.concatenate(sampleslists_z) 
#np.savetxt("use200RsamplePerevent.txt", np.c_[flat_Mz_all, flat_z_all])
meanxi1 = np.array(medianlist_Mz)
meanxi2 = np.array(medianlist_z)

#evaluating grids
Mz = np.logspace(3, 8, 200)
z = np.logspace(0, np.log10(20), 200)
Log10Mmesh, zmesh = np.meshgrid(np.log10(Mz), z,  indexing='ij')



plt.figure()
for sampleMz, sample_z in zip(sampleslists_Mz, sampleslists_z):
    plt.scatter(np.log10(sampleMz), sample_z, marker='+')
plt.xlabel('Log10[Mz]')
plt.ylabel('z')
plt.title("all events 200 random samples")
plt.savefig('Scattered_200randomPE_event_data.png')
plt.show()
#quit()

sampleslists = np.vstack((np.array(sampleslists_Mz), np.array(sampleslists_z))).T
sample = np.vstack((np.log10(meanxi1), meanxi2)).T

###########Pdet corrected for Gaussian Only ##########
from scipy.interpolate import RegularGridInterpolator
data = np.loadtxt("Data_uniform_pdet_Mz_z_optPdet_MaxoptSNR_MFpdet.txt").T
M_values = np.unique(data[0])
z_values = np.unique(data[1])#.unique()
interp = RegularGridInterpolator((M_values, z_values), data[-1].reshape(len(M_values), len(z_values)), fill_value=0.0, bounds_error=False)
pdet_Mz_z_lists = interp((10**(Log10Mmesh), zmesh))

# Plotting Pdet
pdet_Mz_z_lists[pdet_Mz_z_lists <=1e-2] = 0.0

u_plot.new2DKDE(Log10Mmesh, zmesh, pdet_Mz_z_lists, title='Pdet')
xy_grid_pts = np.array(list(map(np.ravel, [Log10Mmesh, zmesh]))).T

#### Default alpha and bw parameters for KDE
alphagrid = opts.alpha_grid
bwgrid = opts.bw_grid 

##First median samples KDE
current_kde, errorkdeval, errorbBW, erroraALP = u_awkde.get_Ndimkde(sample, xy_grid_pts, alphagrid, bwgrid, ret_kde=True)
# reshape KDE to XX grid shape 
print("pdet shape, kde shape =", pdet_Mz_z_lists.shape, errorkdeval.shape)
ZZ = errorkdeval.reshape(Log10Mmesh.shape)

u_plot.new2DKDE(Log10Mmesh, zmesh, ZZ, title='medianKDE', iterN=0, saveplot=True)

########## ASTROPHYSICAL KDE/RATE
data = np.loadtxt("Plank15dataPopIII.dat").T
Theory_z = data[1]
TheoryMtot = data[0]
LogMTh = np.log10(TheoryMtot)
z_eval = np.logspace(0, np.log10(20), 200)
sampleTh = np.vstack((LogMTh, Theory_z)).T
intKDE = u_awkde.N_dim_KDE_awkde(sampleTh, xy_grid_pts, alp=0.5, gl_bandwidth=0.1)
intZZ = intKDE.reshape(Log10Mmesh.shape)
u_plot.new2DKDE(Log10Mmesh, zmesh, intZZ, title='intrinsicKDE', saveplot=True)
OneD_integrals(Log10Mmesh, zmesh, intZZ, title='intrinsicKDE', iterN=0, saveplot=True)


rate = division_a_by_b(ZZ, pdet_Mz_z_lists)
u_plot.new2DKDE(Log10Mmesh, zmesh, rate, title='rate')
plt.show()

discard = 10#opts.buffer_start   
Nbuffer = 100#opts.buffer_interval #100 buffer [how many (x-numbers of ) previous iteration to use in reweighting with average of those past (x-numbers of ) iteration results

OneD_integrals(Log10Mmesh, zmesh, ZZ)
OneD_integrals(Log10Mmesh, zmesh, rate)
#quit()

iter2Drate_list = []
iterkde_list = []
iterbwlist = []
iteralplist = []
kdeobjectlist = []
#rate_list = [] # for1Drate

#### We want to save data for rate(m1, m2) in HDF file 
frateh5 = h5.File('saving_iterativecases.hdf5', 'w')
dsetxx = frateh5.create_dataset('data_ln10Mz', data=Log10Mmesh)
dsetxx.attrs['xname']='ln10_Mz'
dsetyy = frateh5.create_dataset('data_z', data=zmesh)
dsetxx.attrs['yname']='z'
#

for i in range(1000+discard):
    print("i - ", i)
    rwsamples = []
    for sampleMz, sample_z in zip(sampleslists_Mz, sampleslists_z):
        samples= np.vstack((np.log10(sampleMz), sample_z)).T
        if i < discard + Nbuffer :
            pdet_Mz_z_interp = interp((sampleMz, sample_z))
            rwsample= get_reweighted_sample(samples, pdet_Mz_z_interp, current_kde, bootstrap=opts.bootstrap_option)
        else: 
            rwsample= median_bufferkdelist_reweighted_samples(samples, np.log10(Mz) , z, iter2Drate_list[-100:], bootstrap_choice=opts.bootstrap_option)
        rwsamples.append(rwsample)
    if opts.bootstrap_option =='poisson':
        rwsamples = np.concatenate(rwsamples)
    print("iter", i, "  totalsamples = ", len(rwsamples))

    current_kde, current_kdeval, shiftedbw, shiftedalp = u_awkde.get_Ndimkde(np.array(rwsamples), xy_grid_pts, alphagrid, bwgrid, ret_kde=True)
    current_kdeval = current_kdeval.reshape(Log10Mmesh.shape)
    frateh5.create_dataset('kde_iter{0:04}'.format(i), data=current_kdeval)

    current_rateval = division_a_by_b(current_kdeval, pdet_Mz_z_lists)
    frateh5.create_dataset('rate_iter{0:04}'.format(i), data=current_rateval)
        #rate1D =len(rwsamples)*u_rate.dr_of_m1(p1grid, min(p2grid), current_kde, newcorrectVTval)
    if i%50==0: 
        u_plot.new2DKDE(Log10Mmesh, zmesh, current_kdeval, title='kde', iterN=i, saveplot=True)
        u_plot.new2DKDE(Log10Mmesh, zmesh, current_rateval, title='rate', iterN=i, saveplot=True)
        OneD_integrals(Log10Mmesh, zmesh, current_rateval, title='Rate', iterN=i, saveplot=True)
    iterbwlist.append(shiftedbw)
    iteralplist.append(shiftedalp)
    if i < discard:
        continue
    iterkde_list.append(current_kdeval)
    iter2Drate_list.append(current_rateval)
    kdeobjectlist.append(current_kde)
    #rate_list.append(rate1D)

    #if i%100==discard-1:
    if i > discard and i%Nbuffer==0:
        iterstep = int(i-(discard))
        print(iterstep)
        
        u_plot.histogram_datalist(iterbwlist[-100:], dataname='bw', pathplot=opts.pathplot, Iternumber=iterstep)
        u_plot.histogram_datalist(iteralplist[-100:], dataname='alpha', pathplot=opts.pathplot, Iternumber=iterstep)
        u_plot.new2DKDE(Log10Mmesh, zmesh, np.percentile(iter2Drate_list[-100:], 50, axis=0), title='Average_rate', iterN=i, saveplot=True)
        OneD_integrals(Log10Mmesh, zmesh, np.percentile(iter2Drate_list[-100:], 50, axis=0), title='ave_Rate', iterN=i, saveplot=True)

frateh5.create_dataset('bandwidths', data=iterbwlist)
frateh5.create_dataset('alphas', data=iteralplist)
frateh5.close()


#alpha bw plots
u_plot.bandwidth_correlation(iterbwlist, number_corr=discard, error=0.02,  pathplot=opts.pathplot)
u_plot.bandwidth_correlation(iteralplist, number_corr=discard,  error=0.02, param='alpha',pathplot=opts.pathplot)
u_plot.bandwidth_correlation(iterbwlist, number_corr=discard, error=0.0, pathplot=opts.pathplot)
u_plot.bandwidth_correlation(iteralplist, number_corr=discard, error=0.0, param='alpha', pathplot=opts.pathplot)

