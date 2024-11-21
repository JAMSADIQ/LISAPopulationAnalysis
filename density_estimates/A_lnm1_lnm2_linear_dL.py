import sys
import matplotlib
sys.path.append('/home/jam.sadiq/PopModels/selectioneffects/cbc_pdet/pop-de/popde/')
import density_estimate as d
import adaptive_kde as ad
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import h5py as h5
import scipy
from scipy.interpolate import RegularGridInterpolator
from matplotlib import rcParams
from matplotlib.colors import LogNorm, Normalize
import matplotlib.colors as colors

rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=18
rcParams["ytick.labelsize"]=18
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=18
rcParams["axes.labelsize"]=18
rcParams["axes.grid"] = True
rcParams["grid.color"] = 'grey'
rcParams["grid.linewidth"] = 1.
#rcParams["grid.linestyle"] = ':'
rcParams["grid.alpha"] = 0.6


# we will need data files and utils file so add a path to them
#sys.path.append('../../../')
import utils_awkde as u_awkde
import utils_plot as u_plot
import alternative_utils_plot as u_plot2
import o123_class_found_inj_general as u_pdet
#careful parsers 
parser = argparse.ArgumentParser(description=__doc__)
# use code get_gwtc_data_samples.py in bin directory
parser.add_argument('--datafilename1', help='h5 file containing N samples for m1for all gw bbh event')
parser.add_argument('--datafilename2', help='h5  file containing N sample of parameter2 for each event, ')
parser.add_argument('--datafilename3', help='h5  file containing N sample of redshift for each event, ')
parser.add_argument('--parameter1', help='name of parameter which we use for x-axis for KDE', default='m1')
parser.add_argument('--parameter2', help='name of parameter which we use for y-axis for KDE [can be m2, Xieff, DL]', default='m2')
parser.add_argument('--parameter3', help='name of parameter which we use for y-axis for KDE [can be m2, Xieff, DL]', default='DL')
### For KDE in log parameter we need to add --logkde 
parser.add_argument('--logkde', action='store_true',help='if True make KDE in log params but results will be in onlog')
bwchoices= np.logspace(-1.5, -0.1, 15).tolist() 
parser.add_argument('--bw-grid', default= bwchoices, nargs='+', help='grid of choices of global bandwidth')
alphachoices = np.linspace(0.1, 1.0, 10).tolist()
parser.add_argument('--alpha-grid', nargs="+", default=alphachoices, type=float, help='grid of choices of sensitivity parameter alpha for local bandwidth')

# limits on KDE evulation: 
parser.add_argument('--m1-min', help='minimum value for primary mass m1', type=float)
parser.add_argument('--m1-max', help='maximum value for primary mass m1', type=float)
parser.add_argument('--Npoints', default=100, type=int, help='Total points on which to evaluate KDE')
#m2-min must be <= m1-min
parser.add_argument('--param2-min', default=2.95, type=float, help='minimum value of m2 ,chieff =-1, DL= 1Mpc, used, must be below m1-min')
parser.add_argument('--param2-max', default=100.0, type=float, help='max value of m2 used, could be m1-max for chieff +1, for DL 10000Mpc')
parser.add_argument('--param3-min', default=10, type=float, help='minimum value of m2 ,chieff =-1, DL= 1Mpc, used, must be below m1-min')
parser.add_argument('--param3-max', default=10000.0, type=float, help='max value of m2 used, could be m1-max for chieff +1, for DL 10000Mpc')

# analysis on mock data or gw data.
parser.add_argument('--type-data', choices=['gw_pe_samples', 'mock_data'], help='mock data for some power law with gaussian peak or gwtc  pe samples data. h5 files for two containing data for median and sigma for m1')
#only for  mock data we will need this 
parser.add_argument('--power-alpha', default=0.0, help='power in power law sample (other than gaussian samples) in true distribution', type=float)
parser.add_argument('--fp-gauss', default=0.6, help='fraction of gaussian sample in true distribution', type=float)
parser.add_argument('--fpopchoice', default='kde', help='choice of fpop to be rate or kde', type=str)
parser.add_argument('--mockdataerror', default='fixed', help='mockdata error = 5 if fixed otherwise use np.random.randint(minval, maxval)', type=str)

#EMalgorithm reweighting 
parser.add_argument('--reweightmethod', default='bufferkdevals', help='Only for gaussian sample shift method: we can reweight samples via buffered kdevals(bufferkdevals) or buffered kdeobjects (bufferkdeobject)', type=str)
parser.add_argument('--reweight-sample-option', default='reweight', help='choose either "noreweight" or "reweight" if reweight use fpop prob to get reweight sample (one sample for no bootstrap or no or multiple samples for poisson)', type=str)
parser.add_argument('--bootstrap-option', default='poisson', help='choose either "poisson" or "nopoisson" if None it will reweight based on fpop prob with single reweight sample for eaxh event', type=str)

#### buffer iteratio
parser.add_argument('--buffer-start', default=500, type=int, help='start of buffer in reweighting.')
parser.add_argument('--buffer-interval', default=100, type=int, help='interval of buffer that choose how many previous iteration resulkts we use in next iteration for reweighting.')
parser.add_argument('--NIterations', default=1000, type=int, help='Total Iterations in reweighting')

#plots and saving data
parser.add_argument('--pathplot', default='./', help='public_html path for plots', type=str)
parser.add_argument('--pathtag', default='re-weight-bootstrap_', help='public_html path for plots', type=str)
parser.add_argument('--output-filename', default='MassRedshift_with_reweight_output_data_', help='write a proper name of output hdf files based on analysis', type=str)
opts = parser.parse_args()



#############for ln paramereter we need these
def prior_factor_function(samples):
    """ 
    LVC use uniform-priors for masses 
    in linear scale. so
    reweighting need a constant factor

   note that in the reweighting function 
   if we use input masses/dL in log
   form so when we need to factor
   we need non-log mass/dL  so we take exp
    if we use non-cosmo_files we need 
    dL^3 factor 
    """
    m1val, m2val , dLval = samples[:, 0], samples[:, 1], samples[:, 2]
    if opts.logkde:
        factor = 1.0/(dLval)**2  # log-masses handle in the input_transf
    else:
        factor = np.ones_like(m1val)
    return factor


###### reweighting 

def get_random_sample(original_samples, bootstrap='poisson'):
    """without reweighting"""
    rng = np.random.default_rng()
    if bootstrap =='poisson':
        reweighted_sample = rng.choice(original_samples, np.random.poisson(1))
    else:
        reweighted_sample = rng.choice(original_samples)
    return reweighted_sample

def apply_min_cap_function(pdet_list):
  """Applies the min(10, 1/pdet) function to each element in the given list.

  Args:
    pdet_list: A list of values.

  Returns:
    A new list containing the results of applying the function to each element.
  """

  result = []
  for pdet in pdet_list:
    #result.append(min(10, 1 / pdet))
    result.append(max(0.03, pdet))
  return np.array(result)

def get_reweighted_sample(original_samples, vtvals, fpop_kde, bootstrap='poisson', prior_factor=prior_factor_function):
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
    fkde_samples = fpop_kde.evaluate_with_transf(original_samples) / apply_min_cap_function(vtvals)

    if opts.logkde:
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


def median_bufferkdelist_reweighted_samples(sample, vtvals, interp, bootstrap_choice='poisson', prior_factor=prior_factor_function):
    """
    added a prior factor to handle non uniform prior factor
    for ln parameter kde or rate
    inputs
    and based on what choice of bootstrap is given
    """
    #Take the medians of kde and use it in interpolator
    #median_kde_values = np.percentile(kdelist, 50, axis=0)
    #print("shape of median for interpolator", median_kde_values.shape)
    #interp = RegularGridInterpolator((m1val, m2val, dLval), median_kde_values.T, bounds_error=False, fill_value=0.0)
    kde_interp_vals = interp(sample)/apply_min_cap_function(vtvals)
    if opts.logkde:
        kde_interp_vals  *= prior_factor(sample)
    norm_mediankdevals = kde_interp_vals/sum(kde_interp_vals)
    rng = np.random.default_rng()
    if bootstrap_choice =='poisson':
        reweighted_sample = rng.choice(sample, np.random.poisson(1), p=norm_mediankdevals)
    else:
        reweighted_sample = rng.choice(sample, p=norm_mediankdevals)
    return reweighted_sample


#######################################################################
injection_file = "endo3_bbhpop-LIGO-T2100113-v12.hdf5"
with h5.File(injection_file, 'r') as f:
    T_obs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
    N_draw = f.attrs['total_generated']

    m1 = f['injections/mass1_source'][:]
    m2 = f['injections/mass2_source'][:]
    s1x = f['injections/spin1x'][:]
    s1y = f['injections/spin1y'][:]
    s1z = f['injections/spin1z'][:]
    s2x = f['injections/spin2x'][:]
    s2y = f['injections/spin2y'][:]
    s2z = f['injections/spin2z'][:]
    z = f['injections/redshift'][:]
    dLp = f["injections/distance"][:]
    m1_det = m1#*(1.0 +  z)
    p_draw = f['injections/sampling_pdf'][:]
    pastro_pycbc_bbh = f['injections/pastro_pycbc_bbh'][:]


# Calculate min and max for dLp
min_dLp, max_dLp = min(dLp), max(dLp)

#####################################
#here we need pdet
run_fit = 'o3'
run_dataset = 'o3'
dmid_fun = 'Dmid_mchirp_fdmid_fspin' #'Dmid_mchirp_fdmid'
emax_fun = 'emax_exp'
alpha_vary = None
g = u_pdet.Found_injections(dmid_fun = 'Dmid_mchirp_fdmid_fspin', emax_fun='emax_exp', alpha_vary = None, ini_files = None, thr_far = 1, thr_snr = 10)

#If we want to compute pdet as we dont have pdet file alreadt use below line and comment the next after it  "w"(if want new pdet file)  versus "r"(if have file)
#fpdet = h5.File('checkbasedNnoncosmo_GWTC3_pdet_datafile.h5', 'w')
fpdet = h5.File('checkbasedNnoncosmo_GWTC3_pdet_datafile.h5', 'r')
fz = h5.File('Final_noncosmo_GWTC3_redshift_datafile.h5', 'r')
dz = fz['randdata']
f1 = h5.File('Final_noncosmo_GWTC3_m1srcdatafile.h5', 'r')#m1
d1 = f1['randdata']
f2 = h5.File('Final_noncosmo_GWTC3_m2srcdatafile.h5', 'r')#m2
d2 = f2['randdata']
f3 = h5.File('Final_noncosmo_GWTC3_dL_datafile.h5', 'r')#dL
d3 = f3['randdata']
print(d1.keys())
sampleslists1 = []
medianlist1 = []
eventlist = []
sampleslists2 = []
medianlist2 = []
sampleslists3 = []
medianlist3 = []
pdetlists = []
for k in d1.keys():
    eventlist.append(k)
    if (k  == 'GW190719_215514_mixed-nocosmo' or k == 'GW190805_211137_mixed-nocosmo'):
        print(k)
        m1_values = d1[k][...]#*(1.0 + dz[k][...])
        m2_values = d2[k][...]#*(1.0 + dz[k][...])
        d_Lvalues = d3[k][...]
        pdet_values = fpdet[k][...]
        #find unreasonable dL vals in PE samples
        dL_indices = [i for i, dL in enumerate(d_Lvalues) if (dL < min_dLp  or dL > max_dLp)]
        m1_values = [m for i, m in enumerate(m1_values) if i not in  dL_indices]
        m2_values = [m for i, m in enumerate(m2_values) if i not in  dL_indices]
        d_Lvalues = [dL for i, dL in enumerate(d_Lvalues) if i not in dL_indices]
        #pdet_values = np.zeros(len(d_Lvalues))
        #for i in range(len(d_Lvalues)):
        #    pdet_values[i] = u_pdet.get_pdet_m1m2dL(d_Lvalues[i], m1_values[i], m2_values[i], classcall=g)      
        
        #fpdet.create_dataset(k, data=pdet_values)
        #pdet_values = [pdet for i, pdet in enumerate(pdet_values) if i not in dL_indices]
        #still some bad indices
        pdetminIndex = np.where(np.array(pdet_values) < 0.0001)[0]
        m1_values = np.delete(m1_values, pdetminIndex).tolist()
        m2_values = np.delete(m2_values, pdetminIndex).tolist()
        d_Lvalues = np.delete(d_Lvalues, pdetminIndex).tolist()
        pdet_values = np.delete(pdet_values, pdetminIndex).tolist()

    else:
        m1_values = d1[k][...]#*(1.0 + dz1[k][...])
        m2_values = d2[k][...]#*(1.0 + dz1[k][...])
        d_Lvalues = d3[k][...]
        # if we want to compute pdet use line after below line 
        pdet_values = fpdet[k][...]
        #pdet_values =  np.zeros(len(d_Lvalues))
        #for i in range(len(d_Lvalues)):
        #    print(i)
        #   pdet_values[i] = u_pdet.get_pdet_m1m2dL(d_Lvalues[i], m1_values[i], m2_values[i], classcall=g)
        #fpdet.create_dataset(k, data=pdet_values)
    pdetlists.append(pdet_values)
    sampleslists1.append(m1_values)
    sampleslists2.append(m2_values)
    sampleslists3.append(d_Lvalues)
    medianlist1.append(np.percentile(m1_values, 50)) 
    medianlist2.append(np.percentile(m2_values, 50)) 
    medianlist3.append(np.percentile(d_Lvalues, 50))

f1.close()
f2.close()
f3.close()
fz.close()
fpdet.close()

meanxi1 = np.array(medianlist1)
meanxi2 = np.array(medianlist2)
meanxi3 = np.array(medianlist3)
flat_samples1 = np.concatenate(sampleslists1).flatten()
flat_samples2 = np.concatenate(sampleslists2).flatten()
flat_samples3 = np.concatenate(sampleslists3).flatten()
flat_pdetlist = np.concatenate(pdetlists).flatten()
print("min max m1 =", np.min(flat_samples1), np.max(flat_samples1))
print("min max m2 =", np.min(flat_samples2), np.max(flat_samples2))
print("min max dL =", np.min(flat_samples3), np.max(flat_samples3))

############### We need plotting  work in progress
# Create the scatter plot for pdet 
from mpl_toolkits.mplot3d import Axes3D
# 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(flat_samples1, flat_samples2, flat_samples3, c=flat_pdetlist, cmap='viridis', s=10)
plt.colorbar(sc, label=r'$p_\mathrm{det}(m_1, m_2, d_L)$')
ax.set_xlabel(r'$m_{1, source} [M_\odot]$', fontsize=20)
ax.set_ylabel(r'$m_{2, source} [M_\odot]$', fontsize=20)
ax.set_xlim(min(flat_samples1), max(flat_samples1))
ax.set_ylim(min(flat_samples2), max(flat_samples2))
ax.set_zlim(min(flat_samples3), max(flat_samples3))
ax.set_zlabel(r'$d_L [Mpc]$', fontsize=20)
plt.tight_layout()
plt.savefig(opts.pathplot+"pdet3Dscatter.png")
plt.close()

##########################################
sampleslists = np.vstack((flat_samples1, flat_samples2, flat_samples3)).T
sample = np.vstack((meanxi1, meanxi2, meanxi3)).T
print(sampleslists.shape)
#to make KDE (2DKDE)  # we use same limits on m1 and m2 
if opts.m1_min is not None and opts.m1_max is not None:
    xmin, xmax = opts.m1_min, opts.m1_max
else:
    xmin, xmax = np.min(flat_samples1), np.max(flat_samples1)
#if  flat_samples1 is list of arrays
#xmin = min(a.min() for a in flat_samples1)
#xmax = max(a.max() for a in flat_samples1)

if opts.param2_min is not None and opts.param2_max is not None:
    ymin, ymax = opts.param2_min, opts.param2_max
else:
    ymin, ymax = np.min(flat_samples2) , np.max(flat_samples2)

if opts.param3_min is not None and opts.param3_max is not None:
    zmin, zmax = opts.param3_min, opts.param3_max
else:
    zmin, zmax = np.min(flat_samples3) , np.max(flat_samples3)

xmin, xmax = 3, 100 #np.min(flat_samples1), np.max(flat_samples1)
ymin, ymax = 3, 100  #np.min(flat_samples2) , np.max(flat_samples2)
zmin, zmax = 10, 5000 #np.min(flat_samples3) , np.max(flat_samples3)
Npoints = opts.Npoints
#######################################################
################ We will be using masses in log scale so better to use
##### If we want to use log param we need proper grid spacing in log scale
p1grid = np.logspace(np.log10(xmin), np.log10(xmax), Npoints) 
p2grid = np.logspace(np.log10(ymin), np.log10(ymax), Npoints) 
p3grid = np.logspace(np.log10(zmin), np.log10(zmax), 30)#Npoints) 
#mesh grid points 
XX, YY, ZZ = np.meshgrid(p1grid, p2grid, p3grid,  indexing='ij')
#input_transf['log', 'log', None] will automatically do it
xy_grid_pts = np.array(list(map(np.ravel, [XX, YY, ZZ]))).T
print(xy_grid_pts.shape)
#since we only need log on masses but not dL
sample = np.vstack((meanxi1, meanxi2, meanxi3)).T
################################################################################
def get_kde_obj_eval(sample, eval_pts, rescale_arr, alphachoice, input_transf=('log', 'log', 'none'), mass_symmetry=False):
    #Apply m1-m2 symmetry in the samples before fitting
    if mass_symmetry:
        m1 = sample[:, 0]  # First column corresponds to m1
        m2 = sample[:, 1]  # Second column corresponds to m2
        dL = sample[:, 2]  # Third column corresponds to dL
        sample2 = np.vstack((m2, m1, dL)).T
        #Combine both samples into one array
        symsample = np.vstack((sample, sample2))
        kde_object = ad.KDERescaleOptimization(symsample, stdize=True, rescale=rescale_arr, alpha=alphachoice, dim_names=['lnm1', 'lnm2', 'dL'], input_transf=input_transf)
    else:
        kde_object = ad.KDERescaleOptimization(sample, stdize=True, rescale=rescale_arr, alpha=alphachoice, dim_names=['lnm1', 'lnm2', 'dL'], input_transf=input_transf)
    dictopt, score = kde_object.optimize_rescale_parameters(rescale_arr, alphachoice, bounds=((0.01,10),(0.01, 10),(0.01, 10) ,(0,1)), disp=True)#, xatol=1e-5, fatol=0.01)
    kde_vals = kde_object.evaluate_with_transf(eval_pts)
    optbwds = 1.0/dictopt[0:-1]
    print(optbwds)
    optalpha = dictopt[-1]
    print("opt results = ", dictopt)
    return  kde_object, kde_vals, optbwds, optalpha


##First median samples KDE
init_rescale_arr = [1., 1., 1.]
init_alpha_choice = [0.5]
current_kde, errorkdeval, errorbBW, erroraALP = get_kde_obj_eval(sample, xy_grid_pts, init_rescale_arr, init_alpha_choice, mass_symmetry=True)
bwx, bwy, bwz = errorbBW[0], errorbBW[1], errorbBW[2]
print(errorbBW)
# reshape KDE to XX grid shape 
pdfKDE = errorkdeval.reshape(XX.shape)
# Choose the index to slice along X3
#dL_index = np.searchsorted(p3grid,  500)#500Mpc
#dL_index_val = p3grid[dL_index]


def get_sliced_data(xx, yy, kde3D, rate3D, dLgrid, dL_sliceval=500):
    dL_index = np.searchsorted(dLgrid,  dL_sliceval)#500Mpc
    dL_index_val = dLgrid[dL_index]
    KDE_slice = kde3D[:, :, dL_index]  # Sliced values of F at the chosen x3
    Rate_slice = rate3D[:, :, dL_index]  # Sliced values of F at the chosen x3
    M1_slice, M2_slice = xx[:, :, dL_index], yy[:, :, dL_index]  
    return M1_slice, M2_slice, KDE_slice, Rate_slice

########## We for the moment just make symmetry KDE plotas for m1-m2 analysis
#for this we need slicing of data at fixed dL 
if opts.logkde:
    if opts.parameter3=='dL':
        print("dL case we need to use prior factor but in reweighting steps not here?")
        nlZZ = pdfKDE/ZZ**2
    else:
        nlZZ = pdfKDE # for m1-m2 case

#########need to fix the plots
import pickle
#print(XX.shape)
#if opts.fpopchoice == 'rate':
PDET = np.zeros((Npoints, Npoints, len(p3grid)))
PDET = u_pdet.get_pdet_m1m2dL(XX, YY, ZZ, classcall=g) 
    ## Set all values in `pdet` less than 0.1 to 0.1
    #PDET = np.maximum(PDET, 0.1)
###  save the to use next time
with open('In3Dm1m2dLwith_max003LongGridwithPEMdetpickle_testpdet_ij.data', 'wb') as f:
    pickle.dump(PDET, f)
    #print(np.min(PDET))
PDET = np.maximum(PDET, 0.03)
#######use the saved data
    #with open('with_max01LongGridwithPEMdetpickle_testpdet_ij.data', 'rb') as f:
    #    PDET  = pickle.load(f)
current_rateval = len(meanxi1)*pdfKDE/PDET 
#Rate_slice = current_rateval[:, :, dL_index]  # Sliced values of F at the chosn x3
M1_slice, M2_slice, KDE_slice, Rate_slice = get_sliced_data(XX, YY, pdfKDE, current_rateval, p3grid, dL_sliceval=500)
u_plot2.twoDKDEplot(meanxi1, meanxi2, M1_slice, M2_slice,  KDE_slice, pathplot=opts.pathplot, x_label='m1', y_label='m2',  fixeddLval=500)
u_plot2.twoDRateplot(meanxi1, meanxi2, M1_slice, M2_slice, Rate_slice, pathplot=opts.pathplot, x_label='m1', y_label='m2')
print("kdeshape", pdfKDE.shape)
print("pdetshape", PDET.shape)
print("slice_rate_shape", Rate_slice.shape)
klists = []
rlists =[]
for i in range(10):
    klists.append(KDE_slice)
    rlists.append(Rate_slice)

#test average KDE and Rate plot

u_plot2.average2Dkde_plot(meanxi1, meanxi2, M1_slice, M2_slice, klists, pathplot=opts.pathplot, titlename=0, plot_label='KDE', x_label='m1', y_label='m2', plottag='test')
u_plot2.average2Dkde_plot(meanxi1, meanxi2, M1_slice, M2_slice, rlists, pathplot=opts.pathplot, titlename=0, plot_label='Rate', x_label='m1', y_label='m2', plottag='test')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(XX.flatten(), YY.flatten(), ZZ.flatten(), c=PDET.flatten(), cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=1e-5))
## Add a colorbar
fig.colorbar(sc, shrink=0.5, aspect=5)
ax.set_xlabel(r'$m_{1, source} [M_\odot]$', fontsize=20)
ax.set_ylabel(r'$m_{2, source} [M_\odot]$', fontsize=20)
ax.set_zlabel(r'$d_L [Mpc]$', fontsize=20)
ax.set_title('3D Plot of pdet(m1, m2, dL)')
plt.savefig(opts.pathplot+"testpdet3Dm1m2dLpng")
plt.close()
### reweighting EM algorithm
Total_Iterations = int(opts.NIterations)
discard = int(opts.buffer_start)   # how many iterations to discard default =5
Nbuffer = int(opts.buffer_interval) #100 buffer [how many (x-numbers of ) previous iteration to use in reweighting with average of those past (x-numbers of ) iteration results

iter3Dkde_list = []
iter2Drate_list = []
iter2Dkde_list = []
iter3Drate_list = []
iterbwxlist = []
iterbwylist = []
iterbwzlist = []
iteralplist = []
#### We want to save data for rate(m1, m2) in HDF file 
frateh5 = h5.File(opts.output_filename+'withMsrc_discarded_100_uniform_priorfactor_dLsquare_2Dkde_rate_lnm1_lnm2_dL.hdf5', 'w')
dsetxx = frateh5.create_dataset('data_xx', data=XX)
dsetxx.attrs['xname']='xx'
dsetyy = frateh5.create_dataset('data_yy', data=YY)
dsetyy.attrs['yname']='yy'
dsetzz = frateh5.create_dataset('data_zz', data=ZZ)
dsetzz.attrs['zname']='zz'

for i in range(Total_Iterations + discard):
    print("i - ", i)
    if i >= discard + Nbuffer:
        #bufffer_kdes_median = np.percentile(iterkde_list[-Nbuffer:], 50, axis=0)
        #buffer_interp = RegularGridInterpolator((p1grid, p2grid), bufffer_kdes_median.T, bounds_error=False, fill_value=0.0)
        buffer_kdes_mean = np.mean(iter3Dkde_list[-Nbuffer:], axis=0)
        buffer_interp = RegularGridInterpolator((p1grid, p2grid, p3grid), buffer_kdes_mean, bounds_error=False, fill_value=0.0)
        #Take the medians of kde and use it in interpolator
    rwsamples = []
    for samplem1, samplem2, sample3, pdet_k in zip(sampleslists1, sampleslists2, sampleslists3,  pdetlists):
        samples= np.vstack((samplem1, samplem2, sample3)).T
        #symmtery will beapplied inside kde method
        if i < discard + Nbuffer :
            rwsample = get_reweighted_sample(samples, pdet_k, current_kde, bootstrap=opts.bootstrap_option)
        else: 
            rwsample= median_bufferkdelist_reweighted_samples(samples, pdet_k, buffer_interp, bootstrap_choice=opts.bootstrap_option)
        rwsamples.append(rwsample)
    if opts.bootstrap_option =='poisson':
        rwsamples = np.concatenate(rwsamples)
    print("iter", i, "  totalsamples = ", len(rwsamples))
    current_kde, current_kdeval, shiftedbw, shiftedalp = get_kde_obj_eval(np.array(rwsamples), xy_grid_pts, init_rescale_arr, init_alpha_choice, input_transf=('log', 'log', 'none'))
    bwx, bwy, bwz = shiftedbw[0], shiftedbw[1], shiftedbw[2]
    print("bwvalues", bwx, bwy, bwz)
    current_kdeval = current_kdeval.reshape(XX.shape)
    iter3Dkde_list.append(current_kdeval)
    iterbwxlist.append(bwx)
    iterbwylist.append(bwy)
    iterbwzlist.append(bwz)
    iteralplist.append(shiftedalp)
    frateh5.create_dataset('kde_iter{0:04}'.format(i), data=current_kdeval)
    if opts.fpopchoice == 'rate':
        current_rateval = len(rwsamples)*current_kdeval/PDET

        frateh5.create_dataset('rate_iter{0:04}'.format(i), data=current_rateval)
        iter3Drate_list.append(current_rateval)
        #get 2D results
        M1_slice, M2_slice, KDE_slice, Rate_slice = get_sliced_data(XX, YY, pdfKDE, current_rateval, p3grid, dL_sliceval=500)
        iter2Drate_list.append(Rate_slice)
        iter2Dkde_list.append(KDE_slice)
    
    #if i > discard and i%Nbuffer==0:
    if i > 1 and i%Nbuffer==0:
        iterstep = int(i)
        print(iterstep)
        u_plot.histogram_datalist(iterbwxlist[-Nbuffer:], dataname='bwx', pathplot=opts.pathplot, Iternumber=iterstep)
        u_plot.histogram_datalist(iterbwylist[-Nbuffer:], dataname='bwy', pathplot=opts.pathplot, Iternumber=iterstep)
        u_plot.histogram_datalist(iterbwzlist[-Nbuffer:], dataname='bwz', pathplot=opts.pathplot, Iternumber=iterstep)
        u_plot.histogram_datalist(iteralplist[-Nbuffer:], dataname='alpha', pathplot=opts.pathplot, Iternumber=iterstep)
        #######need to work on plots
        #if opts.logkde:
        u_plot2.average2Dkde_plot(meanxi1, meanxi2, M1_slice, M2_slice, iter2Dkde_list[-Nbuffer:], pathplot=opts.pathplot, titlename=i, plot_label='KDE', x_label='m1', y_label='m2')
        u_plot2.average2Dkde_plot(meanxi1, meanxi2, M1_slice, M2_slice, iter2Drate_list[-Nbuffer:], pathplot=opts.pathplot, titlename=i, plot_label='Rate', x_label='m1', y_label='m2')

        #     u_plot.average2DlineardLrate_plot(meanxi1, meanxi2, XX, YY, iterkde_list[-Nbuffer:], pathplot=opts.pathplot, titlename=i, plot_label='KDE', x_label='m1', y_label='dL', show_plot= False)
        #if opts.fpopchoice == 'rate':
        #     u_plot.average2DlineardLrate_plot(meanxi1, meanxi2, XX, YY, iter2Drate_list[-Nbuffer:], pathplot=opts.pathplot, titlename=i, plot_label='Rate', x_label='m1', y_label='dL', show_plot= False)
            #u_plot.average2Dkde_plot(meanxi1, meanxi2, XX, YY, iter2Drate_list[-Nbuffer:], pathplot=opts.pathplot, titlename=i, plot_label='Rate', x_label='m1', y_label='dL', show_plot= False)
frateh5.create_dataset('xbandwidths', data=iterbwxlist)
frateh5.create_dataset('ybandwidths', data=iterbwylist)
frateh5.create_dataset('zbandwidths', data=iterbwzlist)
frateh5.create_dataset('alphas', data=iteralplist)
frateh5.close()

u_plot2.average2Dkde_plot(meanxi1, meanxi2, M1_slice, M2_slice, iter2Dkde_list[discard:], pathplot=opts.pathplot, titlename=1001, plot_label='KDE', x_label='m1', y_label='m2', plottag='allKDEscombined_')
u_plot2.average2Dkde_plot(meanxi1, meanxi2, M1_slice, M2_slice, iter2Drate_list[discard:], pathplot=opts.pathplot, titlename=1001, plot_label='Rate', x_label='m1', y_label='m2', plottag='allRatescombined_')

#u_plot.average2DlineardLrate_plot(meanxi1, meanxi2, XX, YY, iterkde_list[discard:], pathplot=opts.pathplot+'allKDEscombined_', titlename=1001, plot_label='KDE', x_label='m1', y_label='dL', show_plot= False)
#u_plot.average2DlineardLrate_plot(meanxi1, meanxi2, XX, YY, iter2Drate_list[discard:], pathplot=opts.pathplot+'allKDEscombined_', titlename=1001, plot_label='Rate', x_label='m1', y_label='dL', show_plot= False)

#alpha bw plots
u_plot.bandwidth_correlation(iterbwxlist, number_corr=discard, error=0.02,  pathplot=opts.pathplot+'bwx_')
u_plot.bandwidth_correlation(iterbwylist, number_corr=discard, error=0.02,  pathplot=opts.pathplot+'bwy_')
u_plot.bandwidth_correlation(iterbwzlist, number_corr=discard, error=0.02,  pathplot=opts.pathplot+'bwz_')
u_plot.bandwidth_correlation(iteralplist, number_corr=discard,  error=0.02, param='alpha',pathplot=opts.pathplot, log=False)
#u_plot.bandwidth_correlation(iterbwxlist, number_corr=discard, error=0.0, pathplot=opts.pathplot+'bwx')
#u_plot.bandwidth_correlation(iterbwylist, number_corr=discard, error=0.0, pathplot=opts.pathplot+'bwy')
#u_plot.bandwidth_correlation(iterbwzlist, number_corr=discard, error=0.0, pathplot=opts.pathplot+'bwz')
#u_plot.bandwidth_correlation(iteralplist, number_corr=discard, error=0.0, param='alpha', pathplot=opts.pathplot, log=False)

