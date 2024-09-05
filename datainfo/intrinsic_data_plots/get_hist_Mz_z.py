import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.lines import Line2D
import matplotlib.colors

rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=15
rcParams["ytick.labelsize"]=15
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=16
rcParams["axes.labelsize"]=18
rcParams["axes.grid"] = True
rcParams["grid.color"] = 'black'
#rcParams["grid.linewidth"] = 2.
rcParams["grid.alpha"] = 0.4

#data_title= 'intrinsic'
data_title= 'detected'
d = np.loadtxt("/home/jsadiq/Research/E_awKDE/CatalogLISA/lensedPopIII/combined_intrinsicdata100years_Mz_z_withPlanck_cosmology.dat").T
if data_title == 'detected':
    #d = np.loadtxt("detected_from_intrinsic_data_Mz_z_SNR_MC_extrinsic_angles_with_threshold_SNR8.dat").T
    d = np.loadtxt("/home/jsadiq/Research/E_awKDE/CatalogLISA/lensedPopIII/Corrected_for_timeofObsrandom_detected_events_given_intrinsic_events_with_4year_observations_for_lensed_data_Mz_z_based_on_optSNR_with_MC_on_extrinsic_params_with_threshold_SNR8.dat").T
    #d = np.loadtxt("SourceMasscorrected_tobs_events_given_intrinsic_events_with_4year_observations_for_lensed_data_Msource_z_based_on_optSNR_with_MC_on_extrinsic_params.dat").T
Mz, redshift = d[0], d[1]
Mz = Mz/(1.0 +  redshift) # source frame mass
#### fixing z issue  in data
log1pz = np.log(redshift + 1)
sc = log1pz + np.random.normal(0, 0.043, size=log1pz.shape) 
redshift = np.exp(sc) - 1.0
Nobs = int(len(Mz)/100.) #observation per year by averaging
Nevents = len(Mz) 
weights_per_year = np.ones(len(Mz))*0.01#/100 or multiply by 0.01
############## Making KDE and fix with normalization to get rates
from sklearn.neighbors import KernelDensity
#lets use 0.5 a bw its just a rough estimate
kdeMz =  KernelDensity(bandwidth=0.5, kernel='gaussian')
kdeMz.fit(np.log(Mz)[:, np.newaxis])
eval_LogMz = np.log(np.logspace(2.0, 9, 200)) #on log values
# Evaluate the KDEs
log_density_Mz = kdeMz.score_samples(eval_LogMz[:, np.newaxis])
KDE_Mz = np.exp(log_density_Mz)*Nobs
########redshift
kde_z = KernelDensity(bandwidth=0.5, kernel='gaussian')
kde_z.fit(redshift[:, np.newaxis])
eval_z = np.linspace(np.min(redshift), 20., 200) #on log values
# Evaluate the KDEs
log_density_z = kde_z.score_samples(eval_z[:, np.newaxis])
KDE_z = np.exp(log_density_z)*Nobs


fig, ax = plt.subplots(figsize=(8,6))
ax.plot(np.exp(eval_LogMz), KDE_Mz)
ax.hist(Mz, bins=np.logspace(2, 9, 21), weights=weights_per_year, histtype='step', label=data_title)
# Set the axes to a logarithmic scale
ax.set_xscale('log')
ax.set_yscale('log')
# Set custom ticks for the x-axis
ax.set_xticks([100, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9],
           ['100', '1000', r'$10^4$', r'$10^5$', r'$10^6$', r'$10^7$', r'$10^8$', r'$10^9$'])
# Set custom ticks for the y-axis
ax.set_yticks([0.01, 0.1, 1, 10, 100], ['0.01', '0.1', '1', '10', '100'])
ax.set_xlim(90, 2e9)
#ax.set_ylim(ymin=0.1)
ax.set_xlabel(r"$M_z [M_\odot]$", fontsize=20)
ax.set_xlabel(r"$M_\mathrm{source} [M_\odot]$", fontsize=20)
ax.set_ylabel(r"$\frac{d^2N}{d\mathrm{log}(M_z) dt} [yr^{-1}]$", fontsize=22, verticalalignment='center', labelpad=18)
ax.set_ylabel(r"$\frac{d^2N}{d\mathrm{log}(M_{\mathrm{source}}) dt} [yr^{-1}]$", fontsize=22, verticalalignment='center', labelpad=18)
ax.legend(fontsize=20)
plt.tight_layout()
plt.savefig("Rate_histogram_4_yrs_snrbasedfiltered_intrinsic_data_Msrc_popIII.png")

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(eval_z, KDE_z, label='KDE Nobs yr')
ax.hist(redshift, bins=np.linspace(0, 20, 22), weights=weights_per_year, histtype='step', label=data_title)
ax.set_ylabel(r"$\frac{d^2N}{dz dt} [yr^{-1}]$", fontsize=24, verticalalignment='center', labelpad=20)
ax.set_xlabel(r"$\mathrm{redshift}$", fontsize=20)
ax.set_xticks([0, 5, 10, 15, 20], ['0', '5', '10', '15', '20'])
# Set custom ticks for the y-axis
#ax.set_yticks([0., 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], ['0', '5', '10', '15', '20','25', '30', '35', '40', '45', '50'])
ax.set_yticks([0., 5, 10, 15, 20, 25], ['0', '5', '10', '15', '20','25'])
ax.legend(fontsize=20)
ax.set_xlim(0, 20)
#ax.set_ylim(ymin=-0.1)
plt.savefig("Rate_histogram_4_yrs_snrbasedfiltered_intrinsic_data_z_popIII.png")
plt.tight_layout()
plt.show()


