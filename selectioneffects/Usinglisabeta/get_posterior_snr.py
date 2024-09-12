# use lisabeta to get optimal SNRs
############Imports
import sys
import numpy as np
import matplotlib.pyplot as plt
#fixed Style
from matplotlib import rcParams
rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=14
rcParams["ytick.labelsize"]=14
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=16
rcParams["axes.labelsize"]=16
rcParams["axes.grid"] = True
rcParams["grid.color"] = 'grey'
rcParams["grid.linewidth"] = 1.
rcParams["grid.linestyle"] = ':'
rcParams["grid.alpha"] = 0.8
import os
import h5py as h5
import itertools
import copy
import json
from astropy.cosmology import Planck15 as cosmo
from tqdm import tqdm as tqdm
import lisabeta
import lisabeta.pyconstants as pyconstants
import lisabeta.tools.pytools as pytools
import lisabeta.tools.pyspline as pyspline
import lisabeta.tools.pyoverlap as pyoverlap
import lisabeta.lisa.pyresponse as pyresponse
import lisabeta.lisa.snrtools as snrtools
import lisabeta.lvk.pyLVKresponse as pyLVKresponse
import lisabeta.lisa.pyLISAnoise as pyLISAnoise
import lisabeta.lisa.lisatools as lisatools
import lisabeta.lisa.lisa as lisa
import lisabeta.utils.plotutils as plotutils

import astropy
from astropy import cosmology, units
from astropy.cosmology import  z_at_value, Planck15

def get_zarray(DLarray_Mpc):
    zvals = np.zeros_like(DLarray_Mpc)
    for it in range(len(zvals)):
        zvals[it] = z_at_value(Planck15.luminosity_distance,  float(DLarray_Mpc[it])*units.Mpc)
    print("done")
    return zvals


#specify these from PE files
def get_SNR_at_posterior(m1z, m2z, chi1, chi2, Deltat, dist, inc, phi, lamBda, beta, psi, waveform_params):
    params_base = {
    "m1": m1z, # m1*(1+z)
    "m2": m2z, #m2*(1+z),
    "chi1": chi1,
    "chi2": chi2,
    "Deltat": Deltat,
    "dist": dist,
    "inc": inc,
    "phi": phi,
    "lambda": lamBda,
    "beta": beta,
    "psi": psi,
    "Lframe": True
    }
    # Base waveform params
    tdisignal = lisa.GenerateLISATDISignal_SMBH(params_base, **waveform_params)
    #print(tdisignal['waveform_params/modes'])
    return tdisignal['SNR']

with open('ftag58', 'r')as fileall:
    ftags = [line.strip() for line in fileall.readlines()] 
#Need h5, jons files  with correct path to them
for ftag in ftags:
    fnameh5 = 'catalog_1_yrs.txt.'+ftag+'.h5'
    fnamejson = 'catalog_1_yrs.txt.'+ftag+'.json'
    #print(os.path.basename(fnameh5)) 
    filename =  str(os.path.splitext(os.path.basename(fnameh5))[0])
    import json
    # Path to your JSON file
    file_path = 'json_catalog_1_yrs.txt.'+ftag+'.json'
    print(file_path)
    # Load the JSON data from the file
    with open(file_path, 'r') as file:
        datatrue = json.load(file)
    params = datatrue['params']
    trueD = params['dist']
    trueSNR = datatrue['SNR']
    print("True SNR = ", trueSNR) 
    with open(fnamejson, 'r') as filej:
        data = json.load(filej)
    waveform_params = data['waveform_params']
    #print(type(waveform_params), waveform_params)
    
    with h5.File(fnameh5, 'r') as fh5:
        qv = fh5['q'][...]
        Mv = fh5['M'][...]
        DLv = fh5['dist'][...]
        maxDL = np.max(DLv)
        #print(z_at_value(Planck15.luminosity_distance,  float(maxDL)*units.Mpc))
        lnlik = fh5['lnlike']
        print("min likelihood", np.min(lnlik))
        qv = fh5['q'][...]
        chi1v = fh5['chi1'][...]
        chi2v = fh5['chi2'][...]
        Deltatv = fh5['Deltat'][...]
        betav =  fh5['beta'][...]
        incv =  fh5['inc'][...]
        psiv =  fh5['psi'][...]
        phiv =  fh5['phi'][...] 
        lambdav = fh5['lambda'][...]
        Mz_vals = Mv
        DL_vals =  DLv
        #z_vals = get_zarray(DL_vals)
        q_vals = qv
    
        Xi1_vals = chi1v
        Xi2_vals = chi2v
        Deltat_vals = Deltatv
        inc_vals = incv
        psi_vals = psiv
        phi_vals = phiv
        beta_vals = betav
        lambda_vals = lambdav
    
        m1_vals = q_vals*Mz_vals / (1. + q_vals)
        m2_vals = Mz_vals / (1. + q_vals)
    fh5.close()
     
    
    snr_arr = []

    #print("totalsamples = ", len(Mz_vals))
    for k in range(len(Mz_vals)):
        #print("sample number = ", k)
        snr = get_SNR_at_posterior(m1_vals[k], m2_vals[k], Xi1_vals[k], Xi2_vals[k], Deltat_vals[k], DL_vals[k], inc_vals[k], phi_vals[k], lambda_vals[k], beta_vals[k], psi_vals[k], waveform_params)
        snr_arr.append(snr)
        #print(f'snr = {snr}')
    
    np.savetxt("data_posteriors_DL_SNR_vals_{0}.txt".format(filename), np.c_[DL_vals, snr_arr])
    plt.figure(figsize=(12, 8))
    plt.title("event = "+ftag)
    plt.hist(snr_arr, bins=50, histtype='step', label='posterior-SNR')
    plt.axvline(x=trueSNR, color='red', linewidth=3, label='True-SNR')
    plt.xlabel("SNR")
    plt.semilogy()
    plt.legend()
    plt.savefig("plot_SNR_hist_{0}.png".format(filename))
    plt.close()

