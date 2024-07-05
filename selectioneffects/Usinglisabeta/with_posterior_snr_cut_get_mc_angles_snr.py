import os
import numpy as np
import lisabeta
import lisabeta.pyconstants as pyconstants
import lisabeta.tools.pytools as pytools
import lisabeta.lisa.pyLISAnoise as pyLISAnoise
import lisabeta.lisa.ldcnoise as ldcnoise
import lisabeta.lisa.lisa as lisa
import lisabeta.lisa.lisatools as lisatools
from astropy.cosmology import Planck15 as cosmo
import matplotlib.pyplot as plt

import scipy
from scipy.special import i0, erf
def Prob_matchfilter_SNR(optimal_SNR):
    """
    Calculate the pdet of match filter SNR given the optimal SNR.

    Parameters:
    optimal_SNR (float): The optimal signal-to-noise ratio (SNR).

    Formula  = 0.5 * (1.0 + erf((8. - optimal_SNR) / np.sqrt(2)))

    Notes:
    - The error function (erf) is imported from scipy.special module.
    - The SNR value is used to compute the error probability using the given formula.
    """
    # if list convert it to an array
    optimal_SNR = np.array(optimal_SNR)
    return 0.5 * (1.0 + erf((-8.0 + optimal_SNR) / np.sqrt(2)))



# Definition to randomize orientations
def draw_random_angles():
    phi = np.random.uniform(low=-np.pi, high=np.pi)
    inc = np.arccos(np.random.uniform(low=-1., high=1.))
    lambd = np.random.uniform(low=-np.pi, high=np.pi)
    beta = np.arcsin(np.random.uniform(low=-1., high=1.))
    psi = np.random.uniform(low=0., high=np.pi)
    return np.array([phi, inc, lambd, beta, psi])

import astropy
from astropy import cosmology
from astropy import units
from astropy.cosmology import  z_at_value, Planck15
 
def get_zarray(DLarray_Mpc):
    zvals = np.zeros_like(DLarray_Mpc)
    for it in range(len(zvals)):
        zvals[it] = z_at_value(Planck15.luminosity_distance,  float(DLarray_Mpc[it])*units.Mpc)
    print("done")
    return zvals

import json
import h5py as h5


def get_mc_angles_snr(filetag, Nsample=200, MC_iters=1000):
    """
    call json file for waveform, h5 file for intrinsic params
    compute 1000 snr at one pe sample
    """
    #posterior-SNR
    data_posterior = np.loadtxt('posterior_snr/data_posteriors_DL_SNR_vals_catalog_1_yrs.txt.'+filetag+'.txt').T
    posterior_snr = data_posterior[-1]
    #criteria to choose PE sample  pos-snr > 4.0
    indices = np.argwhere( posterior_snr > 4.0).flatten()
    with open('catalog_1_yrs.txt.'+filetag+'.json', 'r') as jsonfile:
        data_json = json.load(jsonfile)
        waveform_params = data_json['waveform_params']
    with h5.File('catalog_1_yrs.txt.'+filetag+'.h5', 'r') as fh5:
        #only for one event
        lnlik= fh5['lnlike'][...]
        qvall = fh5['q'][...][indices]
        print(len(qvall))
        
        Mv = fh5['M'][...][indices]
        DLv = fh5['dist'][...][indices]
        #zvals = get_zarray(DLv)
        lnlik = fh5['lnlike'][...][indices]
        chi1v = fh5['chi1'][...][indices]
        chi2v = fh5['chi2'][...][indices]
        Deltatv = fh5['Deltat'][...][indices]
        posterior_snrv = posterior_snr[indices] 
        random_indices = np.random.choice(np.arange(len(qvall)), int(Nsample))
        Mv = Mv[random_indices]
        #print(Mv)
        DLv = DLv[random_indices]
        lnlik = lnlik[random_indices]
        qv = qvall[random_indices]
        chi1v = chi1v[random_indices] 
        chi2v = chi2v[random_indices]
        Deltatv = Deltatv[random_indices] 
        posterior_snrv = posterior_snrv[random_indices] 
        m1v = qv*Mv / (1. + qv)
        m2v = Mv / (1. + qv)
        print("max-q, max-DL, min-lnlik = ", np.max(qv), np.max(DLv), np.min(lnlik))
    fh5.close()

    with h5.File('out_putcatalog_1_yrs.txt.'+filetag+'.h5', 'w') as fh5:
        param_and_val_list = []
    # Now we use MC over angles
        for k in range(len(Mv)):
            params_base = {
            "m1": m1v[k], # m1*(1+z)
            "m2": m2v[k], #m2*(1+z),
            "chi1": chi1v[k],
            "chi2": chi2v[k],
            "Deltat": Deltatv[k],
            "dist": DLv[k],
            "Lframe": True
            }
            snr_arr = []
            inc_arr = []
            for i in range(MC_iters):
                phi, inc, lambd, beta, psi = draw_random_angles()
                params_base['inc'] = inc
                params_base['phi'] = phi
                params_base['lambda'] = lambd
                params_base['beta'] = beta
                params_base['psi'] = psi
                tdisignal = lisa.GenerateLISATDISignal_SMBH(params_base, **waveform_params)
                snr_val = tdisignal['SNR']
                snr_arr.append(snr_val)
                inc_arr.append(inc)

            print(len(snr_arr))
            print("done index, max-minsnr", k,  np.max(snr_arr), np.min(snr_arr))
            #compute pdet
#            if np.max(snr_arr)> 50.0:
#                plt.plot(inc_arr, snr_arr, 'r+')
#                plt.xlabel("inc")
#                plt.ylabel("snr")
#                plt.show()
            pdet = np.sum(np.array(snr_arr) > 8.0)/MC_iters
            mf_pdet = sum(Prob_matchfilter_SNR(snr_arr))/len(snr_arr)
            if pdet == 0.0:
                print("indx, pdet, mfpdet, m1, m2, DL, lnlik, pos_snr = ", indices[k], pdet, mf_pdet, m1v[k], m2v[k], DLv[k], lnlik[k], posterior_snrv[k], )
            else:
                print("ok")
            param_and_val_list.append([Mv[k], qv[k], chi1v[k], chi2v[k], DLv[k], pdet])
            fh5.create_dataset(f'array_iter{k:04}', data=snr_arr)
        fh5.create_dataset('Mz_q_Xi1_Xi2_DL_pdet_lists', data=param_and_val_list)
    fh5.close()
    return 0

with open('f377', 'r')as fileall:
    ftags = [line.strip() for line in fileall.readlines()]
for ftag in ftags:
    get_mc_angles_snr(ftag, Nsample=100, MC_iters=1000)

