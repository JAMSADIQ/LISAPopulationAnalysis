import os
import numpy as np
import json
import h5py as h5
import lisabeta
import lisabeta.pyconstants as pyconstants
import lisabeta.tools.pytools as pytools
import lisabeta.lisa.pyLISAnoise as pyLISAnoise
import lisabeta.lisa.ldcnoise as ldcnoise
import lisabeta.lisa.lisa as lisa
import lisabeta.lisa.lisatools as lisatools
import astropy
from astropy import cosmology
from astropy import units
from astropy.cosmology import  z_at_value, Planck15
import matplotlib.pyplot as plt

import scipy
from scipy.special import i0, erf


def Prob_matchfilter_SNR(optimal_SNR, threshold_SNR):
    """
    Calculate the probability that matched filter SNR will be above threshold
    given the optimal SNR.

    Parameters:
    optimal_SNR (float): The optimal signal-to-noise ratio (SNR).
    threshold_SNR (float): Threshold for detection of signal
    
    Notes:
    - The error function (erf) is imported from scipy.special
    """
    optimal_SNR = np.array(optimal_SNR)
    return 0.5 * (1.0 + erf((-threshold_SNR + optimal_SNR) / np.sqrt(2)))


# Definition to randomize orientations
def draw_random_angles():
    phi = np.random.uniform(low=-np.pi, high=np.pi)
    inc = np.arccos(np.random.uniform(low=-1., high=1.))
    lambd = np.random.uniform(low=-np.pi, high=np.pi)
    beta = np.arcsin(np.random.uniform(low=-1., high=1.))
    psi = np.random.uniform(low=0., high=np.pi)
    return np.array([phi, inc, lambd, beta, psi])

 
def get_zarray(DLarray_Mpc):
    zvals = np.zeros_like(DLarray_Mpc)
    for it in range(len(zvals)):
        zvals[it] = z_at_value(Planck15.luminosity_distance,  DLarray_Mpc[it]*units.Mpc)
    print("done")
    return zvals

def lensing_distribution(dL):
    """from paper Eq3 or see Paper arXiv 1601.07112 dL eq"""
    #dL is in Mpc
    zval = z_at_value(Planck15.luminosity_distance,  float(dL)*units.Mpc)
    sigma_lense = dL * 0.066*( (1 - (1. + zval)**(-0.25))/(0.25) )**(1.8)
    return np.random.normal(loc=dL, scale=sigma_lense, size=1)

def get_mc_angles_snr(filetag, snr_threshold, Nsample=200, MC_iters=1000, apply_lensing=False):
    """
    call json file for waveform, h5 file for intrinsic params
    compute 1000 snr at one pe sample
    """
    #posterior-SNR  criteria
    data_posterior = np.loadtxt('posterior_snr/data_posteriors_DL_SNR_vals_catalog_1_yrs.txt.'+filetag+'.txt').T
    posterior_snr = data_posterior[-1]
    #only samples with snr > 4
    cidx = np.argwhere(posterior_snr > 4.0).flatten()

    # Using Updated data json file for waveform params for SNR calculations
    with open('catalog_1_yrs.txt.'+filetag+'.json', 'r') as jsonfile:
        data_json = json.load(jsonfile)
        waveform_params = data_json['waveform_params']
    with h5.File('catalog_1_yrs.txt.'+filetag+'.h5', 'r') as fh5:
        qvall = fh5['q'][...][cidx]
        Mv = fh5['M'][...][cidx]
        DLv = fh5['dist'][...][cidx]
        #zvals = get_zarray(DLv)
        lnlik = fh5['lnlike'][...][cidx]
        chi1v = fh5['chi1'][...][cidx]
        chi2v = fh5['chi2'][...][cidx]
        Deltatv = fh5['Deltat'][...][cidx]
        posterior_snrv = posterior_snr[cidx] 

        # Choose Nsamples random values
        indices = np.random.choice(np.arange(len(Mv)), int(Nsample))
        Mv = Mv[indices]
        DLv = DLv[indices]
        zvals = get_zarray(DLv)
        lnlik = lnlik[indices]
        qv = qvall[indices]
        chi1v = chi1v[indices] 
        chi2v = chi2v[indices]  
        Deltatv = Deltatv[indices] 
        posterior_snrv = posterior_snrv[indices] 
        m1v = qv*Mv / (1. + qv)
        m2v = Mv / (1. + qv)
        print("max-q, max-DL, min-lnlik = ", np.max(qv), np.max(DLv), np.min(lnlik))
    fh5.close()

    with h5.File('new_save_pdet_with_time_to_merger_randomize/out_putcatalog_1_yrs.txt.'+filetag+'.h5', 'w') as fh5:
        param_and_val_list = []
        for k in range(len(Mv)):
            params_base = {
            "m1": m1v[k], # notice lisa-beta take redshifted masses )
            "m2": m2v[k], # # m1 m2 are redshifted masses,
            "chi1": chi1v[k],
            "chi2": chi2v[k],
            "Deltat": Deltatv[k], # we are not using these values now in new analysis
            "dist": DLv[k],
            "Lframe": True
            }
            snr_arr = []
            inc_arr = []
            dist_value = DLv[k]
            # Now we use MC over angles and random time to merger
            for i in range(MC_iters):
                params_base['Deltat'] = np.random.uniform(0.0, 4.0) * 3.154e+7
                #######Modify line below for lensing
                if apply_lensing == True:
                    params_base['dist'] = lensing_distribution(dist_value)
                ###################
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
            pdet = np.sum(np.array(snr_arr) > snr_threshold)/MC_iters
            mf_pdet = sum(Prob_matchfilter_SNR(snr_arr, snr_threshold))/len(snr_arr)
            if pdet == 0.0:
                print("indx, pdet, mfpdet, m1, m2, DL, lnlik, pos_snr = ", indices[k], pdet, mf_pdet, m1v[k], m2v[k], DLv[k], lnlik[k], posterior_snrv[k], )
            else:
                print("ok")
            param_and_val_list.append([Mv[k], qv[k], chi1v[k], chi2v[k], DLv[k], mf_pdet])
            fh5.create_dataset(f'array_iter{k:04}', data=snr_arr)
        fh5.create_dataset('Mz_q_Xi1_Xi2_DL_mfpdet_lists', data=param_and_val_list)
    fh5.close()
    return 0


# ftag are all files tag like 49.101  in a column file
with open('ftag', 'r') as fileall:
    ftags = [line.strip() for line in fileall.readlines()]
for ftag in ftags:
    get_mc_angles_snr(ftag, snr_threshold=8., Nsample=100, MC_iters=1000, apply_lensing=True)

