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


#########Note we are using fixed t0 and timetomerger_max that can change SNR
waveform_params_mbhb = {
    "minf": 1e-4,
    "maxf": 0.5,
    "t0": 0.0, # NOTE: this is set to tc_guess converted in yrs. 
    # tc = t0 + Deltat. Choosing t0=0 is same as tc = Deltat, which is how the parameter files are defined.
    "timetomerger_max": 4,
    "fend": None,
    "tmin": None,
    "tmax": None,
    "phiref": 0.0,
    "fref_for_phiref": 0.0,
    "tref": 0.0,
    "fref_for_tref": 0.0,
    "force_phiref_fref": True,
    "toffset": 0.0,
    "modes": None,
    "TDI": "TDIAET",
    "acc": 1e-4,
    "order_fresnel_stencil": 0,
    "approximant": "IMRPhenomHM",
    "modes": [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)],
    "LISAconst": "Proposal",
    "responseapprox": "full",
    "frozenLISA": False,
    "TDIrescaled": False,
    "LISAnoise": { # NOTE: this is SciRDv1, not the Sangria training set noise
        "InstrumentalNoise": "SciRDv1",
        "WDbackground": True,
        "WDduration" : 3.0,
        "lowf_add_pm_noise_f0": 0.0,
        "lowf_add_pm_noise_alpha": 2.0
    }
}



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

parameters = ['m1', 'm2', 'chi1', 'chi2', 'Deltat', 'dist', 'inc', 'phi', 'lambda', 'beta', 'psi']
import h5py as h5
import pathlib
listfile = list(pathlib.Path('./').glob('*.h5'))

for i in range(len(listfile)):
    eventNameh5 = os.path.basename(listfile[i])
    eventName = os.path.splitext(eventNameh5)[0]
    print("filename   = ", eventName)
    snr_lists = []
    q_lists =  []
    DL_lists = []
    with h5.File(listfile[i], 'r') as fh5:
        Mv = fh5['M'][...]
        DLv = fh5['dist'][...]
        qv = fh5['q'][...]
        m1v = qv*Mv / (1. + qv)
        m2v = Mv / (1. + qv)
        chi1v = fh5['chi1'][...]
        chi2v = fh5['chi2'][...]
        incv = fh5['inc'][...]
        phiv = fh5['phi'][...]
        lambdav = fh5['lambda'][...]
        psiv = fh5['psi'][...]
        betav = fh5['beta'][...]
        Deltatv = fh5['Deltat'][...]
        #z_vals = get_zarray(DLv) 
        fh5.close()
        for k in np.random.choice(np.arange(len(Mv)), 500):#range(len(Mv)):
            print("k =", k)
            params={}
            plist = [m1v[k], m2v[k], chi1v[k], chi2v[k], Deltatv[k], DLv[k], incv[k], phiv[k], lambdav[k], betav[k], psiv[k]]
            for i, item in enumerate(parameters):
                params[parameters[i]] = plist[i]
            params['Lframe'] = True
            tdisignal = lisa.GenerateLISATDISignal_SMBH(params, **waveform_params_mbhb)
            snr_val = tdisignal['SNR']
     
            snr_lists.append(snr_val)
            q_lists.append(qv[k])
            DL_lists.append(DLv[k])
    
    medianMz = np.median(Mv)
    z_lists = get_zarray(DL_lists)
    medianDL = np.median(DLv)
    
    plt.figure()
    plt.plot(z_lists, snr_lists, 'b+')
    plt.xlabel("z", fontsize=15)
    plt.ylabel("opt-snr")
    plt.title(f"with IMRphenomHM, " +eventName)
    plt.savefig("z_For"+eventName+".png")
    plt.figure()
    plt.plot(q_lists, snr_lists, 'b+')
    plt.xlabel("q", fontsize=15)
    if np.max(qv)>10:
        plt.semilogx()
    plt.ylabel("opt-snr")
    plt.title(f"with IMRphenomHM, " +eventName)
    plt.savefig("q_For"+eventName+".png")
    plt.show()
