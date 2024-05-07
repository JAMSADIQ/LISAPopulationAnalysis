############Imports
import sys
import numpy as np
import matplotlib.pyplot as plt

import os
import h5py as h5
import itertools
import copy
import json
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
from astropy import cosmology
from astropy import units
from astropy.cosmology import  z_at_value, Planck15
def get_zarray(DLarray_Mpc):
    zvals = np.zeros_like(DLarray_Mpc)
    for it in range(len(zvals)):
        zvals[it] = z_at_value(Planck15.luminosity_distance,  float(DLarray_Mpc[it])*units.Mpc)
    return zvals


#specify these from Params files
def get_optimalSNR(m1z, m2z, chi1, chi2, dist, waveform_params):
    Deltat = 0.0
    phi = np.random.uniform(low=-np.pi, high=np.pi)
    inc = np.arccos(np.random.uniform(low=-1., high=1.))
    lamBda = np.random.uniform(low=-np.pi, high=np.pi)
    beta = np.arcsin(np.random.uniform(low=-1., high=1.))
    psi = np.random.uniform(low=0., high=np.pi)

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
    tdisignal = lisa.GenerateLISATDISignal_SMBH(params_base, **waveform_params)
    return tdisignal['SNR']


##############File like PE sampls json file with deltaT = 0 and tob = 1####
with open('standardjsonfile.json', 'r') as filej:
        data = json.load(filej)
waveform_params = data['waveform_params']

import pathlib
#######combine all files for 100 years of Catalogs 
listfile = list(pathlib.Path('./').glob('*.txt.*'))
snr_threshold = 8.0
snr_arr = []
Mall = []
zall = []
Mz_det = []
z_det = []

def swap_elements(m1val, m2val):
    """
    since catalog data can have m1> m2 or vice versa
    """
    result_m1val = []
    result_m2val = []
    for val1, val2 in zip(m1val, m2val):
        if val1 > val2:
            result_m1val.append(val1)
            result_m2val.append(val2)
        else:
            result_m1val.append(val2)
            result_m2val.append(val1)
    return result_m1val, result_m2val

for filename in listfile:
    print(filename)
    d = np.genfromtxt(filename, skip_header=1).T
    z_cosmo = d[0]
    #we need planck cosmology
    dL = d[1]
    z = get_zarray(dL)
    m1 = d[2]
    m2 = d[3]
    m1, m2 = swap_elements(m1, m2)
    m1z = m1 *(1.0 + z)
    m2z = m2 *(1.0 + z)
    xi1 = d[4]
    xi2 = d[5]
    Mz = m1z + m2z 
    indx = np.argwhere(np.isnan(Mz))
    Mzv = np.delete(Mz, indx)
    dLv = np.delete(dL, indx)
    zv = np.delete(z, indx)
    Mall.append(Mzv)
    zall.append(zv)
    m1zv = np.delete(m1z, indx)
    m2zv = np.delete(m2z, indx)
    xi1v = np.delete(xi1, indx)
    xi2v = np.delete(xi2, indx)
    ############### Compute optimal SNR ################# 
    for k in range(len(Mzv)):
        print("totalsamples = ", len(Mzv))
        print("m1, m2 = ", m1zv[k], m2zv[k])
        snr = get_optimalSNR(m1zv[k], m2zv[k], xi1v[k], xi2v[k], dLv[k], waveform_params)
        if snr >= snr_threshold:
            snr_arr.append(snr)
            Mz_det.append(Mzv[k])
            z_det.append(zv[k])
        else:
            print("non detected")

Mzcombine = np.array(Mz_det)
zcombine = np.array(z_det)
snrcombine = np.array(snr_arr)
np.savetxt("detected_intrinsic_Mz_z_optSNR.txt", np.c_[Mzcombine, zcombine, snrcombine])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.hist(np.log10(Mall), density=True, bins=50, histtype='step',  lw=2, color='k', ls='--', label = 'allevents')
ax1.hist(np.log10(Mzcombine), density=True, bins=50, histtype='step', lw=2, color='r', label='detected events')
ax1.set_xlabel("Log10[Mz]")
ax1.legend()
ax2.hist(zall, density=True, bins=50, histtype='step', lw=2, ls='--', color='k', label = 'allevents')
ax2.hist(zcombine, density=True, bins=50, histtype='step', lw=2, color='r',  label='detected events')
ax2.set_xlabel("z")
ax2.set_xlim(xmax =21)
ax2.legend()
fig.suptitle('popIII intrinsic 100 years', fontsize=16)
plt.tight_layout()
plt.show()

