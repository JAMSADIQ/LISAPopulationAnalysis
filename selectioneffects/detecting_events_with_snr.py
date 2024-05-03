from sensitivity import Sn
from waveforms import fMax, Aeff, AeffphenomA
import numpy as np
import scipy.integrate as integrate
import pandas as pd
from constants import *
import traceback
import matplotlib.pyplot as plt
import h5py as h5
def Fp(psi, theta, phi):
    return 0.5 * np.cos(2 * psi) * (1 + np.cos(theta)**2) * np.cos(2 * phi) - np.sin(2 * psi) * np.cos(theta) * np.sin(2 * phi)

def Fx(psi, theta, phi):
    return 0.5 * np.sin(2 * psi) * (1 + np.cos(theta)**2) * np.cos(2 * phi) + np.cos(2 * psi) * np.cos(theta) * np.sin(2 * phi)

def Q2(psi, theta, phi, iota):
    return Fp(psi, theta, phi)**2 * ((1 + np.cos(iota)**2) / 2)**2 + Fx(psi, theta, phi)**2 * np.cos(iota)**2


Tobs=4
df=1/(Tobs*yr)

def integrandSNRav(f,m1z,m2z,dL,chi1,chi2,tc,duration):
    ndet=2
    Q=2/5
    return 4*Q**2*ndet*Aeff(f,m1z,m2z,dL,chi1,chi2)**2/Sn(f,duration)

def integrandSNR(f,m1z,m2z,dL,chi1,chi2,psi,theta,phi,iota,tc,duration):
    ndet=2
    return 4*Q2(psi, theta, phi, iota)*ndet*Aeff(f,m1z,m2z,dL,chi1,chi2)**2/Sn(f,duration)

def integrandSNRavPhenomA(f,m1z,m2z,dL,tc,duration):
    ndet=2
    Q=2/5
    return 4*Q**2*ndet*AeffphenomA(f,m1z,m2z,dL)**2/Sn(f,duration)

def f0(tau,m1z,m2z):
    eta=m1z*m2z/(m1z+m2z)**2
    Mc=eta**(3/5)*(m1z+m2z)
    return 1/np.pi*(5/256*1/(tau*yr))**(3/8)*(G*Mc*Msun/c**3)**(-5/8)


def SNRav(m1z,m2z,dL,chi1,chi2,tc,duration):
    finitial=f0(tc,m1z,m2z)
    f2=min(fMax(m1z,m2z,dL,chi1,chi2),1)
    f1=max(finitial,1.e-5)
    I,err=integrate.quad(integrandSNRav,f1,f2,args=(m1z,m2z,dL,chi1,chi2,tc,duration))
    return np.sqrt(I)

def SNR(m1z,m2z,dL,chi1,chi2,psi,theta,phi,iota,tc,duration):
    """
    intrinsic params,  m1z, m2z, DL, Xi1, Xi2
    sky angles for MC  psi, theta, phi, iota
    and
    tc =   do I need MC here [0-1] np.random.uniform
    duration =  tobs [time to merger?] fixed
    """
    finitial=f0(tc,m1z,m2z)
    f2=min(fMax(m1z,m2z,dL,chi1,chi2),1)
    f1=max(finitial,1.e-5)
    I,err=integrate.quad(integrandSNR,f1,f2,args=(m1z,m2z,dL,chi1,chi2,psi,theta,phi,iota,tc,duration))
    return np.sqrt(I)

def SNRavPhenomA(m1z,m2z,dL,tc,duration):
    finitial=f0(tc,m1z,m2z)
    f2=min(fMax(m1z,m2z,dL,0,0),1)
    f1=max(finitial,1.e-5)
    I,err=integrate.quad(integrandSNRavPhenomA,f1,f2,args=(m1z,m2z,dL,tc,duration))
    return np.sqrt(I)


m1arr, m2arr, dLarr, xi1arr, xi2arr = np.loadtxt('all_intrinsic_catalog_data_100years_m1_m2_dL_xi1_xi2.dat', unpack=True)
from astropy import cosmology
from astropy import units
from astropy.cosmology import  z_at_value, Planck15

def get_zarray(DLarray_Mpc):
    zvals = np.zeros_like(DLarray_Mpc)
    for it in range(len(zvals)):
        zvals[it] = z_at_value(Planck15.luminosity_distance,  float(DLarray_Mpc[it])*units.Mpc)
    print("done")
    return zvals


detectMz = []
detectz = []
z = get_zarray(dLarr)
m1z = m1arr*(1.0 + z)
m2z = m2arr*(1.0 + z)
Mz = m1z + m2z
tobs = 4.0
snr_fixed = 8.0
for i in range(len(m1arr)):
    psi = np.random.uniform(0, 2*np.pi)
    theta = np.arccos(np.random.uniform(-1.,1.))
    phi = np.random.uniform(0, 2*np.pi)
    iota = np.arccos(np.random.uniform(-1.,1.))
    tc = np.random.uniform(0.0, 1.0)
    snr_sq_val = SNR(m1z[i], m2z[i], dLarr[i], xi1arr[i], xi2arr[i], psi,theta,phi,iota, tc, tobs)**2
    if np.sqrt(snr_sq_val) >= snr_fixed:
        print("detect Mz, z =  ",  Mz[i], z[i])
        detectMz.append(Mz[i])    
        detectz.append(z[i])    
    else:
        print("not detected")

np.savetxt("detected_intrinsiceventsparamsMz_z.txt", np.c_[detectMz, detectz])
