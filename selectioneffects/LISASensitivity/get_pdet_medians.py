from sensitivity import Sn
from waveforms import fMax, Aeff, AeffphenomA
import numpy as np
import scipy.integrate as integrate
import pandas as pd
from constants import *


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



def get_pdet(m1z, m2z, dL, chi1, chi2, num_samples=1000,nr_fixed = 8.0, tobs = 4.0):
    total = 0
    num_samples=1000
    snr_fixed = 8.0
    snr_array = np.zeros(num_samples)
    for ix in range(num_samples):
        psi = np.random.uniform(0, 2*np.pi)
        theta = np.arccos(np.random.uniform(-1.,1.))
        phi = np.random.uniform(0, 2*np.pi)
        iota = np.arccos(np.random.uniform(-1.,1.))
        tc = np.random.uniform(0.0, 1.0)
        snr_sq_val = SNR(m1z, m2z, dL, chi1, chi2, psi,theta,phi,iota, tc, tobs)**2
        snr_array[ix] = np.sqrt(snr_sq_val)
        total += snr_sq_val 
    pdet = np.sum(snr_array > snr_fixed)/num_samples
    ave_snr = np.sqrt(total/num_samples)
    return pdet, ave_snr

import numpy as np
import astropy
from astropy import cosmology
from astropy import units
from astropy.cosmology import  z_at_value, Planck15#
#Here I need to call my files to get m1, m2, dL chi1, chi2 and save Pdet 
combine_M_median= []
combine_M200= []
combine_DL_median= [] #get redshift will be too much for all samples
combine_z_median= []
combine_z200= []
combine_pdet = []
import h5py as h5
file_path = 'filename' 
with open(file_path, 'r')as file:
    lines = file.readlines()
    for line in lines:
        f = line.strip()
        print("processing file = ", f)
        fh5 = h5.File(f, "r")
        Mv = fh5['M'][...]
        DLv = fh5['dist'][...]
        qv = fh5['q'][...]
        chi1v = fh5['chi1'][...]
        chi2v = fh5['chi2'][...]
        print("max-min Xi = ", np.min(chi1v), np.max(chi1v))
        Mz = np.median(Mv)
        q = np.median(qv)
        m1z = Mz*q/(1.+q)
        m2z = Mz*1.0/(1.+q)
        chi1 = np.median(chi1v)
        chi2 = np.median(chi2v)
        DL =  np.median(DLv)
        
        pdet, snr = get_pdet(m1z, m2z, DL, chi1, chi2)
        if pdet == 0.0:
            print("not detectable")
        else:
            print("pdet = ", pdet)
        combine_M200.append(Mz)
        combine_z200.append(DL)
        combine_pdet.append(pdet)
                
        median_z = z_at_value(Planck15.luminosity_distance,  float(DL)*units.Mpc).value
        combine_z_median.append(median_z)
        fh5.close()

z_median = np.array(combine_z_median) #z= 
DL200 = np.hstack(combine_z200)
Mtot200 = np.hstack(combine_M200)
pdet200 = np.hstack(combine_pdet)
np.savetxt("Medians_dataMz_DL_z_Pdet.txt", np.c_[Mtot200, DL200, z_median, pdet200])
