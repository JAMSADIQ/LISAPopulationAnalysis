from sensitivity import Sn
from waveforms import fMax, Aeff, AeffphenomA
import numpy as np
import scipy.integrate as integrate
import pandas as pd
from constants import *
import traceback
import matplotlib.pyplot as plt

def save_data_to_file(data, filename='Latest_enrico_code_output_dataM_DL_Pdet.txt'):
    try:
        with open(filename, 'a') as file:
             file.write('        '.join(map(str, data)) + '\n')
    except Exception as e:
        print(f"Error saving data: {e}")



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
combine_M100= []
combine_DL_median= [] #get redshift will be too much for all samples
combine_z_median= []
combine_DL100= []
combine_pdet = []
combine_indices = []
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
        indices = np.random.choice(len(Mv), 400)
        combine_indices.append(indices)
        Mz = Mv[indices]
        q = qv[indices]
        m1z = Mz*q/(1.+q)
        m2z = Mz*1.0/(1.+q)
        chi1 = chi1v[indices]
        chi2 = chi2v[indices]
        DL =  DLv[indices]
        
        for  jx in range(len(Mz)):
            pdet, snr = get_pdet(m1z[jx], m2z[jx], DL[jx], chi1[jx], chi2[jx])
            if pdet == 0.0:
                print("not detectable with, Mz, DL =", Mz[jx], DL[jx])
            else:
                save_data_to_file([Mz[jx], DL[jx] , pdet])
                print(f"Iteration {jx} completed: {Mz[jx]}  {pdet}")
                print("M, DL, pdet = ", Mz[jx], DL[jx] , pdet)
                combine_M100.append(Mz[jx])
                combine_DL100.append(DL[jx])
                combine_pdet.append(pdet)
            Mzjx = Mz[jx]
            DLjx = DL[jx]
           # f_name.write(f'{Mzjx}  \t {DLjx}  \t {pdet} \n')
                
        chose_M = Mv.copy()#np.random.choice(M, 200).tolist()
        median_M = np.median(chose_M)
        chose_DL = DLv.copy()
        median_DL = np.median(chose_DL)
        median_z = z_at_value(Planck15.luminosity_distance,  float(median_DL)*units.Mpc).value
        combine_z_median.append(median_z)
        combine_DL_median.append(median_DL)
        combine_M_median.append(median_M)
        fh5.close()

#f_name.close()

DL_median = np.array(combine_DL_median)
z_median = np.array(combine_z_median) #z= 
DL100 = np.hstack(combine_DL100)
Mtot_median = np.array(combine_M_median)
Mtot100 = np.hstack(combine_M100)
pdet100 = np.hstack(combine_pdet)
np.savetxt("median100Posteriors_dataMz_z_DL.txt", np.c_[Mtot_median, z_median, DL_median])
np.savetxt("new100_dataMz_DL_Pdet.txt", np.c_[Mtot100, DL100, pdet100])
np.savetxt("randomindices_perfile.txt", np.c_[combine_indices])

def get_zarray(DLarray_Mpc):
    zvals = np.zeros_like(DLarray_Mpc)
    for it in range(len(zvals)):
        zvals[it] = z_at_value(Planck15.luminosity_distance,  float(DLarray_Mpc[it])*units.Mpc)
    print("done")
    return zvals
z200 = get_zarray(DL200)
print("total detected event =", len(z200))
np.savetxt("new100_dataMz_z_Pdet.txt", np.c_[Mtot200, z200, pdet200])



weightspdet = 1.0/pdet200
weights_obs = weightspdet/np.sum(weightspdet) #/200 for 200 samples per-event


#Plots with intrinsic Data
data = np.loadtxt("/u/j/jsadiq/Documents/E_awKDE/CatalogLISA/EnricotheoreticalData/popIII/Recompute/dataPopIII.dat").T
TheoryMtot = data[0]
print(len(TheoryMtot))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.title("popIII model")
ax1.hist(np.log10(Mtot200), density=True, weights=weights_obs,bins=25, histtype='step', label='Weighted', lw=2)
ax1.hist(np.log10(TheoryMtot), density=True, bins=25, histtype='step', label='intrinsic', lw=2, ls='--')
ax1.hist(np.log10(Mtot200), density=True,bins=25, histtype='step', label='Unweighted-observed', lw=1, ls='--')
ax1.semilogx()
ax1.set_xlabel("Log10[Mz]", fontsize=18)
ax1.legend()

Theory_z = data[1]
#plt.figure()
plt.title("popIII model")
ax2.hist(z200, density=True, bins=30, weights=weights_obs, histtype='step', label='weighted', lw=2)
ax2.hist(Theory_z, density=True, bins=30, histtype='step', ls='--', label='intrinsic', lw=2)
ax2.hist(z200, density=True, bins=30, histtype='step', label='Unweighted', lw=1, ls='--')

ax2.set_xlabel("z", fontsize=18)
ax2.legend(fontsize=16)
fig.suptitle('popIII model', fontsize=16)
plt.tight_layout()
#plt.savefig("popIII_histogram.png")
plt.show()
quit()


#KDE results

#Construct a KDE and plot it
LogM200 = np.asarray(np.log10(Mtot200))
z_eval = np.logspace(-2, np.log10(20), 200)
M_eval = np.logspace(2, 10, 200)
LogM_eval = np.log10(M_eval)

pdfZ_int = gaussian_kde(Theory_z, bw_method='mine')
print("kde prepared = ", pdfZ_int)

Theory_z_kde = pdfZ_int(z_eval)
print("kde evaluates")

print("shape of z200 = ", z200.shape)
pdfZ = gaussian_kde(z200, weights=weights_obs)#, bw_method='mine')
pz_kde = pdfZ(z_eval)
pdfZnw = gaussian_kde(z200)#, bw_method='mine')
pz_kdenw = pdfZnw(z_eval)

LogMTh = np.log10(TheoryMtot)
pdfMtot_int = gaussian_kde(LogMTh, bw_method='scott')
Theory_pMtot_kde = pdfMtot_int(LogM_eval)
pdfMtot = gaussian_kde(LogM200, weights=weights_obs, bw_method='scott')
pMtot_kde = pdfMtot(LogM_eval)
pdfMtotnw = gaussian_kde(LogM200, bw_method='scott')
pMtot_kdenw = pdfMtotnw(LogM_eval)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(LogM_eval, pMtot_kde, lw=2, label='Weighted kde')
ax1.plot(LogM_eval, Theory_pMtot_kde, lw=2,  label='intrinsic')
ax1.plot(LogM_eval, pMtot_kdenw, lw=2, label='Unweighted')
ax1.set_xlabel("Log10[Mtot]", fontsize = 18)
ax1.set_ylabel("p(Log10[Mtot])", fontsize=18)
ax1.legend(fontsize=13, loc=1)
ax2.plot(z_eval, pz_kde, lw=2, label='Weighted')
ax2.plot(z_eval, Theory_z_kde, lw=2, label='intrinsic')
ax2.plot(z_eval, pz_kdenw, lw=2, label='Unweighted')
ax2.set_xlabel("z", fontsize = 18)
ax2.set_ylabel("p(z)", fontsize=18)
ax2.legend(fontsize=13)
fig.suptitle('popIII model', fontsize=16)
plt.tight_layout()
plt.show()

