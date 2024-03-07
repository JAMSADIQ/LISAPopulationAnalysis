import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("new100_dataMz_z_Pdet.txt").T
z100 = data[1]
Mtot100 = data[0]
pdet100 = data[2]




weightspdet = 1.0/pdet100
weights_obs = weightspdet/np.sum(weightspdet) #/100 for 100 samples per-event


#Plots with intrinsic Data
data = np.loadtxt("/u/j/jsadiq/Documents/E_awKDE/CatalogLISA/EnricotheoreticalData/LS-nod-noSN/Recomputed/dataLSnodnoSN.dat").T
TheoryMtot = data[0]
print(len(TheoryMtot))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.title("popIII model")
ax1.hist(np.log10(Mtot100), density=True, weights=weights_obs,bins=25, histtype='step', label='Weighted', lw=2)
ax1.hist(np.log10(TheoryMtot), density=True, bins=25, histtype='step', label='intrinsic', lw=2, ls='--')
ax1.hist(np.log10(Mtot100), density=True,bins=25, histtype='step', label='Unweighted-observed', lw=1, ls='--')
ax1.semilogx()
ax1.set_xlabel("Log10[Mz]", fontsize=18)
ax1.legend()

Theory_z = data[1]
#plt.figure()
plt.title("popIII model")
ax2.hist(z100, density=True, bins=30, weights=weights_obs, histtype='step', label='weighted', lw=2)
ax2.hist(Theory_z, density=True, bins=30, histtype='step', ls='--', label='intrinsic', lw=2)
ax2.hist(z100, density=True, bins=30, histtype='step', label='Unweighted', lw=1, ls='--')

ax2.set_xlabel("z", fontsize=18)
ax2.legend(fontsize=16)
fig.suptitle('popIII model', fontsize=16)
plt.tight_layout()
#plt.savefig("popIII_histogram.png")
plt.show()
#quit()


#KDE results

#Construct a KDE and plot it
LogM100 = np.asarray(np.log10(Mtot100))
z_eval = np.logspace(-2, np.log10(20), 200)
M_eval = np.logspace(2, 10, 200)
LogM_eval = np.log10(M_eval)

pdfZ_int = gaussian_kde(Theory_z, bw_method='mine')
print("kde prepared = ", pdfZ_int)

Theory_z_kde = pdfZ_int(z_eval)
print("kde evaluates")

print("shape of z100 = ", z100.shape)
pdfZ = gaussian_kde(z100, weights=weights_obs)#, bw_method='mine')
pz_kde = pdfZ(z_eval)
pdfZnw = gaussian_kde(z100)#, bw_method='mine')
pz_kdenw = pdfZnw(z_eval)

LogMTh = np.log10(TheoryMtot)
pdfMtot_int = gaussian_kde(LogMTh, bw_method='scott')
Theory_pMtot_kde = pdfMtot_int(LogM_eval)
pdfMtot = gaussian_kde(LogM100, weights=weights_obs, bw_method='scott')
pMtot_kde = pdfMtot(LogM_eval)
pdfMtotnw = gaussian_kde(LogM100, bw_method='scott')
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

