# get redshift from dL and dL from redshift
import numpy as np
import matplotlib.pyplot as plt


import astropy
import astropy.units as u
from astropy.cosmology import  z_at_value, Planck15
from astropy.cosmology import Planck15 as cosmo


def get_zarray(DLarray_Mpc):
    zvals = np.zeros_like(DLarray_Mpc)
    for it in range(len(zvals)):
        zvals[it] = z_at_value(Planck15.luminosity_distance,  float(DLarray_Mpc[it])*u.Mpc)
    print("done")
    return zvals

#default output unit is Mpc
def get_dLGpc(z_array):
    dL = Planck15.luminosity_distance(redshift)
    # Convert the result to megaparsecs (Mpc)
    dL_Mpc = dL.to(u.Gpc)
    return dL_Mpc.value

#TEST
dLMpc = [1000., 20000., 200000., 290000.]
redshift = get_zarray(dLMpc)
for i in range(4):
    print("dLMpc given, z got = ", dLMpc[i] , redshift[i])

luminosity_distance_Gpc = get_dLGpc(redshift)
for i in range(4):
    print("z given, dL Gpc got = ", redshift[i], luminosity_distance_Gpc[i])
quit()
# Plot redshift vs luminosity distance
plt.scatter(redshift, luminosity_distance_Gpc, alpha=0.5)
plt.title('Luminosity Distance vs Redshift (Planck 15 Cosmology)')
plt.xlabel('Redshift')
plt.ylabel('Luminosity Distance (Gpc)')
plt.grid(True)
plt.show()

