import numpy as np
import h5py as h5


fname = []
with open('cutfilename', 'r')as file:
    lines = file.readlines()
    for line in lines:
        fname.append(line.strip())

fulllist = [200]*91
lst = [200]*91
lst[5] = 187
lst[7] = 195
lst[25] = 104
lst[28] = 198
lst[29] = 198
lst[35] = 198
lst[44] = 190
lst[45] = 196
lst[48] = 197
lst[51] = 199
lst[53] = 185
lst[55] = 173
lst[57] = 185
lst[58] = 199
lst[59] = 174
lst[64] = 190
lst[69] = 192
lst[70] = 162
lst[72] = 197
lst[79] = 163
lst[80] = 195
lst[82] = 198
lst[86] = 194
lst[87] = 197


data = np.loadtxt("new100_dataMz_z_Pdet.txt").T
Mz = data[0]
z_redshift = data[1]
pdet = data[2]

median_Mz = []
median_z = []
#Line numbers  and get data fo samples and save them with medians as well
hf = h5.File("catalog200samples_data.hdf5", "w")
Mzgrp = hf.create_group("Mz_samples")
zgrp = hf.create_group("z_samples")
pdetgrp = hf.create_group("Pdet_samples")

for i in range(len(lst)):
    filename = fname[i]
    print(filename)
    if i == 0:
        Marray = Mz[:lst[0]]
        zarray = z_redshift[:lst[0]]
        pdetarray = pdet[:lst[0]]
    elif i == len(lst)-1:
        Marray = Mz[-lst[i]:]
        zarray = z_redshift[-lst[i]:]
        pdetarray = pdet[-lst[i]:]
    else:
        Marray = Mz[sum(lst[:i]): sum(lst[:i+1])]
        zarray = z_redshift[sum(lst[:i]): sum(lst[:i+1])]
        pdetarray = pdet[sum(lst[:i]): sum(lst[:i+1])]
    median_Mz.append(np.median(Marray))
    median_z.append(np.median(zarray))
    
    Mzgrp.create_dataset(filename, data=Marray)
    zgrp.create_dataset(filename, data=zarray)
    pdetgrp.create_dataset(filename, data=pdetarray)


hf.create_dataset("Mzmedians", data=median_Mz)
hf.create_dataset("zmedians", data=median_z)
hf.close()

