import numpy as np

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
array_lengths = lst.copy()


import h5py as h5
data = np.loadtxt("new100_dataMz_z_Pdet.txt").T
Mz = data[0]
z_redshift = data[1]
pdet = data[2]

for i in range(len(lst)):
    if i == 0:
        Marray = Mz[:lst[0]]
    elif i == len(lst)-1:
        Marray = Mz[-lst[i]:]
    else:
        print("indixes = ", i, sum(lst[:i-1]), sum(lst[:i]))
        Marray = Mz[sum(lst[:i]): sum(lst[:i+1])]
    #print(lst[i], type(Marray), Marray.shape)

z_redshift = data[1]
pdet = data[2]


