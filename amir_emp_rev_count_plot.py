# Reversals plots
# one food and two foods (60, 90, 120)
# September 2019 

import numpy as np
import matplotlib.pyplot as plt

RE_af1_1 = np.loadtxt('UMAP_data/reversal_locs/reversal_locs1.txt')
RE_af1_2 = np.loadtxt('UMAP_data/reversal_locs/reversal_locs2.txt')
RE_af1_3 = np.loadtxt('UMAP_data/reversal_locs/reversal_locs3.txt')

RE_af1_1_f = RE_af1_1.flatten()
RE_af1_2_f = RE_af1_2.flatten()
RE_af1_3_f = RE_af1_3.flatten()

RE_af1_1_nz = RE_af1_1_f[np.nonzero(RE_af1_1_f)]
RE_af1_2_nz = RE_af1_2_f[np.nonzero(RE_af1_2_f)]
RE_af1_3_nz = RE_af1_3_f[np.nonzero(RE_af1_3_f)]

fig = plt.figure(1)
plt.hist((180/np.pi)*RE_af1_1_nz, np.linspace(-180,180,25))

fig = plt.figure(2)
plt.hist((180/np.pi)*RE_af1_2_nz, np.linspace(-180,180,25))

fig = plt.figure(3)
plt.hist((180/np.pi)*RE_af1_3_nz, np.linspace(-180,180,25))

fig = plt.figure(10)
plt.hist((180/np.pi)*RE_af1_1_nz, np.linspace(-180,180,25))
plt.hist((180/np.pi)*RE_af1_2_nz, np.linspace(-180,180,25))
plt.hist((180/np.pi)*RE_af1_3_nz, np.linspace(-180,180,25))

plt.show()