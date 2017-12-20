# -*- coding: utf-8 -*-

from glob import glob

from h5py import File
from numpy import average
from dask.array import from_delayed, concatenate, cov
from dask.delayed import delayed
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt


# %%
offset = 4000
fr, to = 5500, 6500


def shapes(filename):
    with File(filename, 'r') as f:
        bp = f['Background_Period'][...]
        bunches = f['bunches'][...]
    return (bunches % bp != 0).sum()
    

@delayed
def process(filename):
    with File(filename, 'r') as f:
        bp = f['Background_Period'][...]
        bunches = f['bunches'][...]
        tofs = f['digitizer/channel3'][:, 0:to].astype('float64')
        arrs = (average(tofs[:, 0:offset], 1)[:, None] - tofs)[:, fr:]
        where = bunches % bp != 0
    return arrs[where, :]


# %%
run = 365
globbed = glob('/home/ldm/ExperimentalData/Online4LDM/20144078'
               '/Test/Run_{:03d}/rawdata/*.h5'.format(run))
arr = concatenate([from_delayed(process(g), [shapes(g), to-fr], 'float64')
                   for g in globbed])
with ProgressBar():
    img = cov(arr.T).compute()

# %%
plt.figure()
plt.pcolormesh(img, cmap='RdBu')
m = min(abs(img.min()), abs(img.max()))*0.5
plt.clim(-m, m)
plt.colorbar()
plt.title('cov map run={}'.format(run))
plt.show()
