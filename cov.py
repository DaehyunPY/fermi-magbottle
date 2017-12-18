# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %% add path 
from glob import glob

from h5py import File
from cytoolz import partial, reduce
from numpy import average
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt


# %%
def process(filename):
    with File(filename, 'r') as f:
        offset = 4000
        fr, to = 5000, 8000
        bp = f['Background_Period'][...]
        bunches = f['bunches'][...]
        where = bunches%bp != 0
        tofs = f['digitizer/channel3'][where, 0:to].astype('int64')
        intensities = (f['photon_diagnostics/FEL01/I0_monitor/iom_sh_a_pc']
                        [where].astype('float64'))
        norm = average(tofs[:, 0:offset], 1)
        # arrs = (norm[:, None]-tofs)[:, fr:]
        arrs = (norm[:, None]-tofs)[:, fr:]/intensities[:, None]
        n = where.sum()
        return {
            'n': n,
            'avg': sum(arrs)/n,
            'out': sum(arr[:, None]@arr[None, :] for arr in arrs)/n
        }


def unpack(d):
    return {
        'n': d['n'],
        'avg': d['avg']*d['n'],
        'out': d['out']*d['n']
    }


def sumup(d1, d2):
    return {
        'n': d1['n']+d2['n'],
        'avg': d1['avg']+d2['avg'],
        'out': d1['out']+d2['out']
    }


def pack(d):
    return {
        'n': d['n'],
        'avg': d['avg']/d['n'],
        'out': d['out']/d['n']
    }


# %%
globbed = glob('/home/ldm/ExperimentalData/Online4LDM/20144078'
               '/Test/Run_132/rawdata/*.h5')  # change run number!
with ProgressBar():
    d = pack(
        from_sequence(globbed)
            .map(process)
            .map(unpack)
            .reduction(partial(reduce, sumup), partial(reduce, sumup))
            .compute()
    )

# %%
plt.figure()
plt.subplot(121)
plt.plot(d['avg'])

plt.subplot(122)
img = d['out']-d['avg'][:, None]@d['avg'][None, :]
plt.imshow(img, cmap='RdBu')
lim = 800, 1200  # rescale region of interest!
plt.xlim(*lim)
plt.ylim(*lim)
m = min(abs(img.min()), abs(img.max()))
# m = 20  # rescale color map!
plt.clim(-m, m)
plt.colorbar()
plt.show()

