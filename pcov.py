# -*- coding: utf-8 -*-

from glob import glob

from h5py import File
from numpy import average, append
from dask.array import from_delayed, concatenate, cov
from dask.delayed import delayed
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt


# %%
offset = 4000
fr, to = 5750, 6500


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
        tofs = f['digitizer/channel1'][:, 0:to].astype('float64')
        arrs = (average(tofs[:, 0:offset], 1)[:, None] - tofs)[:, fr:]
        # intensities = (
        #     f['photon_diagnostics/FEL01/I0_monitor/iom_sh_a_pc'][...]
        #         .astype('float64')
        # )
        intensities = arrs.sum(1)
        where = bunches % bp != 0
    return append(intensities[where, None], arrs[where, :], axis=1)


# %%
run = 452
globbed = glob('/home/antoine/online4ldm_local/20144078'
               '/Test/Run_{:03d}/rawdata/*.h5'.format(run))[:10]
arr = concatenate([from_delayed(process(g), [shapes(g), to-fr+1], 'float64')
                   for g in globbed])
with ProgressBar():
    img = cov(arr.T).compute()

# %%
def scale(arr):
    m = min(abs(arr.min()), abs(arr.max()))*0.5
    return -m, m


img_cov = img[1:, 1:]
img_icov = img[1:, 0][:, None] @ img[0, 1:][None, :] / img[0, 0]
img_pcov = img_cov - img_icov

plt.figure()
plt.subplot(221)
plt.pcolormesh(img_pcov, cmap='RdBu')
plt.clim(*scale(img_pcov))
plt.colorbar()
plt.title('pcov run={}'.format(run))

plt.subplot(222)
plt.pcolormesh(img_cov, cmap='RdBu')
plt.clim(*scale(img_cov))
plt.colorbar()
plt.title('cov run={}'.format(run))

plt.subplot(223)
plt.pcolormesh(img_icov, cmap='RdBu')
plt.clim(*scale(img_icov))
plt.colorbar()
plt.title('icov run={}'.format(run))
plt.show()
