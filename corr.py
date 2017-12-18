# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %% add path 
from glob import glob

from h5py import File
from numpy import average, histogram, arange, linspace
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
from dask.multiprocessing import get as multiprocessing_get
import matplotlib.pyplot as plt


# %%
def spectra(filename):
    with File(filename, 'r') as f:
        bp = f['Background_Period'][...]
        bunches = f['bunches'][...]
        where = bunches%bp != 0
        yield from f['digitizer/channel3'][where, ...].astype('int64')


globbed = glob(
    '/Volumes/store/20144078'
    # '/home/ldm/ExperimentalData/Online4LDM/20144078'
    '/Test/Run_224/rawdata/*.h5')  # change run number!
with ProgressBar():
    tof = from_sequence(globbed).map(spectra).flatten().mean().compute()

# %%
plt.figure()
plt.plot(tof)
plt.grid()
plt.show()


# %%
def process(filename):
    with File(filename, 'r') as f:
        offset = 4000
        bins = [5905, 5945, 5990, 6055, 6105, 6185, 6315, 6400]  # double check bins!
        to = bins[-1]
        bp = f['Background_Period'].value
        bunches = f['bunches'][...]
        where = bunches%bp != 0
        tofs = f['digitizer/channel3'][where, 0:to].astype('int64')
        intensities = (
            f['photon_diagnostics/FEL01/I0_monitor/iom_sh_a_pc'][where]
                .astype('float64')
        )
        norm = average(tofs[:, 0:offset], 1)
        arrs = (norm[:, None]-tofs)/intensities[:, None]
        n, m = arrs.shape
        idx = arange(m)
        fmt = 'peak{}'.format
        for bunch, arr in zip(bunches[where], arrs):
            hist, *_ = histogram(idx, bins, weights=arr)
            yield {
                'bunch': bunch,
                **{fmt(i): h for i, h in enumerate(hist)}
            }

# %%
with ProgressBar():
    df = (
        from_sequence(globbed)
            .map(process)
            .flatten()
            .to_dataframe()
            .compute(get=multiprocessing_get)
            .set_index('bunch')
    )


# %%
def bins(idx):  # rescale region of interest!
    # return 100
    if idx == 0:
        return linspace(0, 50, 101)
    elif idx == 1:
        return linspace(0, 100, 101)
    elif idx == 2:
        return linspace(0, 150, 101)
    elif idx == 3:
        return linspace(0, 50, 101)
    elif idx == 4:
        return linspace(0, 100, 101)
    elif idx == 5:
        return linspace(100, 400, 101)
    elif idx == 6:
        return linspace(0, 100, 101)
    else:
        return 100


keys = sorted([k for k in df.keys() if k.startswith('peak')])
n = len(keys)
cov = df.cov()

plt.figure(figsize=(20, 20))
for i in range(n):
    for j in range(n):
        if i > j:
            continue
        plt.subplot(n, n, n*n-n*(j+1)+(i+1))
        xbins = bins(i)
        ybins = bins(j)
        plt.hist2d(df[keys[i]], df[keys[j]], bins=(xbins, ybins), cmap='Greys')
        # plt.axis('equal')
        # plt.gca().set_xticklabels([])
        # plt.gca().set_yticklabels([])
        plt.title('p{0} vs p{1} cov={2:1.0f}'.format(
            i, j, cov[keys[i]][keys[j]])
        )
plt.tight_layout()
plt.subplot(224)
plt.plot(tof)
plt.xlim(5500, 7500)
plt.show()
