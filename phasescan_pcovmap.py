# -*- coding: utf-8 -*-

from glob import glob

from cytoolz import concat
from pandas import DataFrame
from h5py import File
from numpy import average, append, ceil
from dask.array import from_delayed, concatenate, cov
from dask.delayed import delayed
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt


# %%
def info(filename):
    with File(filename, 'r') as f:
        phase = round(
            float(f['photon_source/FEL01/PhaseShifter4/DeltaPhase'].value), 2
        )
        bp = f['Background_Period'].value
        bunches = f['bunches'][...]
        where = bunches % bp != 0
        return {
            'filename': filename,
            'phase': phase,
            'size': where.sum()
        }


runs = {442, 443}
path = ('/home/ldm/ExperimentalData/Online4LDM/20144078'
        '/Test/Run_{:03d}/rawdata/*.h5'.format)
globbed = concat(glob(path(r)) for r in runs)
infos = DataFrame(list(info(f) for f in globbed))


# %%
offset = 4000
fr, to = 5750, 6500


@delayed
def process(filename):
    with File(filename, 'r') as f:
        bp = f['Background_Period'].value
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
def target(query):
    arr = concatenate([
        from_delayed(process(f), [s, to - fr + 1], 'float64')
        for _, f, s in query[['filename', 'size']].itertuples()
    ])

    with ProgressBar():
        mat = cov(arr.T).compute()
    mat_cov = mat[1:, 1:]
    mat_icov = mat[1:, 0][:, None] @ mat[0, 1:][None, :] / mat[0, 0]
    mat_pcov = mat_cov - mat_icov
    return mat_pcov

phases = infos['phase'].unique()
targets = {phase: target(infos[infos['phase'] == phase]) for phase in phases}


# %%
def scale(arr):
    m = min(abs(arr.min()), abs(arr.max())) * 0.5
    return -m, m


n = int(ceil(len(targets) ** 0.5))
plt.figure(figsize=(12, 12))
for i, (k, t) in enumerate(targets.items()):
    plt.subplot(n, n, i + 1)
    plt.pcolormesh(t, cmap='RdBu')
    plt.clim(*scale(t))
    plt.title('ph={:1.2f}'.format(k))
plt.tight_layout()
plt.show()
