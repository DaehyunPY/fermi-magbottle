# -*- coding: utf-8 -*-
from glob import glob

from h5py import File
from numpy import average, arange, linspace, fromiter
from cytoolz import concat
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
from dask.multiprocessing import get as multiprocessing_get
import matplotlib.pyplot as plt


# %%
offset, fr, to = 4000, 5500, 6500
path = ('/home/ldm/ExperimentalData/Online4LDM/20144078'
        '/Test/Run_{:03d}/rawdata/*.h5').format
runs = {519}
globbed = sorted(concat(glob(path(r)) for r in runs))

with File(globbed[0]) as f:
    bp = f['Background_Period'].value
    bunches = f['bunches'][...]
    where = bunches % bp != 0
    tof = average(f['digitizer/channel1'][where, 0:to].astype('float64'), 0)
    del bunches, where

# %%
bins = [  # 515
    [5825, 5845],
    [5853, 5880],
    [5890, 5918],
    [5925, 5950],
    [5968, 5995],
    [6018, 6064],
    [6084, 6115],
    [6161, 6196],
    [6256, 6307],
    [6390, 6427]
]
plt.figure()
plt.plot(tof)
for b in bins:
    plt.axvspan(*b, 0, 1000, alpha=0.5)
plt.xlim(fr, to)
plt.minorticks_on()
plt.grid(which='both')
plt.show()


def process(filename):
    with File(filename, 'r') as f:
        bunches = f['bunches'][...]
        hor_spectra = (
            f['/photon_diagnostics/Spectrometer/hor_spectrum'][...]
                .astype('float64')
        )
        n = arange(hor_spectra.shape[1])
        hors = fromiter((average(n, weights=h) for h in hor_spectra),
                        'float')
        ver_spectra = (
            f['/photon_diagnostics/Spectrometer/vert_spectrum'][...]
                .astype('float64')
        )
        m = arange(ver_spectra.shape[1])
        vers = fromiter((average(m, weights=v) for v in ver_spectra),
                        'float')
        intensities = (
            f['photon_diagnostics/FEL01/I0_monitor/iom_sh_a_pc'][...]
                .astype('float64')
        )
        tofs = f['digitizer/channel1'][:, 0:to].astype('int64')
        arrs = average(tofs[:, 0:offset], 1)[:, None] - tofs
        fmt = 'peak{}'.format
        for bunch, hor, ver, inten, arr in zip(
            bunches, hors, vers, intensities, arrs
        ):
            yield {
                'bunch': bunch,
                'hor': hor,
                'ver': ver,
                'intensity': inten,
                **{fmt(i): arr[b0:b1].sum() for i, (b0, b1) in enumerate(bins)}
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
keys = sorted([k for k in df.keys() if k.startswith('peak')])
n = len(keys)
tyield = sum(df[key] for key in keys)

ilim = [0, 20]
tlim = [0, 50000]
rlim = [0, 1]
slim = [0, 30000]
ratios = df['peak2'] / df['peak5']
sums = df['peak2'] + df['peak5']
good = (
    (df.index % bp != 0) &
    (ilim[0] < df['intensity']) & (df['intensity'] < ilim[1]) &
    (tlim[0] < tyield) & (tyield < tlim[1]) &
    (rlim[0] < ratios) & (ratios < rlim[1]) &
    (slim[0] < sums) & (sums < slim[1])
)

corr = df[good][keys].corr()

plt.figure(figsize=(15, 15))
for i in range(n):
    for j in range(n):
        if i >= j:
            continue
        plt.subplot(n, n, n * n - n * (j + 1) + (i + 1))
        plt.hist2d(df[good][keys[i]], df[good][keys[j]],
                   bins=(100, 100), cmap='Greys')
        # plt.axis('equal')
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.title('{0}vs{1} corr{2:1.2f}'.format(
            i, j, corr[keys[i]][keys[j]])
        )
plt.tight_layout()

plt.subplot(4, 4, 11)
plt.hist(tyield[df.index % bp != 0], 100, histtype='step')
plt.axvspan(*tlim, 0, 1000, alpha=0.5)
plt.title('Total Electron Yield')
plt.grid()

plt.subplot(4, 4, 12)
plt.hist(df['intensity'], 100, histtype='step')
plt.axvspan(*ilim, 0, 1000, alpha=0.5)
plt.title('IOM')
plt.grid()

plt.subplot(4, 4, 14)
plt.hist(sums[df.index % bp != 0], bins=100, histtype='step')
plt.axvspan(*slim, 0, 1000, alpha=0.5)
plt.title('peak2 + peak5')
plt.grid()

plt.subplot(4, 4, 15)
plt.hist(ratios[(0 < ratios) & (ratios < 1)],
         bins=linspace(0, 1, 101), histtype='step')
plt.axvspan(*rlim, 0, 1000, alpha=0.5)
plt.title('peak2 / peak5')
plt.grid()

plt.subplot(4, 4, 16)
plt.plot(tof)
for i, (b0, b1) in enumerate(bins):
    plt.axvspan(b0, b1, 0, 1000, alpha=0.5)
    plt.text((b0 + b1) / 2, tof[b0:b1].min(), 'p{}'.format(i),
             horizontalalignment='center')
plt.xlim(fr, to)
plt.title('runs={}'.format(runs))
plt.grid()
# plt.tight_layout()
plt.show()
