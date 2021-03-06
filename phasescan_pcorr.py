# -*- coding: utf-8 -*-
from glob import glob

from h5py import File
from numpy import average
from cytoolz import concat
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
from dask.multiprocessing import get as multiprocessing_get
import matplotlib.pyplot as plt


# %%
offset, fr, to = 4000, 5700, 7000
path = ('/home/ldm/ExperimentalData/Online4LDM/20144078'
        '/Test/Run_{:03d}/rawdata/*.h5').format
runs = {541}
globbed = sorted(concat(glob(path(r)) for r in runs))

with File(globbed[0]) as f:
    bp = f['Background_Period'].value
    bunches = f['bunches'][...]
    where = bunches % bp != 0
    tof = average(f['digitizer/channel1'][where, 0:to].astype('float64'), 0)
    del bunches, where

# %%
bins = [  # 515 541
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
        phase = round(
            float(f['photon_source/FEL01/PhaseShifter4/DeltaPhase'].value), 2
        )
        intensities = (
            f['photon_diagnostics/FEL01/I0_monitor/iom_sh_a_pc'][...]
                .astype('float64')
        )
        tofs = f['digitizer/channel1'][:, 0:to].astype('float64')
        arrs = average(tofs[:, 0:offset], 1)[:,  None] - tofs
        fmt = 'peak{}'.format
        for bunch, inten, arr in zip(bunches, intensities, arrs):
            yield {
                'bunch': bunch,
                'intensity': inten,
                'phase': phase,
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
good = (
    (df.index % bp != 0)
)
groupped = df[good].dropna().groupby('phase')[['intensity', *keys]]
counts = groupped.count()
corr = groupped.corr()

plt.figure(figsize=(15, 15))
for ix, kx in enumerate(keys):
    for iy, ky in enumerate(keys):
        if ix >= iy:
            continue
        plt.subplot(n, n, n * n - n * (iy + 1) + (ix + 1))
        pcorr = ((corr[kx].loc[:, ky] -
                  corr[kx].loc[:, 'intensity'] *
                  corr[ky].loc[:, 'intensity']) /
                 (1 - corr[kx].loc[:, 'intensity'] ** 2) ** 0.5 /
                 (1 - corr[kx].loc[:, 'intensity'] ** 2) ** 0.5)
        plt.plot(pcorr, '.-')
        plt.ylim(-1, 1)
        # plt.gca().set_yticks([0])
        plt.twinx()
        plt.plot(counts[kx], 'k.-', alpha=0.1)
        plt.gca().set_yticks([])
        plt.ylim(0, None)
        plt.title('{0}, {1}'.format(ix, iy))
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.plot(tof)
for i, (b0, b1) in enumerate(bins):
    plt.axvspan(b0, b1, 0, 1000, alpha=0.5)
    plt.text((b0 + b1) / 2, tof[b0:b1].min(), 'p{}'.format(i),
             horizontalalignment='center')
plt.title('runs={}, pcorr'.format(runs))
plt.xlim(fr, to)
plt.grid()
plt.show()
