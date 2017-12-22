# -*- coding: utf-8 -*-
from glob import glob

from h5py import File
from numpy import average, ceil, linspace
from cytoolz import concat
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
from dask.multiprocessing import get as multiprocessing_get
import matplotlib.pyplot as plt


# %%
offset, fr, to = 4000, 5700, 7000
path = ('/home/ldm/ExperimentalData/Online4LDM/20144078'
        '/Test/Run_{:03d}/rawdata/*.h5').format
runs = set(range(531, 534))
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
        run = int(filename.split('/')[-1].split('_')[1])
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
                'run': run,
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
good = (
    (df.index % bp != 0)
)
m = int(ceil((len(runs) + 2) ** 0.5))
n = m if m * (m - 1) < len(runs) + 2 else m - 1

# x, y = 3, 4
x, y = 4, 6
kx, ky = (keys[i] for i in (x, y))
xbin, ybin = (linspace(*df[good][k].agg(['min', 'max']), 101)
              for k in (kx, ky))
ordered = runs

plt.figure(figsize=(15, 10))
for i, r in enumerate(ordered):
    plt.subplot(n, m, (i + 1))
    where = good & (df['run'] == r)
    hist, *_ = plt.hist2d(df[where][kx], df[where][ky], bins=(xbin, ybin))
    # plt.gca().set_xticks([])
    # plt.gca().set_yticks([])
    corr = df[where][['intensity', kx, ky]].corr()
    pcorr = ((corr[kx][ky] -
              corr[kx]['intensity'] *
              corr[ky]['intensity']) /
             (1 - corr[kx]['intensity'] ** 2) ** 0.5 /
             (1 - corr[kx]['intensity'] ** 2) ** 0.5)
    cx, cy = (average((xbin[1:] + xbin[:-1]) / 2, weights=hist.sum(1)),
              average((ybin[1:] + ybin[:-1]) / 2, weights=hist.sum(0)))
    plt.plot(cx, cy, 'wo')
    plt.title('run={}, pcorr={:1.2f}, c={:1.0f}'.format(
        r, pcorr, (cx ** 2 + cy ** 2) ** 0.5
    ))
plt.tight_layout()

plt.subplot2grid((n, m), (n - 1, m - 2), colspan=2)
plt.plot(tof)
for i in [x, y]:
    b0, b1 = bins[i]
    plt.axvspan(b0, b1, 0, 1000, alpha=0.5)
    plt.text((b0 + b1) / 2, tof[b0:b1].min(), 'p{}'.format(i),
             horizontalalignment='center')
plt.title('runs={}'.format(runs))
plt.xlim(fr, to)
plt.grid()
plt.show()
