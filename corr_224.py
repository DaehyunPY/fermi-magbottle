# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from glob import glob

from h5py import File
from numpy import average, arange, linspace, fromiter
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
from dask.multiprocessing import get as multiprocessing_get
import matplotlib.pyplot as plt


# %%
offset = 4000
fr, to = 5500, 6500
run = 224


def spectra(filename):
    with File(filename, 'r') as f:
        bp = f['Background_Period'].value
        bunches = f['bunches'][...]
        where = bunches % bp != 0
        yield from f['digitizer/channel3'][where, 0:to].astype('int64')


globbed = glob(
    # '/Volumes/store/20144078'
    '/home/ldm/ExperimentalData/Online4LDM/20144078'
    '/Test/Run_{:3d}/rawdata/*.h5'.format(run))  # change run number!
with File(globbed[0]) as f:
    bp = f['Background_Period'].value
with ProgressBar():
    tof = from_sequence(globbed).map(spectra).flatten().mean().compute()

# %%
bins = (
    slice(5912, 5920),
    slice(5952, 5965),
    slice(6002, 6015),
    slice(6062, 6073),
    slice(6130, 6145),
    slice(6210, 6240),
    slice(6325, 6340)
)
plt.figure()
plt.plot(tof)
for b in bins:
    plt.axvspan(b.start, b.stop, 0, 1000, alpha=0.5)
plt.xlim(fr, to)
plt.minorticks_on()
plt.grid(which='both')
plt.show()


# %%
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
        tofs = f['digitizer/channel3'][:, 0:to].astype('int64')
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
                **{fmt(i): arr[b].sum() for i, b in enumerate(bins)}
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
tlim = [0, 6000]
rlim = [0, 1]
slim = [0, 5000]
ratios = df['peak2'] / df['peak5']
sums = df['peak2'] + df['peak5']
good = (
    (df.index % bp != 0) &
    (ilim[0] < df['intensity']) & (df['intensity'] < ilim[1]) &
    (tlim[0] < tyield) & (tyield < tlim[1]) &
    (rlim[0] < ratios) & (ratios < rlim[1]) &
    (slim[0] < sums) & (sums < slim[1])
)



# %%
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
        # plt.gca().set_xticklabels([])
        # plt.gca().set_yticklabels([])
        plt.title('p{0} vs p{1} corr={2:1.2f}'.format(
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
for i, b in enumerate(bins):
    plt.axvspan(b.start, b.stop, 0, 1000, alpha=0.5)
    plt.text((b.start + b.stop) / 2, tof[b].min(), 'p{}'.format(i),
             horizontalalignment='center')
plt.xlim(fr, to)
plt.title('run={}'.format(run))
plt.grid()
plt.tight_layout()
plt.show()