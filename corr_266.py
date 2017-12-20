# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from glob import glob

from h5py import File
from numpy import average, histogram, arange, linspace, fromiter
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
from dask.multiprocessing import get as multiprocessing_get
import matplotlib.pyplot as plt


# %%
offset = 4000
fr, to = 5500, 6500
run = 266


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
it = iter(from_sequence(globbed).map(spectra).flatten())

plt.figure()
plt.plot(next(it))
plt.plot(tof)
plt.xlim(fr, to)
plt.minorticks_on()
plt.grid(which='both')
plt.show()


# %%
bins = (
    slice(5831, 5837),
    slice(5862, 5868),
    slice(5895, 5903),
    slice(5933, 5940),
    slice(5978, 5988),
    slice(6030, 6045),
    slice(6097, 6107)
)
plt.figure()
plt.plot(tof)
for b in bins:
    plt.axvspan(b.start, b.stop, 0, 1000, alpha=0.5)
plt.xlim(fr, to)
plt.grid()
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
summed0 = sum(df[key] for key in keys)
# summed1 = sum(df[key] for key in {'peak0', 'peak1'})
# summed2 = sum(df[key] for key in {'peak6', 'peak7'})

ilim = [11, 12]
tlim = [4000, 4800]
rlim = [0.5, 0.55]
ratios = df['peak2']/df['peak5']
good = (
    (df.index % bp != 0) &
    (ilim[0] < df['intensity']) & (df['intensity'] < ilim[1]) &
    (tlim[0] < summed0) & (summed0 < tlim[1]) &
    (rlim[0] < ratios) & (ratios < rlim[1])
#    (400 < summed1) & (summed1 < 500) &
#    (400 < summed2) & (summed2 < 500)
)
plt.figure()
# plt.hist2d(df[good]['peak2'], df[good]['peak5'], bins=(100, 100))
plt.hist(ratios, bins=linspace(0, 1, 101), histtype='step')
plt.show()



# %%
def pbins(idx):  # rescale region of interest!
    return 100
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


corr = df[good][keys].corr()

plt.figure(figsize=(16, 16))
for i in range(n):
    for j in range(n):
        if i >= j:
            continue
        plt.subplot(n, n, n * n - n * (j + 1) + (i + 1))
        xbins = pbins(i)
        ybins = pbins(j)
        plt.hist2d(df[good][keys[i]], df[good][keys[j]],
                   bins=(xbins, ybins), cmap='Greys')
        # plt.axis('equal')
        # plt.gca().set_xticklabels([])
        # plt.gca().set_yticklabels([])
        plt.title('p{0} vs p{1} corr={2:1.2f}'.format(
            i, j, corr[keys[i]][keys[j]])
        )
plt.tight_layout()

plt.subplot(4, 4, 11)
plt.hist(summed0, 100, histtype='step')
plt.axvspan(*tlim, 0, 1000, alpha=0.5)
plt.title('Total Electron Yield')
plt.grid()

plt.subplot(4, 4, 12)
plt.hist(df['intensity'], 100, histtype='step')
plt.axvspan(*ilim, 0, 1000, alpha=0.5)
plt.title('IOM')
plt.grid()

plt.subplot(4, 4, 15)
plt.hist(ratios, bins=linspace(0, 1, 101), histtype='step')
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

# %%
sum(df[key] for key in keys)