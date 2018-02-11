from itertools import chain
from glob import iglob

from numpy import arange, around, sum
import numpy as np
from h5py import File
from dask import compute
from dask.multiprocessing import get as  multiprocessing_get
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from cytoolz import memoize


# %%
fr, to = 5060, 10000
t = arange(fr, to)
e = np.power(3.4662936e3 / (t - 5059), 2) + 270;

def spectra(filename):
    try:
        with File(filename, 'r') as f:
            bp = f['Background_Period'][...]
            # bp = 3
            bunches = f['bunches'][...]
            mods = bunches % bp == 0
            tofs = f['digitizer/channel3'][:, fr:to].astype('float64')
            delays = around(f['user_laser/delay_line/position'][...], 2)
            for mod, tof, delay in zip(mods, tofs, delays):
                if delay != 0:
                    yield {'bg': mod, 'tof': tof, 'delay': delay}
    except Exception as e:
        print('Error at {}:'.format(filename))
        print(e)
        yield from ()


# %%
@memoize
def average_tof_bkg(run, *other_runs):
    runs = run, *other_runs
    globbed = tuple(chain(*(iglob(
        '/home/ldm/.gvfs/store on online4ldm.esce/20149046'
        '/Test/Run_{:03d}/rawdata/*.h5'.format(r)) for r in runs)))
    df = from_sequence(globbed).map(spectra).flatten().to_dataframe()
    with ProgressBar():
        sig_g = df[~df['bg']].groupby('delay')
        bkg_g = df[df['bg']].groupby('delay')
        sig, n, bkg, m = compute(
            sig_g['tof'].apply(sum, meta=('tof', 'object')),
            sig_g['delay'].count(),
            bkg_g['tof'].apply(sum, meta=('tof', 'object')),
            bkg_g['delay'].count(),
            get=multiprocessing_get
        )
        return bkg / m - sig / n


# %%
runs = [165, 166, 167,168,170]
groupped = average_tof_bkg(*runs)

#%% 
plt.figure(figsize=(5, 4))
for delay, tof in groupped.iteritems():
    plt.plot(t, tof, label=delay)
plt.xlabel('tof (ns)')
plt.ylabel('yield')
plt.xlim(5800, 7000)
plt.ylim(-2, 22)
plt.minorticks_on()
plt.grid(which='both')
plt.legend()
plt.tight_layout()
plt.show()

# %%
hv = 1240 / (266 / 13 / 5)
plt.figure(figsize=(10, 8))
for i, (delay, tof) in enumerate(groupped.iteritems()):
    plt.subplot(4, 4, i + 1)
    plt.plot(e, tof/np.sqrt(e), label=delay)
    plt.title(delay)
    plt.xlabel('tof (ns)')
    plt.ylabel('yield')
    plt.xlim(270, 285)
    #plt.ylim(-2, 22)
    plt.minorticks_on()
    plt.grid(which='both')
plt.tight_layout()
plt.show()
