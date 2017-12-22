#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:00:42 2017

@author: ldm
"""
from scipy.optimize import curve_fit
from numpy import sin


# %%
def func(x, amp, f, x0, y0):
    return amp*sin(f*(x-x0))+y0



kx, ky = 'peak4', 'peak6'
pcorr = ((corr[kx].loc[:, ky] -
          corr[kx].loc[:, 'intensity'] *
          corr[ky].loc[:, 'intensity']) /
         (1 - corr[kx].loc[:, 'intensity'] ** 2) ** 0.5 /
         (1 - corr[kx].loc[:, 'intensity'] ** 2) ** 0.5)
phases = pcorr.index

opt, _ = curve_fit(func, phases, pcorr)
print(opt)

plt.figure()
plt.plot(pcorr, '.-')
plt.plot(phases, func(phases, *opt), '.-')
plt.show()

