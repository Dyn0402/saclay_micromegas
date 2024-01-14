#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on January 13 5:08 PM 2024
Created in PyCharm
Created as saclay_micromegas/eyeball_IV_fits.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf

from Measure import Measure


def exp_func(x, a, b):
    return a * np.exp(b * x)


def main():
    v_urw = np.array([300, 310, 320, 330, 340, 350, 360, 370, 380])
    mu_urw = np.array([120, 175, 240, 315, 465, 685, 930, 1270, 1800])
    mu_urw_err = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5])
    p0 = [1, 0.000034]
    popt, pcov = cf(exp_func, v_urw, mu_urw, p0=p0, sigma=mu_urw_err)
    perr = np.sqrt(np.diag(pcov))
    fit_pars = [Measure(val, err) for val, err in zip(popt, perr)]
    print(fit_pars)
    plt.plot(v_urw, mu_urw, marker='o', linestyle='None', label='URW')
    plt.plot(v_urw, exp_func(v_urw, *p0), label='Guess', color='gray')
    plt.plot(v_urw, exp_func(v_urw, *popt), label='Fit', color='red')
    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
