#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 11 08:31 2024
Created in PyCharm
Created as saclay_micromegas/waveform_func_test

@author: Dylan Neff, dn277127
"""

from DreamData import waveform_func, waveform_func_reparam

import numpy as np
import matplotlib.pyplot as plt


def main():
    t = np.linspace(0, 32, 1000)
    t_maxes = [5, 10, 15, 20, 25]
    # q = 2. / 3
    q = 1
    a = 1
    t_shift = 2
    fig, ax = plt.subplots()
    fig_reparam, ax_reparam = plt.subplots()
    for t_max in t_maxes:
        freq = 2 / t_max
        ax.plot(t, waveform_func(t, a, freq, q), label=f't_max={t_max:.2f}')
        ax_reparam.plot(t, waveform_func_reparam(t, a, t_max, t_shift, q), label=f't_max={t_max:.2f}')
        print(f'Val at max time: {waveform_func(t_max, a, freq, q)}')
    ax.legend()
    ax.grid()
    ax_reparam.legend()
    ax_reparam.grid()
    fig.tight_layout()
    fig_reparam.tight_layout()
    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
