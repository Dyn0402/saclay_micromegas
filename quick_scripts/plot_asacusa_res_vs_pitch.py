#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 27 2:38 AM 2024
Created in PyCharm
Created as saclay_micromegas/plot_asacusa_res_vs_pitch.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    pitches = [0.8, 1, 1.5, 2]
    asa1_res = [672.4886616144543, 711.5567440478658, 848.7590942075378, 1079.048655066119]
    asa2_res = [625.3079241461761, 738.3549282018777, 850.6991242501664, 1029.417310909429]
    asa1_res_x = [471, 477, 586, 765]
    asa1_res_y = [480, 528, 614, 761]
    asa2_res_x = [409, 512, 595, 790]
    asa2_res_y = [473, 532, 608, 660]
    fig, ax = plt.subplots()
    ax.plot(pitches, asa1_res, marker='o', label='Asacusa Strip 1')
    ax.plot(pitches, asa2_res, marker='o', label='Asacusa Strip 2')
    ax.set_xlabel('Pitch (mm)')
    ax.set_ylabel('Resolution (μm)')
    ax.grid()
    ax.legend()
    fig.tight_layout()

    fig2, ax2 = plt.subplots()
    ax2.plot(pitches, asa1_res_x, marker='o', ls='none', ms=10, color='blue', alpha=0.8, label='Asacusa Strip 1 X')
    ax2.plot(pitches, asa1_res_y, marker='s', ls='none', ms=10, color='blue', alpha=0.8, label='Asacusa Strip 1 Y')
    ax2.plot(pitches, asa2_res_x, marker='o', ls='none', ms=10, color='orange', alpha=0.8, label='Asacusa Strip 2 X')
    ax2.plot(pitches, asa2_res_y, marker='s', ls='none', ms=10, color='orange', alpha=0.8, label='Asacusa Strip 2 Y')
    ax2.set_xlabel('Pitch (mm)')
    ax2.set_ylabel('Resolution (μm)')
    ax2.grid()
    ax2.legend()
    fig2.tight_layout()

    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
