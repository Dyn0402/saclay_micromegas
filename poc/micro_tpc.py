#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 04 7:06 PM 2025
Created in PyCharm
Created as saclay_micromegas/micro_tpc.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    drift_gap = 30  # mm
    strip_pitch = 0.8 * 2  # mm
    angles = np.linspace(0, np.pi / 4, 1000)

    strips_hit = drift_gap * np.tan(angles) / strip_pitch

    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(angles), strips_hit, linestyle='-')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Number of strips hit')
    ax.set_title('Number of strips hit vs angle')
    ax.grid()
    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()
