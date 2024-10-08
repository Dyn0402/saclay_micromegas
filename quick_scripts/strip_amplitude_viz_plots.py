#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 08 4:26 PM 2024
Created in PyCharm
Created as saclay_micromegas/strip_amplitude_viz_plots.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Define the number of strips in the x and y directions
    num_strips = 7
    y_scale = 1000
    x_scale = 900

    # Create an array for strip numbers (1 to 7)
    strips = np.arange(1, num_strips + 1, 1)

    # Manually specified amplitudes for bottom (y) strips (slightly asymmetric)
    amplitude_y = np.array([0.02, 0.01, 0.5, 0.9, 0.55, 0.03, 0.015]) * y_scale

    # Manually specified amplitudes for top (x) strips (different asymmetry)
    amplitude_x = np.array([0.03, 0.01, 0.45, 1.0, 0.6, 0.02, 0.01]) * x_scale

    # Plot for bottom (y) strips (bar plot)
    plt.figure(figsize=(8, 3))
    plt.bar(strips, amplitude_y, color='blue', alpha=0.7)
    plt.axhline(y=y_scale * 0.1, color='blue', linestyle='--')
    plt.xlabel('Strip Number')
    plt.ylabel('Amplitude')
    plt.ylim(bottom=0)
    plt.xticks(strips)
    plt.tight_layout()

    # Plot for top (x) strips (bar plot)
    plt.figure(figsize=(8, 3))
    plt.bar(strips, amplitude_x, color='red', alpha=0.7)
    plt.axhline(y=x_scale * 0.1, color='red', linestyle='--')
    plt.xlabel('Strip Number')
    plt.ylabel('Amplitude')
    plt.ylim(bottom=0)
    plt.xticks(strips)
    plt.tight_layout()

    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()
