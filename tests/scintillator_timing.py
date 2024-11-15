#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 15 11:13 2024
Created in PyCharm
Created as saclay_micromegas/scintillator_timing

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
from Measure import Measure


def main():
    c = 12.0  # cm/ns  speed of light in scintillator
    length = 60  # cm  length of scintillator

    # Randomly generate positions for particles along length of scintillator and histogram the times it takes for light
    # to travel from the start of the scintillator to the particle

    num_particles = 1000000
    positions = np.random.rand(num_particles) * length
    times = positions / c
    plt.hist(times, bins=100, range=(0, 5), histtype='step')
    plt.xlabel('Time (ns)')
    plt.ylabel('Counts')
    plt.title('Time for light to travel from start of scintillator to particle')

    sigma_angle = np.deg2rad(10.0)  # degrees  width of track angular distribution
    z_separation = 2 * 100  # cm  separation between two scintillators

    # Randomly generate positions and angles for particles along the length of the top scintillator. Calculate the
    # position of the particle when it reaches the bottom scintillator. If both are hit, calculate mean of the times it
    # takes for light to travel from the start of the top scintillator to 0 and from the start of the bottom scintillator
    # to 0.

    num_particles = 1000000
    positions_top = np.random.rand(num_particles) * length
    angles = np.random.normal(0, sigma_angle, num_particles)
    positions_bottom = positions_top + z_separation * np.tan(angles)

    # Filter out particles where positions_bottom is less than 0 or greater than length
    mask = (positions_bottom >= 0) & (positions_bottom <= length)

    times_top = positions_top[mask] / c
    times_bottom = positions_bottom[mask] / c

    time_diff = times_top - times_bottom

    fig, ax = plt.subplots()
    counts, bins = np.histogram(time_diff, bins=100, range=(-5, 5))
    bin_centers = (bins[1:] + bins[:-1]) / 2
    ax.errorbar(bin_centers, counts, yerr=np.sqrt(counts), fmt='o', label='Data')
    popt, pcov = cf(gaus, bin_centers, counts)
    fit_meases = [Measure(val, err) for val, err in zip(popt, np.sqrt(np.diag(pcov)))]
    fit_str = f'Fit:\nA = {fit_meases[0]}\nμ = {fit_meases[1]}\nσ = {fit_meases[2]}'
    ax.plot(bin_centers, gaus(bin_centers, *popt), label='Fit')
    ax.axhline(0, color='black')
    ax.set_xlabel('Time Difference (ns)')
    ax.set_ylabel('Counts')
    ax.set_title('Time Difference between top and bottom scintillator')
    ax.annotate(fit_str, xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top',
                bbox=dict(boxstyle='round', fc='w'))
    ax.legend(loc='upper right')

    plt.show()

    print('donzo')


def gaus(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


if __name__ == '__main__':
    main()
