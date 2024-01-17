#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on January 17 3:46 PM 2024
Created in PyCharm
Created as saclay_micromegas/common_noise_sample_maxima.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    n_events, n_samples = 10, 32
    np.random.seed(0)
    gauss = np.random.normal(300, 10, (n_events, n_samples))
    maxima = np.max(gauss, axis=1)
    maxima_sample_nums = np.argmax(gauss, axis=1)

    fig, ax = plt.subplots()
    plt.title('Gaussian Noise vs Time')
    plt.axhline(300, color='gray')
    plt.xlabel('Sample Number (Time)')
    plt.ylabel('ADC Counts')
    for i, (event, max_val, max_sample_num) in enumerate(zip(gauss, maxima, maxima_sample_nums)):
        plt.axvline(i * n_samples, color='black', ls='--')
        plt.plot(np.arange(i * n_samples, (i + 1) * n_samples), event)
        plt.plot(max_sample_num + i * n_samples, max_val, marker='x', color='red', ls='None')
    plt.axvline(n_samples * n_events, color='black', ls='--')
    fig.tight_layout()

    fig, ax = plt.subplots()
    plt.title('Gaussian Noise Max Sample vs Event Number')
    plt.axhline(300, color='gray')
    plt.plot(maxima, marker='o', ls='None')
    plt.xlabel('Event Number')
    plt.ylabel('Max Sample ADC Counts')
    fig.tight_layout()
    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
