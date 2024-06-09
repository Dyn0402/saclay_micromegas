#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 07 7:41 PM 2024
Created in PyCharm
Created as saclay_micromegas/m3_track_quality.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt

import awkward as ak

from M3RefTracking import M3RefTracking


def main():
    ray_dir = 'C:/Users/Dylan/Desktop/banco_test3/'
    m3_ref_tracking = M3RefTracking(ray_dir)

    # numbers = ak.Array(
    #     [
    #         [0, 1, 2, 3],
    #         [4, 5, 6],
    #         [8, 9, 10, 11, 12],
    #     ]
    # )
    # print(type(numbers))
    #
    # rd = m3_ref_tracking.ray_data[:50]
    # print(type(rd))
    # print(m3_ref_tracking.ray_data)
    # print(rd['Chi2X'] < 10)
    # print(rd['Chi2X'][rd['Chi2X'] < 10])
    # input()
    plot_chi2_distributions(m3_ref_tracking.ray_data)
    print('donzo')


def plot_chi2_distributions(ray_data):
    # ray_data = ak.to_numpy(ray_data)
    chi2_x, chi2_y = ak.to_list(ray_data['Chi2X']), ak.to_list(ray_data['Chi2Y'])
    print(chi2_x)
    print([len(x) for x in chi2_x])
    ray_n = np.array(ray_data['rayN'])
    print(ray_n)

    fig_rayn, ax_rayn = plt.subplots()
    ax_rayn.hist(ray_n, bins=np.arange(-0.5, max(ray_n) + 1.5, 1))
    ax_rayn.set_title('Ray Number Distribution')

    chi2_x, chi2_y = np.array(chi2_x, dtype=list), np.array(chi2_y, dtype=list)
    for ray_num in range(1, max(ray_n) + 1):
        mask = ray_n == ray_num
        chi2_x_ray, chi2_y_ray = chi2_x[mask], chi2_y[mask]
        chi2_x_ray, chi2_y_ray = np.concatenate(chi2_x_ray), np.concatenate(chi2_y_ray)
        print(chi2_x_ray)
        # print([len(x) for x in chi2_x_ray])
        fig, ax = plt.subplots()
        ax.hist(chi2_x_ray, bins=100, alpha=0.5, label='Chi2X')
        ax.hist(chi2_y_ray, bins=100, alpha=0.5, label='Chi2Y')
        ax.set_title(f'Ray {ray_num}')
        ax.legend()
    plt.show()
    # chi2_x, chi2_y = chi2_x[~np.isnan(chi2_x)], chi2_y[~np.isnan(chi2_y)]
    # chi2_x, chi2_y = chi2_x[chi2_x < 100], chi2_y[chi2_y < 100]
    # chi2_x, chi2_y = chi2_x[chi2_x > 0], chi2_y[chi2_y > 0]
    # chi2_x, chi2_y = chi2_x[chi2_x < 10], chi2_y[chi2_y < 10]
    # chi2_x, chi2_y = chi2_x[chi2_x > 0], chi2_y[chi2_y > 0]
    # plt.hist(chi2_x, bins=100, alpha=0.5, label='Chi2X')
    # plt.hist(chi2_y, bins=100, alpha=0.5, label='Chi2Y')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
