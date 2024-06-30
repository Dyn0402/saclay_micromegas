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
    # ray_dir = 'C:/Users/Dylan/Desktop/banco_test3/'
    ray_dir = 'F:/Saclay/banco_data/banco_stats2/'
    m3_ref_tracking = M3RefTracking(ray_dir, single_track=False)

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
    # single_track_test(ray_dir)
    plot_chi2_distributions(m3_ref_tracking.ray_data)
    print('donzo')


def single_track_test(ray_dir):
    m3_ref_tracking = M3RefTracking(ray_dir, single_track=False)
    chi2_x, chi2_y = m3_ref_tracking.ray_data['Chi2X'], m3_ref_tracking.ray_data['Chi2Y']
    num_tracks_per_event = ak.num(m3_ref_tracking.ray_data['X_Up'], axis=1)
    num_events_with_track = ak.sum(num_tracks_per_event > 0)
    og_total_events = len(m3_ref_tracking.ray_data['X_Up'])
    print(f'Original total number of events: {og_total_events}')
    print(f'Original number of events with at least one track: {num_events_with_track}')
    num_good_tracks = ak.sum((chi2_x < m3_ref_tracking.chi2_cut) & (chi2_y < m3_ref_tracking.chi2_cut), axis=1)
    num_events_with_good_track = ak.sum(num_good_tracks > 0)
    print(f'Original number of events with at least one good track: {num_events_with_good_track}')
    m3_ref_tracking.cut_on_det_size()
    chi2_x, chi2_y = m3_ref_tracking.ray_data['Chi2X'], m3_ref_tracking.ray_data['Chi2Y']
    num_good_tracks = ak.sum((chi2_x < m3_ref_tracking.chi2_cut) & (chi2_y < m3_ref_tracking.chi2_cut), axis=1)
    num_events_with_good_track_chi = ak.sum(num_good_tracks > 0)
    print(f'Number of events with at least one good track after cut on detector size: {num_events_with_good_track_chi}')
    num_events_with_single_good_track = ak.sum(num_good_tracks == 1)
    print(f'Number of events with exactly one good track after cut on detector size: {num_events_with_single_good_track}')

    fig, ax = plt.subplots()
    # Make horizontal bar graph of the number of events after each of these steps. Write the numbers in bars
    labels = ['Original', 'Original with Tracks', 'After Chi2 Cut', 'After Detector Size Cut', 'After Single Track Cut']
    num_events = [og_total_events, num_events_with_track, num_events_with_good_track, num_events_with_good_track_chi,
                  num_events_with_single_good_track]
    ax.barh(labels, num_events, height=1)
    for i, num in enumerate(num_events):
        ax.text(num / 2, i, str(num), ha='center', va='center', color='w')
    ax.set_title('Number of Events After Each Cut')
    fig.tight_layout()
    plt.show()

    m3_ref_tracking_single = M3RefTracking(ray_dir, single_track=True)
    print(m3_ref_tracking_single.ray_data['X_Up'])
    print(m3_ref_tracking.ray_data['X_Up'])
    print(m3_ref_tracking_single.ray_data)
    num_events_with_good_track = len(m3_ref_tracking_single.ray_data['X_Up'])
    print(f'Number of events with at least one good track after single track cut: {num_events_with_good_track}')


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
        ax.hist(chi2_x_ray, bins=np.arange(-1, max(chi2_x_ray) + 1, 0.5), alpha=0.5, label='Chi2X')
        ax.hist(chi2_y_ray, bins=np.arange(-1, max(chi2_y_ray) + 1, 0.5), alpha=0.5, label='Chi2Y')
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
