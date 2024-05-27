#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 21 6:09 PM 2024
Created in PyCharm
Created as saclay_micromegas/banco_rate_resolution.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import skewnorm

from cosmic_test import *


def main():
    det_extents, det_verts = define_dets()
    ax, got_track = ray_plot_test(415)
    for det_i, det_i_verts in enumerate(det_verts):
        plot_urw_hit(None, det_i_verts, ax_in=ax)
    coincidence_dets = [[5, 6]]
    ref_rate = 3578 / (60 * 10)  # Hz
    # estimate_coninc_vs_sep(coincidence_dets)
    ref_rate_per_day = ref_rate * 60 * 60 * 24
    hits, coincident_hits = estimate_rate(det_extents, coincidence_dets)
    print(hits)
    print(coincident_hits)
    hits_per_day = np.array(hits) * ref_rate_per_day
    print(f'Reference Detector Hits per day: {ref_rate_per_day:.2f}')
    print('Hits per day: ', ', '.join([f'Det #{det_i}: {hit:.2f}' for det_i, hit in enumerate(hits_per_day)]))
    print('Coincident hits per day: ', ', '.join([f'Coincident #{det_i}: {hit:.2f}' for det_i, hit
                                                  in enumerate(coincident_hits * ref_rate_per_day)]))
    plt.show()
    print('donzo')


def define_dets(last_z=None):
    det_zs = [293.2 + 92 * 3 + 225, 293.2 + 92 * 2 + 225, 293.2 + 92 + 225, 293.2 + 225,
              293.2 + 225 - 92, 1100, 900]  # mm [Top, Bottom] to bottom of the board maybe raise. For data with stand
    if last_z is not None:
        det_zs[-1] = last_z
    det_x_centers = [0, 0, 0, 0, 0, 0, 0]  # mm [Top, Bottom] For data with stand
    det_y_centers = [0, 0, 0, 0, 0, 0, 0]  # mm [Top, Bottom] For data with stand
    det_x_lens = [130, 130, 130, 130, 130, 150, 150]  # mm [Top, Bottom] For data with stand
    det_y_lens = [130, 130, 130, 130, 130, 15, 15]  # mm [Top, Bottom] For data with stand
    det_z_lens = [2, 2, 2, 2, 2, 2, 2]  # mm [Top, Bottom] For data with stand
    det_extents, det_verts = [], []
    for x_cent, y_cent, z_cent, x_len, y_len, z_len \
            in zip(det_x_centers, det_y_centers, det_zs, det_x_lens, det_y_lens, det_z_lens):
        det_extents.append([(-x_len / 2 + x_cent, -y_len / 2 + y_cent, z_cent - z_len / 2),
                            (x_len / 2 + x_cent, y_len / 2 + y_cent, z_cent + z_len / 2)])
        det_verts.append(extent_to_vertices([
            (-x_len / 2 + x_cent, -y_len / 2 + y_cent, z_cent - z_len / 2),
            (x_len / 2 + x_cent, y_len / 2 + y_cent, z_cent + z_len / 2)]))

    return det_extents, det_verts


def estimate_rate(dets_extents, coincidence_dets=None):
    muon_r_skew_norm_pars = {'alpha': 2.23, 'xi': 101.4, 'omega': 212.6}
    det_top_extent = [(-250, -250, 1300), (250, 250, 1304)]
    det_bot_extent = [(-250, -250, 22), (250, 250, 26)]

    n_hits = 1000000
    top_hits_x = np.random.uniform(det_top_extent[0][0], det_top_extent[1][0], n_hits)
    top_hits_y = np.random.uniform(det_top_extent[0][1], det_top_extent[1][1], n_hits)
    hit_r = skewnorm.rvs(muon_r_skew_norm_pars['alpha'], muon_r_skew_norm_pars['xi'],
                         muon_r_skew_norm_pars['omega'], n_hits)
    hit_phi = np.random.uniform(0, 2 * np.pi, n_hits)
    bot_hits_x = hit_r * np.cos(hit_phi) + top_hits_x
    bot_hits_y = hit_r * np.sin(hit_phi) + top_hits_y

    z_top, z_bot = (det_top_extent[0][2] + det_top_extent[1][2]) / 2, (det_bot_extent[0][2] + det_bot_extent[1][2]) / 2
    hits = [0 for _ in dets_extents]
    coincident_hits = [0 for _ in coincidence_dets] if coincidence_dets is not None else None
    for x_t, y_t, x_b, y_b in zip(top_hits_x, top_hits_y, bot_hits_x, bot_hits_y):
        track_x = lambda z: x_t + (x_b - x_t) / (z_top - z_bot) * (z - z_bot)
        track_y = lambda z: y_t + (y_b - y_t) / (z_top - z_bot) * (z - z_bot)
        det_hit = [False for _ in dets_extents]
        for det_i, det_ext in enumerate(dets_extents):
            z_det = (det_ext[0][2] + det_ext[1][2]) / 2
            x_at_det_z = track_x(z_det)
            y_at_det_z = track_y(z_det)
            if point_in_extent(x_at_det_z, y_at_det_z, z_det, det_ext):
                hits[det_i] += 1
                det_hit[det_i] = True
        if coincidence_dets is not None:
            for i, coincidence_group in enumerate(coincidence_dets):
                if all(det_hit[j] for j in coincidence_group):
                    coincident_hits[i] += 1
    # print(hits)
    # print(coincident_hits)

    hits = np.array(hits) / n_hits
    coincident_hits = np.array(coincident_hits) / n_hits if coincidence_dets is not None else None

    # print(hits)
    # print(coincident_hits)

    return hits, coincident_hits


def estimate_banco_res_rate_coverage(dets_extents, coincidence_dets=None):
    muon_r_skew_norm_pars = {'alpha': 2.23, 'xi': 101.4, 'omega': 212.6}
    det_top_extent = [(-250, -250, 1300), (250, 250, 1304)]
    det_bot_extent = [(-250, -250, 22), (250, 250, 26)]

    n_hits = 1000000
    top_hits_x = np.random.uniform(det_top_extent[0][0], det_top_extent[1][0], n_hits)
    top_hits_y = np.random.uniform(det_top_extent[0][1], det_top_extent[1][1], n_hits)
    hit_r = skewnorm.rvs(muon_r_skew_norm_pars['alpha'], muon_r_skew_norm_pars['xi'],
    muon_r_skew_norm_pars['omega'], n_hits)
    hit_phi = np.random.uniform(0, 2 * np.pi, n_hits)
    bot_hits_x = hit_r * np.cos(hit_phi) + top_hits_x
    bot_hits_y = hit_r * np.sin(hit_phi) + top_hits_y

    z_top, z_bot = (det_top_extent[0][2] + det_top_extent[1][2]) / 2, (det_bot_extent[0][2] + det_bot_extent[1][2]) / 2
    hits = [0 for _ in dets_extents]
    coincident_hits = [0 for _ in coincidence_dets] if coincidence_dets is not None else None
    for x_t, y_t, x_b, y_b in zip(top_hits_x, top_hits_y, bot_hits_x, bot_hits_y):
        track_x = lambda z: x_t + (x_b - x_t) / (z_top - z_bot) * (z - z_bot)
    track_y = lambda z: y_t + (y_b - y_t) / (z_top - z_bot) * (z - z_bot)
    det_hit = [False for _ in dets_extents]
    for det_i, det_ext in enumerate(dets_extents):
        z_det = (det_ext[0][2] + det_ext[1][2]) / 2
    x_at_det_z = track_x(z_det)
    y_at_det_z = track_y(z_det)
    if point_in_extent(x_at_det_z, y_at_det_z, z_det, det_ext):
        hits[det_i] += 1
    det_hit[det_i] = True
    if coincidence_dets is not None:
        for
    i, coincidence_group in enumerate(coincidence_dets):
    if all(det_hit[j] for j in coincidence_group):
        coincident_hits[i] += 1
    # print(hits)
    # print(coincident_hits)

    hits = np.array(hits) / n_hits
    coincident_hits = np.array(coincident_hits) / n_hits if coincidence_dets is not None else None

    # print(hits)
    # print(coincident_hits)


return hits, coincident_hits


def estimate_coninc_vs_sep(coincidence_dets):
    zs = np.linspace(50, 1050, 41)
    coinc_rates, indiv_rates = [], []
    for z in zs:
        print(f'\nz: {z}')
        det_extents, det_verts = define_dets(z)
        hits, coincident_hits = estimate_rate(det_extents, coincidence_dets)
        indiv_rates.append((hits[-1] + hits[-2]) / 2)
        coinc_rates.append(coincident_hits[0])
    coinc_frac = np.array(coinc_rates) / np.array(indiv_rates)
    fig, ax = plt.subplots()
    ax.plot(1100 - zs, coinc_frac)
    ax.set_xlabel('Distance between banco arms (mm)')
    ax.set_ylabel('Fraction of single arm hits that are coincident')
    ax.set_ylim(bottom=0)
    ax.grid()
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))


def point_in_extent(x, y, z, extent):
    if extent[0][0] <= x <= extent[1][0] and extent[0][1] <= y <= extent[1][1] and extent[0][2] <= z <= extent[1][2]:
        return True
    return False


if __name__ == '__main__':
    main()
