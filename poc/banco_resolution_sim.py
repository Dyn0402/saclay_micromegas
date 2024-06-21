#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 20 19:17 2024
Created in PyCharm
Created as saclay_micromegas/banco_resolution_sim

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    pix_size = 30  # microns
    n_pix = 10
    track_gaus_sigma = 68  # microns
    hit_threshold = 0.9
    n_events = 2000
    plot_events, plot_spatial_hit_dist = False, True
    # gaus_plot_test()
    x_pix_centers = np.arange(pix_size / 2, (0.5 + n_pix) * pix_size, pix_size)
    y_pix_centers = np.arange(pix_size / 2, (0.5 + n_pix) * pix_size, pix_size)
    hit_center = (pix_size * n_pix / 2, pix_size * n_pix / 2)
    hit_sigma = pix_size
    track_centers = np.random.normal(size=(n_events, 2)) * hit_sigma + np.array(hit_center)
    if plot_spatial_hit_dist:
        fig, ax = plt.subplots()
        ax.scatter(track_centers[:, 0], track_centers[:, 1], alpha=0.5)

    # track_sigma_analysis(track_centers, x_pix_centers, y_pix_centers, hit_threshold, pix_size, plot_events, n_pix)

    residuals = calc_residuals(track_centers, x_pix_centers, y_pix_centers, hit_threshold, pix_size, plot_events, n_pix,
                               track_gaus_sigma)
    plot_residuals(residuals, pix_size)

    plt.show()
    print('donzo')


def track_sigma_analysis(track_centers, x_pix_centers, y_pix_centers, hit_threshold, pix_size, plot_events, n_pix):
    track_sigmas = np.linspace(50, 70, 30)
    ratio_3_to_2, ratio_4_to_2, ratio_4_to_3 = [], [], []
    for track_sigma in track_sigmas:
        print(f'Running track_sigma {track_sigma:.2f}')
        residuals = calc_residuals(track_centers, x_pix_centers, y_pix_centers, hit_threshold, pix_size, plot_events,
                                   n_pix, track_sigma)
        if len(residuals[2]) > 0:
            ratio_3_to_2.append(len(residuals[3]) / len(residuals[2]))
            ratio_4_to_2.append(len(residuals[4]) / len(residuals[2]))
        else:
            ratio_3_to_2.append(float('nan'))
            ratio_4_to_2.append(float('nan'))
        if len(residuals[3]) > 0:
            ratio_4_to_3.append(len(residuals[4]) / len(residuals[3]))
        else:
            ratio_4_to_3.append(float('nan'))
    fig, ax = plt.subplots()
    ax.plot(track_sigmas, ratio_3_to_2, color='blue', label='3 hits / 2 hits')
    ax.plot(track_sigmas, ratio_4_to_2, color='green', label='4 hits / 2 hits')
    ax.plot(track_sigmas, ratio_4_to_3, color='purple', label='4 hits / 3 hits')
    ax.axhspan(0.7, 0.8, color='blue', ls='--', alpha=0.3)
    ax.axhspan(0.9, 1.05, color='green', ls='--', alpha=0.3)
    ax.axhspan(1.1, 1.2, color='purple', ls='--', alpha=0.3)
    ax.set_xlabel('Track Gaussian Sigma')
    ax.set_ylabel('Ratio of events with N hits')
    ax.grid()
    ax.set_ylim(0, 1.5)
    ax.legend()
    fig.tight_layout()


def plot_residuals(residuals, pix_size):
    bin_edges = np.linspace(-pix_size / 2, pix_size / 2, 41)
    fig, ax = plt.subplots()
    n_hits_plot, resolutions, n_events_n_hits = [], [], []
    for n_hits, x_residuals in residuals.items():
        if len(x_residuals) > 0:
            ax.hist(x_residuals, bins=bin_edges, alpha=0.3, label=f'{n_hits} hits')
            n_hits_plot.append(n_hits)
            resolutions.append(np.std(x_residuals))
            n_events_n_hits.append(len(x_residuals))
    ax.set_xlabel('X Residuals')
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.bar(n_hits_plot, resolutions, width=1)
    ax.set_xlabel('Number of Hits')
    ax.set_ylabel(r'Resolution ($\mu m$)')
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.bar(n_hits_plot, n_events_n_hits, width=1)
    ax.set_xlabel('Number of Hits')
    ax.set_ylabel('Number of Events')
    fig.tight_layout()


def calc_residuals(track_centers, x_pix_centers, y_pix_centers, hit_threshold, pix_size, plot_events, n_pix,
                   track_gaus_sigma):
    residuals = {n_hit: [] for n_hit in range(21)}
    for x_track, y_track in track_centers:
        hits = []
        for x_pix in x_pix_centers:  # Horribly slow, have copilot help me speed this up
            for y_pix in y_pix_centers:
                pix_amp = gaus_2d(x_pix, y_pix, 1, x_track, y_track, track_gaus_sigma, track_gaus_sigma)
                # print(f'({x_track:0.2f}, {y_track:0.2f}) --> ({x_pix:0.2f}, {y_pix:0.2f}): {pix_amp:0.5f}')
                if pix_amp > hit_threshold:
                    hits.append(np.array((x_pix, y_pix)))
        hits = np.array(hits)
        if len(hits) > 0:
            center = np.mean(hits, axis=0)
            x_resid = x_track - center[0]
            n_hits = len(hits)
            # if n_hits == 2:
            #     print(f'({x_track:0.2f}, {y_track:0.2f}) --> {center}, {x_resid:.3f}\n{hits}')
            if n_hits in residuals:
                residuals[n_hits].append(x_resid)
            if plot_events:
                fig, ax = plt.subplots()
                ax.scatter([x_track], [y_track], color='blue', alpha=0.5)
                ax.scatter(hits[:, 0], hits[:, 1], color='red', alpha=0.5)
                ax.scatter(center[0], center[1], color='green', alpha=0.5)
                ax.axhline(0, color='black', alpha=0.3)
                ax.axhline(pix_size * (n_pix + 1), color='black', alpha=0.3)
                ax.axvline(0, color='black', alpha=0.3)
                ax.axvline(pix_size * (n_pix + 1), color='black', alpha=0.3)
                plt.show()
    return residuals


def gaus_plot_test():
    fig, ax = plt.subplots()
    xs = np.linspace(-5, 5, 11)
    ys = np.linspace(-5, 5, 101)
    for x in xs:
        ax.plot(ys, gaus_2d(x, ys, 1, 0, 0, 2, 2), label=f'x={x:.2f}')
    ax.set_xlabel('y')
    ax.legend()
    fig.tight_layout()
    plt.show()


def gaus_2d(x, y, a, x_mu, y_mu, x_sig, y_sig):
    return a * np.exp(-0.5 * (((x - x_mu) / x_sig) ** 2 + ((y - y_mu) / y_sig) ** 2))


if __name__ == '__main__':
    main()
