#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 15 10:40 PM 2025
Created in PyCharm
Created as saclay_micromegas/mx17_test.py

@author: Dylan Neff, Dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
from DreamDetector import DreamDetector
from DreamData import read_det_data_vars


def main():
    base_path = 'F:/Saclay/MX17/'
    # run_name = 'night_test_non_zs_10-15-25'
    # run_name = 'night_test_non_zs_10-20-25'
    run_name = 'night_test_non_zs_config2-2_10-23-25'
    # sub_run_name = 'night_test_short'
    sub_run_name = 'night_test_long'
    root_path = f'{base_path}{run_name}/{sub_run_name}/decoded_root/'
    for file in os.listdir(root_path):
        if file.endswith('.root'):
            if '_pedthr_' in file:
                pedestal_path = f'{root_path}{file}'
            elif '_datrun_' in file:
                data_path = f'{root_path}{file}'

    all_data = read_det_data_vars(data_path, ['amplitude', 'eventId', 'timestamp', 'ftst'])
    amplitudes = all_data['amplitude']
    print(amplitudes.shape)

    # events = [3, 4, 5, 6, 7, 8, 9, 10]
    # events = [5]
    # for event in events:
    #     # In event, find channels which have a max amplitude greater than 1000
    #     high_amp_channels = np.where(np.max(amplitudes[event, :, :], axis=1) > 1000)[0]
    #     print(f'High amplitude channels in event 1: {high_amp_channels}')
    #
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     for channel in range(amplitudes.shape[1]):
    #         ax.plot(amplitudes[event, channel, :], marker='none', alpha=0.2)
    #     ax.set_title('Amplitudes for All Channels in First Event')
    #     ax.set_xlabel('Sample Index')
    #     ax.set_ylabel('Amplitude')
    # # plt.show()
    #
    # max_amps = np.max(amplitudes, axis=2)  # shape: (n_events, n_channels)
    # channel_a = 436
    # channel_b = 437
    # # Plot max amplitude of channel_a vs channel_b across all events
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.scatter(max_amps[:, channel_a], max_amps[:, channel_b], alpha=0.5)
    # # Do linear fit, plot and print fit parameters in annotate
    # # Filter out top 1% of amplitudes on channel_a to avoid outliers
    # threshold = np.percentile(max_amps[:, channel_a], 99)
    # fit_amps_a = max_amps[max_amps[:, channel_a] < threshold, channel_a]
    # fit_amps_b = max_amps[max_amps[:, channel_a] < threshold, channel_b]
    # fit_params, fit_cov = cf(linear_func, fit_amps_a, fit_amps_b)
    # fit_x = np.linspace(0, np.max(max_amps[:, channel_a]), 100)
    # fit_y = linear_func(fit_x, *fit_params)
    # ax.plot(fit_x, fit_y, color='red', label=f'Fit: y = {fit_params[0]:.2f}x + {fit_params[1]:.2f}')
    # ax.annotate(f'Slope: {fit_params[0]:.2f}\nIntercept: {fit_params[1]:.2f}',
    #             xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10,
    #             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3),
    #             verticalalignment='top')
    # ax.legend()
    # ax.set_title(f'Max Amplitude: Channel {channel_a} vs Channel {channel_b}')
    # ax.set_xlabel(f'Channel {channel_a} Max Amplitude')
    # ax.set_ylabel(f'Channel {channel_b} Max Amplitude')
    # fig.tight_layout()
    #
    # analyze_channel_correlations(amplitudes, main_channel=436, neighbor_range=15, percentile_cut=95, plot_examples=False)
    # analyze_channel_correlations(amplitudes, main_channel=395, neighbor_range=15, percentile_cut=95, plot_examples=False)
    #
    # # plt.show()
    #
    # channels = [435, 436, 437, 438, 439, 440]
    # # for channel in channels:
    # #     fig, ax = plt.subplots(figsize=(10, 6))
    # #     for event in range(amplitudes.shape[0]):
    # #         ax.plot(amplitudes[event, channel, :], marker='none', color='red', alpha=0.2)
    # #     ax.set_title(f'Amplitudes for Channel {channel} Across All Events')
    # #     ax.set_xlabel('Sample Index')
    # #     ax.set_ylabel('Amplitude')
    # #     fig.tight_layout()
    #
    # n_ch = len(channels)
    # fig, axes = plt.subplots(nrows=n_ch, ncols=1, figsize=(8, 9), sharex='all', gridspec_kw={'hspace': 0})
    # if n_ch == 1:
    #     axes = [axes]
    # for ax, channel in zip(axes, channels):
    #     for event in range(amplitudes.shape[0]):
    #         ax.plot(amplitudes[event, channel, :], marker='none', color='red', alpha=0.2)
    #     ax.set_ylabel('Amplitude')
    #     ax.annotate(f'Channel {channel}', xy=(0.02, 0.85), xycoords='axes fraction', fontsize=10,
    #                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
    # # hide x tick labels for all but the bottom subplot
    # # for ax in axes[:-1]:
    # #     ax.set_xticklabels([])
    # #     ax.tick_params(axis='x', which='both', length=0)
    # axes[-1].set_xlabel('Sample Index')
    # # axes[-1].set_xlim(left=60)
    # fig.subplots_adjust(hspace=0)
    # fig.tight_layout(h_pad=0)
    #
    # channels = [395, 436]
    # for channel in channels:
    #     # # Extract the amplitudes for that channel across all events
    #     data = amplitudes[:, channel, :]  # shape: (n_events, n_samples)
    #
    #     # Flatten to make 2D histogram inputs:
    #     # x = sample index repeated for each event
    #     # y = amplitude values
    #     n_events, n_samples = data.shape
    #     x = np.tile(np.arange(n_samples), n_events)
    #     y = data.flatten()
    #
    #     # Create 2D histogram
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     sample_bins = np.arange(-0.5, n_samples + 1.5, 1)
    #     amp_bins = np.arange(-0.5, np.max(data) + 1.5, 50)
    #     h = ax.hist2d(x, y, bins=(sample_bins, amp_bins), cmap='inferno', norm='log')
    #
    #     ax.set_title(f'Amplitude Distribution for Channel {channel}')
    #     ax.set_xlabel('Sample Index')
    #     ax.set_ylabel('Amplitude')
    #     fig.colorbar(h[3], ax=ax, label='Counts (log scale)')
    #     fig.tight_layout()
    #
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     h = ax.hist2d(x, y, bins=(sample_bins, amp_bins), cmap='inferno', cmin=1, cmax=1000)
    #
    #     ax.set_title(f'Amplitude Distribution for Channel {channel}')
    #     ax.set_xlabel('Sample Index')
    #     ax.set_ylabel('Amplitude')
    #     fig.colorbar(h[3], ax=ax, label='Counts (log scale)')
    #     fig.tight_layout()
    #
    #     pedestals = amplitudes[:, channel, :8]
    #     signals = amplitudes[:, channel, 8:]
    #
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     bins = np.arange(-0.5, np.max(amplitudes) + 1.5, 20)
    #     ax.hist(pedestals.flatten(), bins=bins, histtype='step', label='Pedestal Samples', color='blue', density=True)
    #     ax.hist(signals.flatten(), bins=bins, histtype='step', label='Signal Samples', color='red', density=True)
    #     ax.set_title(f'Pedestal vs Signal Amplitude Distribution for Channel {channel}')
    #     ax.set_xlabel('Amplitude')
    #     ax.set_ylabel('Counts')
    #     ax.legend()
    #     fig.tight_layout()
    #
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     bins = np.arange(369.5, 550 + 1.5, 1)
    #     ax.hist(pedestals.flatten(), bins=bins, histtype='step', label='Pedestal Samples', color='blue', density=True)
    #     ax.hist(signals.flatten(), bins=bins, histtype='step', label='Signal Samples', color='red', density=True)
    #     ax.set_title(f'Pedestal vs Signal Amplitude Distribution for Channel {channel}')
    #     ax.set_xlabel('Amplitude')
    #     ax.set_ylabel('PDF')
    #     ax.legend()
    #     fig.tight_layout()
    #
    #
    #     # det = DreamDetector()
    #     # det.feu_num = 6
    #     # det.feu_connectors = [7]
    #     # det.load_dream_data(root_path, root_path, 4, 'all', 1, hist_raw_amps=True, save_waveforms=True,
    #     #                     waveform_fit_func='parabola_vectorized')
    #     #
    #     # print(f'Loaded {det.dream_data.hits.shape[0]} events with {det.dream_data.hits.shape[1]} channels each.')
    # Sum the waveforms from channels and plot max amplitude distribution of the sum and separate
    sig_chan_a, sig_chan_b = 395, 436
    ped_split_index = 8
    sum_waveforms = amplitudes[:, sig_chan_a, :] + amplitudes[:, sig_chan_b, :]
    avg_maxes = np.max(sum_waveforms, axis=-1) / 2
    sig_chan_a_maxes = np.max(amplitudes[:, sig_chan_a, :], axis=-1)
    sig_chan_b_maxes = np.max(amplitudes[:, sig_chan_b, :], axis=-1)
    sig_chan_a_peds = amplitudes[:, sig_chan_a, :ped_split_index]
    sig_chan_b_peds = amplitudes[:, sig_chan_b, :ped_split_index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(-0.5, np.max(sig_chan_b_maxes) + 21.5, 20)
    bins_ped = np.arange(-0.5, np.max(sig_chan_b_peds) + 1.5, 1)
    ax2 = ax.twinx()
    ax.hist(sig_chan_a_maxes, bins=bins, histtype='step', color='black', label=f'Channel {sig_chan_a} Event Maxes')
    ax.hist(sig_chan_b_maxes, bins=bins, histtype='step', color='green', label=f'Channel {sig_chan_b} Event Maxes')
    ax.hist(avg_maxes, bins=bins, histtype='step', color='red', label=f'({sig_chan_a}+{sig_chan_b})/2 Event Maxes', linewidth=2)
    ped_hist_a, bins_ped, _ = ax2.hist(sig_chan_a_peds.flatten(), bins=bins_ped, color='black', label=f'Channel {sig_chan_a} Pedestal Samples')
    ped_hist_b, bins_ped, _ = ax2.hist(sig_chan_b_peds.flatten(), bins=bins_ped, color='green', label=f'Channel {sig_chan_b} Pedestal Samples')

    # Fit Gaussian to pedestal histograms
    bin_centers = (bins_ped[:-1] + bins_ped[1:]) / 2
    popt_a, _ = cf(gaus, bin_centers, ped_hist_a, p0=[np.max(ped_hist_a), bin_centers[np.argmax(ped_hist_a)], 1.0])
    popt_b, _ = cf(gaus, bin_centers, ped_hist_b, p0=[np.max(ped_hist_b), bin_centers[np.argmax(ped_hist_b)], 1.0])
    fit_x = np.linspace(380, 440, 300)
    fit_y_a = gaus(fit_x, *popt_a)
    fit_y_b = gaus(fit_x, *popt_b)
    ax2.plot(fit_x, fit_y_a, 'k--')
    ax2.plot(fit_x, fit_y_b, 'g--')
    ax2.annotate(f'Chan {sig_chan_a} Pedestal:\n Mean = {popt_a[1]:.2f}\n Sigma = {popt_a[2]:.2f}',
                 xy=(0.6, 0.6), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.3),
                 verticalalignment='top')
    ax2.annotate(f'Chan {sig_chan_b} Pedestal:\n Mean = {popt_b[1]:.2f}\n Sigma = {popt_b[2]:.2f}',
                 xy=(0.6, 0.43), xycoords='axes fraction', fontsize=10, color='green',
                 bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.3),
                 verticalalignment='top')

    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Signal Events')
    ax2.set_ylabel('Pedestal Samples')
    ax.legend(loc='upper right')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.82), bbox_transform=ax.transAxes)
    ax.set_title(f'Event Max Amplitudes and Pedestal Distributions for Channels {sig_chan_a} and {sig_chan_b}')
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(380.5, 620.5, 5)
    bins_ped = np.arange(380.5, 620.5, 1)
    ax2 = ax.twinx()
    ax.set_zorder(2)
    ax.patch.set_visible(False)

    ped_hist_a, bins_ped, _ = ax2.hist(sig_chan_a_peds.flatten(), bins=bins_ped, color='black', zorder=5,
                                       label=f'Channel {sig_chan_a} Pedestal Samples')
    ped_hist_b, bins_ped, _ = ax2.hist(sig_chan_b_peds.flatten(), bins=bins_ped, color='green', zorder=6,
                                       label=f'Channel {sig_chan_b} Pedestal Samples')
    ax.hist(sig_chan_a_maxes, bins=bins, histtype='step', color='black', label=f'Channel {sig_chan_a} Event Maxes', zorder=10)
    ax.hist(sig_chan_b_maxes, bins=bins, histtype='step', color='green', label=f'Channel {sig_chan_b} Event Maxes', zorder=10)
    # ax.hist(avg_maxes, bins=bins, histtype='step', color='red', label=f'({sig_chan_a}+{sig_chan_b})/2 Event Maxes',
    #         linewidth=2, zorder=11)

    # Fit Gaussian to pedestal histograms
    bin_centers = (bins_ped[:-1] + bins_ped[1:]) / 2
    popt_a, _ = cf(gaus, bin_centers, ped_hist_a, p0=[np.max(ped_hist_a), bin_centers[np.argmax(ped_hist_a)], 1.0])
    popt_b, _ = cf(gaus, bin_centers, ped_hist_b, p0=[np.max(ped_hist_b), bin_centers[np.argmax(ped_hist_b)], 1.0])
    fit_x = np.linspace(380, 440, 300)
    fit_y_a = gaus(fit_x, *popt_a)
    fit_y_b = gaus(fit_x, *popt_b)
    ax2.plot(fit_x, fit_y_a, 'k--')
    ax2.plot(fit_x, fit_y_b, 'g--')
    ax2.annotate(f'Chan {sig_chan_a} Pedestal:\n Mean = {popt_a[1]:.2f}\n Sigma = {popt_a[2]:.2f}',
                 xy=(0.4, 0.6), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.3),
                 verticalalignment='top')
    ax2.annotate(f'Chan {sig_chan_b} Pedestal:\n Mean = {popt_b[1]:.2f}\n Sigma = {popt_b[2]:.2f}',
                 xy=(0.4, 0.43), xycoords='axes fraction', fontsize=10, color='green',
                 bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.3),
                 verticalalignment='top')

    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Signal Events')
    ax2.set_ylabel('Pedestal Samples')
    ax.legend(loc='upper center')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.85), bbox_transform=ax.transAxes)
    ax.set_title(f'Event Max Amplitudes and Pedestal Distributions for Channels {sig_chan_a} and {sig_chan_b}')
    fig.tight_layout()


    plt.show()

    print('donzo')


def linear_func(x, m, b):
    return m * x + b


def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def analyze_channel_correlations(amplitudes, main_channel, neighbor_range=5, percentile_cut=99, plot_examples=True):
    """
    For a given main_channel, fit linear relationships between its max amplitudes
    and those of nearby channels. Then plot the slope vs channel number.

    Parameters
    ----------
    amplitudes : np.ndarray
        3D array with shape (n_events, n_channels, n_samples)
    main_channel : int
        Channel number to analyze correlations around.
    neighbor_range : int, optional
        Number of neighboring channels on each side to analyze (default: 5)
    percentile_cut : float, optional
        Percentile cutoff for filtering out high outliers (default: 99)
    plot_examples : bool, optional
        Whether to show example scatter+fit plots for each neighbor (default: True)

    Returns
    -------
    slopes : dict
        Dictionary mapping channel number -> slope from linear fit.
    """

    # Compute maximum amplitude per event per channel
    max_amps = np.max(amplitudes, axis=2)  # shape (n_events, n_channels)

    n_channels = max_amps.shape[1]
    slopes = {}

    # Define which channels to analyze
    channel_list = [
        ch for ch in range(main_channel - neighbor_range, main_channel + neighbor_range + 1)
        if 0 <= ch < n_channels and ch != main_channel
    ]

    main_amps = max_amps[:, main_channel]
    threshold = np.percentile(main_amps, percentile_cut)
    mask = main_amps < threshold
    main_fit_amps = main_amps[mask]

    for ch in channel_list:
        neighbor_fit_amps = max_amps[mask, ch]
        fit_params, _ = cf(linear_func, main_fit_amps, neighbor_fit_amps)
        slopes[ch] = fit_params[0]

        if plot_examples:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(main_fit_amps, neighbor_fit_amps, alpha=0.4, s=10)
            ax.scatter(main_amps[~mask], max_amps[~mask, ch], color='gray', alpha=0.2, s=10, label='Filtered Outliers')
            fit_x = np.linspace(np.min(main_fit_amps), np.max(main_fit_amps), 100)
            fit_y = linear_func(fit_x, *fit_params)
            ax.plot(fit_x, fit_y, 'r', label=f"Slope={fit_params[0]:.2f}")
            ax.legend()
            ax.set_xlabel(f"Channel {main_channel} Max Amplitude")
            ax.set_ylabel(f"Channel {ch} Max Amplitude")
            ax.set_title(f"Linear Fit: Channel {ch} vs {main_channel}")
            plt.tight_layout()

    # Plot slope vs channel number
    fig, ax = plt.subplots(figsize=(8, 5))
    chs = list(slopes.keys())
    slope_vals = [v * 100 for v in slopes.values()]
    ax.plot(chs, slope_vals, 'ko-', label="Slope vs Channel", zorder=10)
    ax.axhline(0, color='gray', linestyle='-')
    ax.axvline(main_channel, color='red', linestyle='--', label='Main Channel')
    ax.set_xlabel("Channel Number")
    ax.set_ylabel("Linear Fit Slope -- Channel Amplitude Relative to Main Channel")
    ax.set_title(f"Crosstalk Size Relative to Main Channel Signal {main_channel}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1f}%"))
    ax.legend()
    plt.tight_layout()

    return slopes


if __name__ == '__main__':
    main()
