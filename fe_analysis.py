#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 14 2:25 PM 2023
Created in PyCharm
Created as saclay_micromegas/fe_analysis.py

@author: Dylan Neff, Dylan
"""

import os
import subprocess
import shutil
import re
from datetime import datetime, timedelta
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from scipy.optimize import curve_fit as cf

import uproot
import awkward as ak

from Measure import Measure


def process_fdf(file, fdf_dir, raw_root_dir, overwrite, file_i):
    root_name = file[:-4] + '.root'
    spacing_time = file_i * 2  # s
    if not overwrite and root_name in os.listdir(raw_root_dir):
        print(f'{root_name} already exists in {raw_root_dir}, skipping')
    else:
        read_fdf_to_root(file, fdf_dir, raw_root_dir, root_name, spacing_time)
    return root_name


def read_fdf_to_root(file, fdf_dir, raw_root_dir, root_name, wait_time):
    cmd = f'source /home/dylan/Software/root/bin/thisroot.sh && ./DreamDataReader {os.path.join(fdf_dir, file)}'
    # os.system(f'source /home/dylan/Software/root/bin/thisroot.sh && ./DreamDataReader {os.path.join(fdf_dir, file)}')
    # subprocess.Popen(['source', '/home/dylan/Software/root/bin/thisroot.sh', f'./DreamDataReader {os.path.join(fdf_dir, file)}'])
    sleep(wait_time)
    try:
        with open('dream_data_reader_output_file_name.txt', 'w') as file:
            file.write(root_name)
        print(f"Filename '{root_name}' written to the file.")
    except Exception as e:
        print(f"Error: {e}")

    subprocess.run(['bash', '-c', cmd])
    raw_root_path = os.path.join(raw_root_dir, root_name)
    shutil.move(root_name, raw_root_path)
    return raw_root_path


def plot_adc_uproot(file_path):
    # Open the ROOT file with uproot
    root_file = uproot.open(file_path)

    # Access the tree in the file
    tree = root_file['T']

    # Get the variable data from the tree
    variable_name = 'StripAmpl'
    variable_data = ak.flatten(tree[variable_name].array(), axis=None)
    print(variable_data)

    # Create a histogram using Matplotlib
    plt.hist(variable_data, bins=50, edgecolor='black')

    # Set plot labels and title
    plt.xlabel(variable_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {variable_name}")

    # Display the plot
    plt.show()


def read_det_data(file_path, num_detectors=None, variable_name='StripAmpl', tree_name='T'):
    # Open the ROOT file with uproot
    root_file = uproot.open(file_path)

    # Access the tree in the file
    tree = root_file[tree_name]

    # Get the variable data from the tree
    variable_data = ak.to_numpy(tree[variable_name].array())
    root_file.close()

    if num_detectors is not None:
        variable_data = variable_data[:, :num_detectors]

    return variable_data


def read_det_data_chunk(chunk, num_detectors=None):
    chunk = ak.to_numpy(chunk)
    if num_detectors is not None:
        chunk = chunk[:, :num_detectors]

    return chunk


def get_pedestals(data):
    strip_medians = np.median(data, axis=3)  # Median of samples in each strip for each event
    strip_medians_transpose = np.transpose(strip_medians, axes=(1, 2, 0))  # Concatenate strips for each event
    strip_means = np.mean(strip_medians_transpose, axis=2)  # Mean of strip medians over all events

    # This takes the median of samples over all events for each strip
    # data_event_concat = np.concatenate(data, axis=2)
    # strip_means = np.mean(data_event_concat, axis=2)
    # strip_medians = np.median(data_event_concat, axis=2)

    return strip_means


def get_common_noise(data, pedestals):
    data_ped_sub = data - pedestals[np.newaxis, :, :, np.newaxis]
    data_ped_sub_trans = np.transpose(data_ped_sub, axes=(0, 1, 3, 2))
    common_noise = np.nanmedian(data_ped_sub_trans, axis=-1)

    return common_noise


def get_pedestals_rms(ped_data, ped_means):
    ped_zeroed = subtract_pedestal(ped_data, ped_means)  # Subtract averages from pedestal data
    ped_zeroed_concat = np.concatenate(ped_zeroed, axis=2)  # Concatenate all events
    ped_rms = np.std(ped_zeroed_concat, axis=2)  # Get RMS of pedestal data for each strip
    # Existing code averages over all strips, but will see if we can get away with strip by strip
    return ped_rms


def gaussian_density(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def gaus_exp(x, a, mu, sigma, b, c):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + b * np.exp(-c * x)


def poly2(x, a, b):
    return a * x ** 2 + b * x


def poly2_center(x, mu, a, b):
    return a * (x - mu) ** 2 + b * (x - mu)


def poly1(x, a):
    return a * x


def gaus_poly(x, a, mu, sigma, b):
    return gaussian(x, a, mu, sigma) + poly1(x, b)


def fit_pedestals(strip_samples):
    bin_edges = np.arange(-0.5, 4097.5, 1)
    mean = np.mean(strip_samples)
    sd = np.std(strip_samples)
    hist, _ = np.histogram(strip_samples, bins=bin_edges, density=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    popt, pcov = cf(gaussian_density, bin_centers, hist, p0=[mean, sd])
    perr = np.sqrt(np.diag(pcov))
    return *popt, *perr


def get_pedestal_fits(ped_data, common_noise=None):
    if common_noise is not None:
        ped_data = ped_data - common_noise[:, :, np.newaxis, :]
    ped_concat = np.concatenate(ped_data, axis=2)  # Concatenate all events
    ped_fits = np.apply_along_axis(fit_pedestals, -1, ped_concat)
    ped_fits = np.transpose(ped_fits, axes=(2, 0, 1))
    ped_fits = dict(zip(['mean', 'sigma', 'mean_err', 'sigma_err'], ped_fits))
    # Existing code averages over all strips, but will see if we can get away with strip by strip
    return ped_fits


def get_noise_thresholds(ped_rms, noise_sigmas=5):
    return noise_sigmas * ped_rms


def plot_1d_data(data, title=None):
    fig, ax = plt.subplots()
    ax.plot(range(len(data)), data)
    ax.set_xlabel('Strip Number')
    ax.set_ylabel("ADC")
    ax.set_title('Strip ADC')
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()


def plot_pedestal_comp(ped_dict):
    num_detectors = len(list(ped_dict.values())[0]['vals'])
    for det_num in range(num_detectors):
        fig, ax = plt.subplots()
        for ped_name, vals_errs in ped_dict.items():
            if 'errs' in vals_errs:
                ax.errorbar(range(len(vals_errs['vals'][det_num])), vals_errs['vals'][det_num],
                            yerr=vals_errs['errs'][det_num], label=ped_name)
            else:
                ax.plot(range(len(vals_errs['vals'][det_num])), vals_errs['vals'][det_num], label=ped_name)
        ax.set_xlabel('Strip Number')
        ax.set_ylabel("ADC")
        ax.set_title(f'Detector #{det_num} Pedestals')
        ax.legend()
        fig.tight_layout()


def plot_2d_data(data_x, data_y):
    fig, ax = plt.subplots()
    # Create a grid of indices
    x_indices, y_indices = np.meshgrid(np.arange(len(data_x)), np.arange(len(data_y)))

    # Compute the sums of x and y values at each point
    sums = data_x[x_indices] + data_y[y_indices]

    # Create a 2D colormesh
    plt.pcolormesh(x_indices, y_indices, sums, cmap='viridis')
    plt.colorbar(label='Sum of Values')

    # Set labels and title
    plt.xlabel('X Index')
    plt.ylabel('Y Index')
    plt.title('Sum of X and Y Values at Each Index')


def plot_combined_time_series(data, max_events=None):
    if len(data) == 0:
        return
    n_events, n_samples_per_event = data.shape[0], data.shape[-1]
    if max_events is not None:
        n_events = min(n_events, max_events)
    for det_num, det in enumerate(np.concatenate(data, axis=2)):
        fig, ax = plt.subplots()
        for strip in det:
            ax.plot(range(len(strip[:n_events * n_samples_per_event])), strip[:n_events * n_samples_per_event])

        # Set plot labels and title
        for event_i in range(n_events):
            ax.axvline(event_i * n_samples_per_event, color='black', ls='--')
        ax.set_xlabel('Sample Number')
        ax.set_ylabel("ADC")
        ax.set_title(f"Detector #{det_num}")


def plot_adc_scatter_vs_strip(data):
    data_transpose = np.transpose(data, (1, 2, 3, 0))
    for det_num, det in enumerate(data_transpose):
        fig, ax = plt.subplots()
        for strip_num, strip in enumerate(det):
            samples = np.concatenate(strip)
            ax.scatter([strip_num] * len(samples), samples, marker='_', alpha=0.2)
        ax.set_xlabel('Strip Number')
        ax.set_ylabel("ADC")
        ax.set_title(f'ADC Distribution per Strip Detector #{det_num}')
        plt.tight_layout()


def subtract_pedestal(data, pedestal):
    return data - pedestal[np.newaxis, :, :, np.newaxis]


def get_sample_max(data):
    return np.max(data, axis=-1)


def get_max_edge_events(data, edge_strips=None):
    """
    Find events in which the detector's max strip's max sample is on the edge of the time window.
    :param data: ADC data of shape (n_events, n_detectors, n_strips, n_samples)
    :param edge_strips: List of strips to consider edge strips. If None, don't filter out edge strips.
    :return: Boolean array of shape (n_events,) where True means event is an edge max event.
    """
    event_samples = data.reshape(data.shape[0], np.product(data.shape[1:]))
    event_max_sample_indices = np.unravel_index(np.argmax(event_samples, axis=-1), data.shape[1:])
    event_max_time_index = event_max_sample_indices[-1]
    time_edge_max_events = np.logical_or(event_max_time_index == 0, event_max_time_index == data.shape[-1] - 1)

    if edge_strips is not None:
        event_max_det_strip_indices = np.array(event_max_sample_indices[:2]).transpose()
        # space_edge_max_events = np.all(edge_strips[np.newaxis, :, :] == event_max_det_strip_indices, axis=1)
        space_edge_max_events = np.any(np.all(edge_strips == event_max_det_strip_indices[:, np.newaxis, :], axis=2),
                                       axis=1)

    return time_edge_max_events if edge_strips is None else np.logical_or(time_edge_max_events, space_edge_max_events)


def get_flat_signals(data, threshold=25):
    """
    Get events in which the max detector's max strip is flat vs time.
    :param data: ADC data of shape (n_events, n_detectors, n_strips, n_samples)
    :return:
    """
    event_samples = data.reshape(data.shape[0], np.product(data.shape[1:]))
    event_max_sample_indices = np.unravel_index(np.argmax(event_samples, axis=-1), data.shape[1:])
    det_max_indices, strip_max_indices, time_max_indices = event_max_sample_indices
    flat_mask = []
    for event_num, event in enumerate(data):
        event_det_max = event[det_max_indices[event_num]]
        event_det_max_strip = event_det_max[strip_max_indices[event_num]]
        event_det_max_strip_sample = event_det_max_strip[time_max_indices[event_num]]
        max_minus_mean = event_det_max_strip_sample - np.mean(event_det_max_strip)
        flat_mask.append(max_minus_mean < threshold)
    return np.array(flat_mask)


def get_flat_signals_gpt(data, threshold=100):
    """
    Get events in which the max detector's max strip is flat vs time.
    :param data: ADC data of shape (n_events, n_detectors, n_strips, n_samples)
    :param threshold: Threshold of max minus mean to consider flat
    :return: flat_mask: Boolean array indicating flat signals for each event
    """
    # Reshape data for easier manipulation
    event_samples = data.reshape(data.shape[0], -1)

    # Find indices of max samples
    event_max_sample_indices = np.argmax(event_samples, axis=-1)
    det_max_indices, strip_max_indices, time_max_indices = np.unravel_index(event_max_sample_indices, data.shape[1:])

    # Extract corresponding strips and samples
    event_det_max = data[np.arange(data.shape[0]), det_max_indices]
    event_det_max_strip = event_det_max[np.arange(data.shape[0]), strip_max_indices]
    event_det_max_strip_sample = event_det_max_strip[np.arange(data.shape[0]), time_max_indices]

    # Calculate max_minus_mean for each event
    mean_values = np.mean(event_det_max_strip, axis=-1, keepdims=True)
    max_minus_mean = event_det_max_strip_sample - mean_values

    # Create flat_mask based on threshold
    flat_mask = max_minus_mean < threshold

    return flat_mask


def plot_position_data(data, event_nums=None, plot_indiv_detectors=False):
    if event_nums is not None:
        data = data[event_nums]
    print(data.shape)
    for event_num, event in enumerate(data):
        fig, ax = plt.subplots()
        for det_num, det in enumerate(event):
            ax.plot(range(len(det)), det, label=f'Detector #{det_num}')
        ax.set_xlabel('Position')
        ax.set_ylabel("Max ADC")
        ax.set_title(f"Event #{event_num}")
        ax.legend()
    if not plot_indiv_detectors:
        data_transpose = np.transpose(data, (1, 0, 2))
        print(data_transpose.shape)
        for det_num, det in enumerate(data_transpose):
            fig, ax = plt.subplots()
            for event_num, event in enumerate(det):
                ax.plot(range(len(event)), event, label=f'Event #{event_num}')
            ax.set_xlabel('Position')
            ax.set_ylabel("Max ADC")
            ax.set_title(f"Detector #{det_num}")
            ax.legend()


def plot_urw_position(data, max_events=None, separate_event_plots=False, thresholds=None, plot_avgs=False):
    if max_events is not None:
        data = data[:max_events]
    data_shape = list(data.shape)
    data_shape[1] = data_shape[1] // 2
    data_shape[2] *= 2
    data = data.reshape(data_shape)
    data_transpose = np.transpose(data, (1, 0, 2))
    y_label = ['x Max Sample ADC', 'y Max Sample ADC']
    fig, axs = plt.subplots(nrows=2, figsize=(10, 5), dpi=144, sharex=True)
    for det_num, det in enumerate(data_transpose):
        for event_num, event in enumerate(det):
            axs[det_num].plot(range(len(event)), event, label=f'Event #{event_num}')
        axs[det_num].set_ylabel(y_label[det_num])
    axs[0].legend(loc='upper left')
    axs[-1].set_xlabel('Strip Number')
    fig.suptitle('URW Max Sample vs Strip Number')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, top=0.94, bottom=0.08, left=0.07, right=0.995)

    if thresholds is not None:
        thresholds_shape = list(thresholds.shape)
        thresholds_shape[0] = thresholds_shape[0] // 2
        thresholds_shape[1] *= 2
        thresholds = thresholds.reshape(thresholds_shape)
    mv_avg_pnts = [3, 4, 5]
    max_data_mv_avg = [np.apply_along_axis(np.convolve, -1, data, np.ones(i) / i, mode='same')
                       for i in mv_avg_pnts]

    if separate_event_plots:
        for event_num, event in enumerate(data):
            fig, axs = plt.subplots(nrows=2, figsize=(10, 5), dpi=144, sharex='col')
            for det_num, det in enumerate(event):
                axs[det_num].axhline(0, color='black', zorder=0)
                one_point = axs[det_num].plot(range(len(det)), det)
                mv_avg_colors = []
                if plot_avgs:
                    for mv_avg in max_data_mv_avg:
                        mv_avg_col = axs[det_num].plot(range(len(det)), mv_avg[event_num][det_num])
                        mv_avg_colors.append(mv_avg_col[0].get_color())
                    weird_thing = np.minimum((det[1:] + det[:-1])[1:], (det[1:] + det[:-1])[:-1]) / 2
                    weird_thing = np.insert(np.insert(weird_thing, len(weird_thing), weird_thing[-1]), 0, weird_thing[0])
                    axs[det_num].plot(range(len(weird_thing)), weird_thing)
                axs[det_num].set_ylabel(y_label[det_num])
                if thresholds is not None:
                    axs[det_num].plot(thresholds[det_num], ls='--', color=one_point[0].get_color())
                    if plot_avgs:
                        for mv_avg_pnt, color in zip(mv_avg_pnts, mv_avg_colors):
                            axs[det_num].plot(thresholds[det_num] / np.sqrt(mv_avg_pnt), ls='--', color=color)
                axs[det_num].annotate(f'Detector Sum: {int(np.sum(det))}', xy=(0.95, 0.1), xycoords='axes fraction', ha='right',
                                va='bottom', bbox=dict(boxstyle='round', fc='salmon'))
            axs[-1].set_xlabel('Strip Number')
            event_sum = np.sum(event)
            axs[0].annotate(f'Event Sum: {int(event_sum)}', xy=(0.95, 0.2), xycoords='axes fraction', ha='right',
                            va='bottom', bbox=dict(boxstyle='round', fc='wheat'))
            fig.suptitle(f'URW Max Sample vs Strip Number Event #{event_num}')
            fig.tight_layout()
            fig.subplots_adjust(hspace=0, top=0.94, bottom=0.08, left=0.07, right=0.995)


def plot_p2_2d(channel_sum, title=None, show_chan_nums=False):
    if show_chan_nums:
        large_pixels = np.arange(32)
        large_pixels = large_pixels.reshape(8, 4)
        large_pixels = large_pixels.transpose()
        small_pixels = np.arange(50).reshape(5, 10)

        vmin, vmax = 0, 50

        fig, ax = plt.subplots()
        im_small = ax.imshow(small_pixels, cmap='plasma', extent=[0, 10, 5, 10], vmin=vmin, vmax=vmax, origin='lower')
        im_large = ax.imshow(large_pixels, cmap='plasma', extent=[0, 10, 0, 5], vmin=vmin, vmax=vmax, origin='lower')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Annotate each pixel with its value
        for pixel_set, y_offset, pos_scale in zip([large_pixels, small_pixels], [0, 5], [5. / 4, 1]):
            for i in range(pixel_set.shape[0]):
                for j in range(pixel_set.shape[1]):
                    plt.annotate(f'{pixel_set[i, j]}', np.array((j + 0.5, i + y_offset + 0.5)) * pos_scale,
                                 color='white', ha='center', va='center')

        # Create a single colorbar for both plots
        cbar = fig.colorbar(im_small, ax=ax, orientation='vertical', pad=0.1)
        cbar.set_label('Value')

    # large_pixels, small_pixels = channel_sum[0][channel_sum[0] <= 0] = np.nan, channel_sum[1][channel_sum[1] <= 0] = np.nan
    large_pixels = channel_sum[0][:32].reshape(8, 4).transpose()
    small_pixels = channel_sum[1][:50].reshape(5, 10)[::-1, ::-1] * (5. / 4)**2
    large_pixel_indices = np.arange(32).reshape(8, 4).transpose()
    small_pixel_indices = np.arange(50).reshape(5, 10)[::-1, ::-1]

    # large_pixels, small_pixels = large_pixels[large_pixels <= 0] = np.nan, small_pixels[small_pixels <= 0] = np.nan
    large_pixels, small_pixels = np.where(large_pixels <= 0, np.nan, large_pixels), np.where(small_pixels <= 0, np.nan, small_pixels)
    large_pixels, small_pixels = np.ma.masked_invalid(large_pixels), np.ma.masked_invalid(small_pixels)

    vmin, vmax = np.min(channel_sum), np.max(channel_sum)

    fig, ax = plt.subplots()
    im_small = ax.imshow(small_pixels, cmap='plasma', extent=[0, 10, 5, 10], vmin=vmin, vmax=vmax, origin='lower')
    im_large = ax.imshow(large_pixels, cmap='plasma', extent=[0, 10, 0, 5], vmin=vmin, vmax=vmax, origin='lower')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('Horizontal Position (arb)')
    ax.set_ylabel('Vertical Position (arb)')
    if title is not None:
        ax.set_title(title)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    if show_chan_nums:  # Annotate each pixel with its value
        for pixel_set, y_offset, pos_scale in zip([large_pixel_indices, small_pixel_indices], [0, 5], [5. / 4, 1]):
            for i in range(pixel_set.shape[0]):
                for j in range(pixel_set.shape[1]):
                    plt.annotate(f'{pixel_set[i, j]}', np.array((j + 0.5, i + y_offset + 0.5)) * pos_scale,
                                 color='white', ha='center', va='center')

    # Create a single colorbar for both plots
    cbar = fig.colorbar(im_small, ax=ax, orientation='vertical', pad=0.1)
    cbar.set_label('ADC Sum Over All Events')
    fig.tight_layout()

    return fig


def plot_det_spectrum(signal_data):
    fig, ax = plt.subplots()
    for det_num, det in enumerate(signal_data):
        ax.hist(det, bins=50, edgecolor='black', label=f'Detector #{det_num}')
    ax.set_xlabel('ADC')
    ax.set_ylabel('Frequency')
    ax.set_title('ADC Spectrum')


def plot_spectrum(signal_sum_data, bins=50, title=None, save_path=None):
    fig, ax = plt.subplots()
    hist, bin_edges = np.histogram(signal_sum_data, bins=bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    ax.bar(bin_centers, hist, width=bin_centers[1] - bin_centers[0], edgecolor='black')
    ax.set_xlabel('ADC')
    ax.set_ylabel('Events')
    if title is None:
        ax.set_title('ADC Sum Spectrum')
    else:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path)


def plot_1d_sample_max_hist(max_data, bins=100, title=None, xlabel='ADC', log=False):
    # print(f'max_data shape: {max_data.shape}')
    # max_data_shape = list(max_data.shape)
    # max_data_shape[1] = max_data_shape[1] // 2
    # max_data_shape[0] = max_data_shape[0] * 2
    # max_data = max_data.reshape(max_data_shape)
    fig, ax = plt.subplots()
    for det_num, det in enumerate(np.transpose(max_data)):
        ax.hist(det, bins=bins, edgecolor='black', label=f'Detector #{det_num}')
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Events')
    if log:
        ax.set_yscale('log')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Sample Max ADC Spectrum')


def plot_sample_maxes(max_data):
    print(max_data.shape)
    det_max = np.max(max_data, axis=2)
    print(det_max.shape)
    print(det_max)
    det_max_trans = np.transpose(det_max)
    print(det_max_trans.shape)
    print(det_max_trans)
    for det_num, det_maxes in enumerate(det_max_trans):
        fig, ax = plt.subplots()
        ax.plot(range(len(det_maxes)), det_maxes, label=f'Detector #{det_num}')

        ax.set_xlabel('Event Number')
        ax.set_ylabel("Max ADC")
        ax.set_title(f"Event Sample Max Detector #{det_num}")


def identify_noise(max_data, noise_threshold=100):
    """
    Identify noise events in max data based on threshold.
    :param max_data:
    :param noise_threshold: If number, use as threshold for all detectors.
                            If 1D array, use as threshold for each detector.
                            If 2D array, use as threshold for each strip of each detector.
    :return: Boolean array of shape (n_events,) where True means event is noise.
    """
    moving_averages = [3, 4, 5]
    noise_strips = max_data < noise_threshold  # Compare strip maxima with the threshold
    noise_mask = np.all(noise_strips, axis=2)  # Mark event as noise if all strips on all detectors below threshold

    if isinstance(noise_threshold, np.ndarray) and noise_threshold.ndim == 2:
        for pnts in moving_averages:
            max_data_mv_avg = np.apply_along_axis(np.convolve, -1, max_data, np.ones(pnts) / pnts, mode='same')
            noise_strips_mv_avg = max_data_mv_avg < noise_threshold / np.sqrt(pnts)  # Compare strip maxima with the threshold
            noise_mask_mv_avg = np.all(noise_strips_mv_avg, axis=2)  # Mark event as noise if all strips on all detectors below threshold
            noise_mask = noise_mask | noise_mask_mv_avg
        # neg_mask = -np.min(max_data, axis=2) > np.max(max_data, axis=2) / 3  # Mark event as noise if large negative
        # noise_mask = noise_mask | neg_mask
    noise_mask = np.any(noise_mask, axis=1)  # Mark event as noise either detector below threshold

    # print(f'max_data shape: {max_data.shape}, noise_threshold shape: {noise_threshold.shape}, '
    #       f'noise_strips shape: {noise_strips.shape}, noise_mask shape: {noise_mask.shape}')

    return noise_mask


def suppress_noise(data, noise_mask):
    data = data[~noise_mask]

    return data


def identify_common_signal(max_data, signal_threshold=400):
    """
    Identify common signal in max data.
    :param max_data:
    :param signal_threshold:
    :return:
    """
    det_max = np.max(max_data, axis=2)  # Get the maximum strip for each detector for each event
    pair_min = np.min(det_max, axis=1)  # Get the minimum detector for the strip maxima for each event
    signal_mask = pair_min > signal_threshold  # Only select events where each detector has a strip above threshold

    return signal_mask


def select_signal(data, signal_mask):
    data = data[signal_mask]

    return data


def get_max_strip(data_max):
    return np.argmax(data_max, axis=2)


def get_strip_max(data_max):
    return np.max(data_max, axis=2)


def identify_spark(data_max, threshold_sigma=10, spark_thresholds=None):
    """
    Filter out events with sparking.
    :param data_max: Max strip ADC for each detector for each event.
    :param threshold_sigma: Number of standard deviations above the mean to set the threshold.
    :param spark_thresholds: If given, use as threshold for each detector.
    :return: Boolean array of shape (n_events,) where True means event is possible spark.
    """

    det_event_avgs = np.mean(data_max, axis=2)  # Average over strips for each detector for each event
    if spark_thresholds is None:  # Calculate thresholds if not given
        det_std = np.std(det_event_avgs, axis=0)  # Standard deviation of strip average for each detector over all events
        det_avg = np.mean(det_event_avgs, axis=0)  # Average of strip average for each detector over all events
        spark_thresholds = det_std * threshold_sigma + det_avg  # Threshold for each detector
    spark_mask = det_event_avgs > spark_thresholds  # Select events where detector strip average above threshold
    spark_mask = np.any(spark_mask, axis=1)  # Select events where at least one detector strip average above threshold

    return spark_mask, spark_thresholds


def plot_spark_metric(data_max, thresholds):
    det_avg = np.transpose(np.mean(data_max, axis=2))
    fig, ax = plt.subplots()
    ax.axhline(0, color='black', zorder=0)
    for det_num, det in enumerate(det_avg):
        scatter = ax.scatter(range(len(det)), det, label=f'Detector #{det_num}')
        color = scatter.get_facecolor()[0]
        ax.axhline(thresholds[det_num], ls='--', color=color, label=f'Det #{det_num} Spark Threshold')
    ax.set_xlabel('Event #')
    ax.set_ylabel('Detector Strip Averaged ADC')
    ax.set_title('High Noise Metric')
    ax.legend()
    fig.tight_layout()


def identify_negatives(data_max):
    """
    Filter out events with negative strip averages.
    :param data_max: Max strip ADC for each detector for each event.
    :return: Boolean array of shape (n_events,) where True means event has a negative ADC detector.
    """
    data_max = data_max.reshape((data_max.shape[0], data_max.shape[1] // 2, data_max.shape[2] * 2))
    det_avg = np.mean(data_max, axis=2)  # Average over strips for each detector for each event
    neg_mask = det_avg < 0  # Select events where detector strip average is negative
    neg_mask = np.any(neg_mask, axis=1)  # Select events where at least one detector strip average is negative

    return neg_mask


def process_chunk(chunk, pedestals, noise_thresholds, num_detectors, connected_channels=None):
    data = read_det_data_chunk(chunk['StripAmpl'], num_detectors)
    data = data.astype(float)
    if connected_channels is not None:  # Nan out disconnected channels
        connected_mask = connected_channels.astype(float)
        connected_mask[~connected_channels] = np.nan
        data = data * connected_mask[np.newaxis, :, :, np.newaxis]
    common_noise = get_common_noise(data, pedestals)
    ped_sub_data = subtract_pedestal(data, pedestals)
    ped_com_sub_data = ped_sub_data - common_noise[:, :, np.newaxis, :]
    max_data = get_sample_max(ped_com_sub_data)
    max_medians = np.nanmedian(max_data, axis=(0, 2))
    max_data = max_data - max_medians[np.newaxis, :, np.newaxis]
    ped_com_sub_data = ped_com_sub_data - max_medians[np.newaxis, :, np.newaxis, np.newaxis]
    max_data = max_data.reshape((max_data.shape[0], max_data.shape[1] // 2, max_data.shape[2] * 2))  # For URWs
    noise_thresholds = noise_thresholds.reshape((noise_thresholds.shape[0] // 2, noise_thresholds.shape[1] * 2))
    max_data = np.nan_to_num(max_data)
    noise_mask = identify_noise(max_data, noise_threshold=noise_thresholds)
    data_no_noise = np.nan_to_num(ped_com_sub_data[~noise_mask])  # Convert disconnected nan channels to 0
    event_numbers = np.arange(data.shape[0])[~noise_mask]
    total_events = data.shape[0]
    # data_no_noise = suppress_noise(ped_com_sub_data, noise_mask)
    # signal_mask = identify_common_signal(max_data, signal_threshold=400)
    # data_signal = select_signal(data, signal_mask)

    return data_no_noise, event_numbers, total_events


def process_chunk_all(chunk, pedestals, num_detectors):
    data = read_det_data_chunk(chunk['StripAmpl'], num_detectors)
    ped_com_sub_data = data  # Hack
    # common_noise = get_common_noise(data, pedestals)
    # ped_sub_data = subtract_pedestal(data, pedestals)
    # ped_com_sub_data = ped_sub_data - common_noise[:, :, np.newaxis, :]

    return ped_com_sub_data


# def process_file(file_path, pedestals, noise_thresholds, num_detectors, connected_channels=None, chunk_size=10000,
#                  filer_noise_events=True):
#     with uproot.open(file_path) as file:
#         tree_names = file.keys()
#
#         events, event_numbers = [], []
#         total_events = 0
#         for tree_name in tree_names:
#             tree = file[tree_name]
#             for chunk in uproot.iterate(tree, branches=['StripAmpl'], step_size=chunk_size):
#                 if filer_noise_events:
#                     chunk_events, event_nums, num_events = process_chunk(chunk, pedestals, noise_thresholds,
#                                                                          num_detectors, connected_channels)
#                     event_numbers.append(event_nums + total_events)
#                     total_events += num_events
#                 else:
#                     chunk_events = process_chunk_all(chunk, pedestals, num_detectors)
#                 events.append(chunk_events)
#
#     return events, event_numbers, total_events


def process_file(file_path, pedestals, noise_thresholds, num_detectors, connected_channels=None, chunk_size=10000,
                 filer_noise_events=True):
    with uproot.open(file_path) as file:
        tree_names = file.keys()
        tree_name = 'T;17' if 'T;17' in tree_names else tree_names[0]
        if tree_name != 'T;17':
            print(f'Warning: Tree name T;17 not found in {tree_names}. \nUsing {tree_name} for {file_path}')

        events, event_numbers = [], []
        total_events = 0
        tree = file[tree_name]
        for chunk in uproot.iterate(tree, branches=['StripAmpl'], step_size=chunk_size):
            if filer_noise_events:
                chunk_events, event_nums, num_events = process_chunk(chunk, pedestals, noise_thresholds,
                                                                     num_detectors, connected_channels)
                event_numbers.append(event_nums + total_events)
                total_events += num_events
            else:
                chunk_events = process_chunk_all(chunk, pedestals, num_detectors)
            events.append(chunk_events)

    return np.concatenate(events, axis=0), np.concatenate(event_numbers, axis=0), total_events


def plot_raw_fe_peak(signal_events_max_sum, bins=50, fill_between_x=None):
    cut_percentile = 99
    percentile_mask = signal_events_max_sum < np.percentile(signal_events_max_sum, cut_percentile)

    all_bins = int(bins * (np.max(signal_events_max_sum) / np.max(signal_events_max_sum[percentile_mask])))
    hist_all, bin_edges_all = np.histogram(signal_events_max_sum, bins=all_bins)
    bin_centers_all = (bin_edges_all[1:] + bin_edges_all[:-1]) / 2
    bin_width_all = bin_edges_all[1] - bin_edges_all[0]

    fig_all, ax_all = plt.subplots()
    ax_all.bar(bin_centers_all, hist_all, width=bin_width_all, color='gray', edgecolor=None, align='center')
    if fill_between_x:
        ax_all.fill_between(fill_between_x, 0, np.max(hist_all) * 1.2, color='gray', alpha=0.2)
    ax_all.set_ylim(bottom=0, top=np.max(hist_all) * 1.2)
    ax_all.set_xlabel('ADC')
    ax_all.set_ylabel('Events')
    ax_all.set_title('Sum of Strip Max ADCs for Pure Signal Events No Percentile Cut')
    fig_all.tight_layout()

    hist, bin_edges = np.histogram(signal_events_max_sum[percentile_mask], bins=bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots()
    ax.bar(bin_centers, hist, width=bin_width, color='blue', edgecolor='black', align='center')
    ax.set_ylim(bottom=0, top=np.max(hist) * 1.2)
    ax.set_xlabel('ADC')
    ax.set_ylabel('Events')
    ax.set_title('Sum of Strip Max ADCs for Pure Signal Events')
    fig.tight_layout()


def fit_fe_peak(signal_events_max_sum, n_bin_vals=None, plot=False, plot_final=False, save_fit_path=None):
    if n_bin_vals is None:
        n_bin_vals = [50, 80, 100, 120, 200, 65]
    percentile_cutoffs, cut_percentile, nsigma_max, nsigma_min, iterations = [40, 97], 98, 3, 3, 3
    background, bkg_pars = poly2_center, 2
    bkg_lower_bounds, bkg_upper_bounds = (-100, -10), (10, 10)
    fit_func = lambda x, a_, mu_, sigma_, *b: gaussian(x, a_, mu_, sigma_) + background(x, mu_, *b)

    percentile_mask = signal_events_max_sum < np.percentile(signal_events_max_sum, cut_percentile)
    signal_events_max_sum = signal_events_max_sum[percentile_mask]
    low_percentile, high_percentile = np.percentile(signal_events_max_sum, percentile_cutoffs)

    y_min, y_max = np.max(signal_events_max_sum), np.min(signal_events_max_sum)
    x_range = y_max - y_min
    # low_cut, high_cut = y_min + percent_cutoffs[0] * x_range, y_max - percent_cutoffs[1] * x_range
    peak_region_mask = (signal_events_max_sum > low_percentile) & (signal_events_max_sum < high_percentile)
    n_bin_vals.append(np.size(signal_events_max_sum[peak_region_mask]) // 10)
    if len(n_bin_vals) == 2:
        n_bin_vals.pop(0)  # If analysis code only run calculated n_bins

    if plot:
        fig_means, ax_means = plt.subplots()
        ax_means.set_ylabel('Mean')
        ax_means.set_xlabel('Iteration')
        ax_means.set_title('Fit Iteration Means')

        fig_sigmas, ax_sigmas = plt.subplots()
        ax_sigmas.set_ylabel('Sigma')
        ax_sigmas.set_xlabel('Iteration')
        ax_sigmas.set_title('Fit Iteration Sigmas')

        fig_events, ax_events = plt.subplots()
        ax_events.set_ylabel('Events')
        ax_events.set_xlabel('Iteration')
        ax_events.set_title('Fit Iteration Events')

        x_plot = np.linspace(min(signal_events_max_sum), max(signal_events_max_sum), 1000)

    for n_bins in n_bin_vals:
        hist, bin_edges = np.histogram(signal_events_max_sum, bins=n_bins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        x_range = bin_edges[-1] - bin_edges[0]
        # low_range, high_range = percent_cutoffs[0] * x_range, percent_cutoffs[0] * x_range
        # lower_bin_cut, upper_bin_cut = bin_edges[0] + low_range, bin_edges[-1] - high_range

        # mask = (bin_centers > lower_bin_cut) & (bin_centers < upper_bin_cut)
        mask = (bin_centers > low_percentile) & (bin_centers < high_percentile)
        x_fit, y_fit = bin_centers[mask], hist[mask]
        mu = Measure(np.average(x_fit, weights=y_fit), 0)  # Average weighted by hist
        sigma = Measure(np.sqrt(np.average((x_fit - mu.val) ** 2, weights=y_fit)), 0)  # Weighted std
        num_events, a = Measure(0, 0), Measure(0, 0)

        mus, mu_errs, sigmas, sigma_errs, events, event_errs = [], [], [], [], [], []

        for i, nsigma in enumerate(np.linspace(nsigma_max, nsigma_min, iterations)):
            fit_range = [mu.val - nsigma * sigma.val, mu.val + nsigma * sigma.val]
            mask = (bin_centers > fit_range[0]) & (bin_centers < fit_range[1])
            x_fit, y_fit = bin_centers[mask], hist[mask]
            err = np.where(y_fit == 0, 1, np.sqrt(y_fit))  # Set error to 1 if hist is 0

            p0 = [0.9 * np.max(y_fit), mu.val, sigma.val, *([0] * bkg_pars)]
            lower_bounds = [0.3 * np.max(y_fit), low_percentile, 0.1 * sigma.val, *bkg_lower_bounds]
            upper_bounds = [1.6 * np.max(y_fit), high_percentile, 1.5 * sigma.val, *bkg_upper_bounds]
            bounds = (lower_bounds, upper_bounds)

            popt, pcov = cf(fit_func, x_fit, y_fit, sigma=err, p0=p0, absolute_sigma=True, bounds=bounds)
            perr = np.sqrt(np.diag(pcov))

            # if not 0 < popt[0] < max_y or not lower_bin_cut < popt[1] < upper_bin_cut or not 0 < popt[2] < upper_bin_cut:
            #     print(f'Bad Fitting. Quitting.')
            #     break

            a, mu, sigma, *bkg = [Measure(val, err) for val, err in zip(popt, perr)]
            num_events = a / bin_width * sigma * np.sqrt(2 * np.pi)
            if plot:
                mus.append(mu.val)
                mu_errs.append(mu.err)
                sigmas.append(sigma.val)
                sigma_errs.append(sigma.err)
                events.append(num_events.val)
                event_errs.append(num_events.err)

                fig, ax = plt.subplots()
                ax.bar(bin_centers, hist, width=bin_width, color='gray', align='center', label='Data')
                ax.bar(x_fit, y_fit, width=x_fit[1] - x_fit[0], color='blue', align='center', label='Fit Data')
                ax.plot(x_plot, fit_func(x_plot, *p0), color='gray', ls=':', label='Initial Guess')
                ax.plot(x_plot, fit_func(x_plot, *popt), color='red', label='Fit')
                ax.plot(x_plot, gaussian(x_plot, *popt[:3]), color='green', label='Gaussian')
                ax.plot(x_plot, background(x_plot, popt[1], *popt[-bkg_pars:]), color='orange', label='Background')
                ax.set_ylim(bottom=0, top=np.max(hist) * 1.2)
                ax.legend()
                ax.set_title(f'{n_bins} Bins Fit Iteration #{i}')
                print(f'{n_bins} Bins Fit Iteration #{i} Fit: a={a}, events={num_events}, mu={mu}, sigma={sigma}, ' +
                      ', '.join([f'bkg_{j + 1}={bkg_j}' for j, bkg_j in enumerate(bkg)]))

        if plot:
            ax_means.errorbar(range(len(mus)), mus, yerr=mu_errs, marker='o', alpha=0.6, label=f'{n_bins} Bins')
            ax_sigmas.errorbar(range(len(sigmas)), sigmas, yerr=sigma_errs, marker='o', alpha=0.6,
                               label=f'{n_bins} Bins')
            ax_events.errorbar(range(len(events)), events, yerr=event_errs, marker='o', alpha=0.6,
                               label=f'{n_bins} Bins')

    if plot:
        ax_means.legend()
        ax_means.grid()
        ax_sigmas.legend()
        ax_sigmas.grid()
        ax_events.legend()
        ax_events.grid()
        fig_means.tight_layout()
        fig_sigmas.tight_layout()
        fig_events.tight_layout()

    if plot_final or save_fit_path is not None:
        fig, ax = plt.subplots()
        ax.bar(bin_centers, hist, width=bin_width, edgecolor='black', align='center')
        fit_lab = (fr'A = {a}' + '\n' + rf'$\mu$ = {mu}' + '\n' + rf'$\sigma$ = {sigma}' +
                   '\n' + f'Events = {num_events}')
        fit_len = fit_range[1] - fit_range[0]
        x_plot = np.linspace(fit_range[0] - 0.2 * fit_len, fit_range[1] + 0.2 * fit_len, 1000)
        ax.plot(x_plot, fit_func(x_plot, *popt), color='red', label='Fit')
        ax.plot(x_plot, gaussian(x_plot, *popt[:3]), color='pink', label='Gaussian')
        ax.plot(x_plot, background(x_plot, popt[1], *popt[-bkg_pars:]), color='orange', label='Background')
        ax.annotate(fit_lab, xy=(0.65, 0.65), xycoords='axes fraction', ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2),
                    fontsize=12, color='black')
        ax.axvline(fit_range[0], ls='--', color='black', label='Fit Range')
        ax.axvline(fit_range[1], ls='--', color='black')
        ax.set_ylim(bottom=0, top=np.max(hist) * 1.2)
        ax.legend()
        ax.set_xlabel('ADC')
        ax.set_ylabel('Events')
        ax.set_title('Sum of Strip Max ADCs for Pure Signal Events')
        fig.tight_layout()
        if save_fit_path is not None:
            fig.savefig(save_fit_path + '.png')
            fig.savefig(save_fit_path + '.pdf')

    return mu, sigma, num_events


def fit_fe_peak2(signal_events_max_sum, bins=50, title='Test', plot=False, plot_raw=False, final_plot=False):
    noise_bins = 200
    noise_range = np.mean(signal_events_max_sum) * 2.5
    noise_events = signal_events_max_sum[signal_events_max_sum < noise_range]
    hist_noise, bin_edges_noise = np.histogram(noise_events, bins=noise_bins)
    bin_centers_noise = (bin_edges_noise[1:] + bin_edges_noise[:-1]) / 2
    bin_width_noise = bin_edges_noise[1] - bin_edges_noise[0]

    p0 = [np.max(hist_noise) * 0.9, np.mean(noise_events), np.std(noise_events) * 0.8]
    y_err = np.where(hist_noise == 0, 1, np.sqrt(hist_noise))
    popt_noise, pcov_noise = cf(gaussian, bin_centers_noise, hist_noise, p0=p0, sigma=y_err, absolute_sigma=True)
    x_plot = np.linspace(bin_edges_noise[0], bin_edges_noise[-1], 1000)

    hist_all, bin_edges_all = np.histogram(signal_events_max_sum, bins=noise_bins)
    bin_centers_all = (bin_edges_all[1:] + bin_edges_all[:-1]) / 2

    # noise_cut = popt_noise[1] + popt_noise[2] * 15
    closest_bin_index = np.abs(bin_centers_all - popt_noise[1]).argmin()  # Find the bin closest to start_x
    # Walk to the right and find the first bin with count zero
    for i in range(closest_bin_index, len(hist_all)):
        if hist_all[i] == 0:
            # The x corresponding to this bin is:
            noise_cut = bin_centers_all[i]
            break
    else:
        print("No bin with zero count found to the right of start_x.")

    if plot_raw:
        above_noise_events = signal_events_max_sum[signal_events_max_sum >= noise_range]
        hist_above_noise, bin_edges_above_noise = np.histogram(above_noise_events, bins=noise_bins)
        bin_centers_above_noise = (bin_edges_above_noise[1:] + bin_edges_above_noise[:-1]) / 2
        bin_width_above_noise = bin_edges_above_noise[1] - bin_edges_above_noise[0]

        fig_noise, ax_noise = plt.subplots()
        ax_noise.bar(bin_centers_noise, hist_noise, width=bin_width_noise, color='gray', edgecolor=None, align='center')
        ax_noise.bar(bin_centers_above_noise, hist_above_noise, width=bin_width_above_noise, color='green', edgecolor=None, align='center')
        ax_noise.plot(x_plot, gaussian(x_plot, *p0), color='gray', label='Guess')
        ax_noise.plot(x_plot, gaussian(x_plot, *popt_noise), color='red', label='Fit')
        ax_noise.axvline(noise_cut, ls='--', color='black', label='Noise Cut')
        ax_noise.set_xlabel('ADC')
        ax_noise.set_ylabel('Events')
        ax_noise.set_yscale('log')
        ax_noise.legend()
        ax_noise.set_ylim(bottom=0.5, top=np.max(hist_noise) * 1.2)
        ax_noise.set_title(f'{title} Noise Fit')
        fig_noise.tight_layout()

    signal_events = signal_events_max_sum[signal_events_max_sum > noise_cut]
    if len(signal_events) <= 10:
        print('Not enough signal events found. Returning.')
        return

    if plot:
        hist, bin_edges = np.histogram(signal_events, bins=bins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        fig, ax = plt.subplots()
        ax.bar(bin_centers, hist, width=bin_width, color='gray', edgecolor=None, align='center', alpha=0.8)
        ax.set_ylim(bottom=0, top=np.max(hist) * 1.2)
        ax.set_xlabel('ADC')
        ax.set_ylabel('Events')
        x_plot = np.linspace(bin_edges[0], bin_edges[-1], 1000)

    fit_func = lambda x, a, mu, sigma, b, c: gaussian(x, a, mu, sigma) + b * (x - mu) + c
    iterations = 3
    colors = ['orange', 'green', 'blue', 'red', 'purple']
    cut_percentile = 100
    percentile_mask = signal_events < np.percentile(signal_events, cut_percentile)
    hist, bin_edges = np.histogram(signal_events[percentile_mask], bins=bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    p0 = [np.max(hist) * 0.9, np.mean(signal_events[percentile_mask]), np.std(signal_events[percentile_mask]) * 0.8, 0, 0]
    lower_bounds = [0.3 * np.max(hist), bin_edges[0], 0.1 * p0[2], -100, -10]
    upper_bounds = [1.6 * np.max(hist), bin_edges[-1], 1.5 * p0[2], 100, 10]
    bounds = (lower_bounds, upper_bounds)
    y_err = np.where(hist == 0, 1, np.sqrt(hist))
    if plot:
        ax.plot(x_plot, fit_func(x_plot, *p0), color='gray', label='Guess')

    popt, perr = p0, np.array(upper_bounds) - np.array(lower_bounds)
    for i, color in zip(range(iterations), colors[:iterations]):
        try:
            popt, pcov = cf(fit_func, bin_centers, hist, p0=p0, sigma=y_err, absolute_sigma=True, bounds=bounds)
            perr = np.sqrt(np.diag(pcov))
            if plot:
                ax.bar(bin_centers, hist, width=bin_width, color=color, edgecolor='black', align='center', alpha=0.5)
                ax.plot(x_plot, fit_func(x_plot, *popt), color=color, label=f'Fit #{i + 1}')
            if i == iterations - 1:
                break
        except RuntimeError:
            print('Fit failed. Returning.')
            break
        p0 = [*popt[:3], 0, 0]
        sigmas = 4 - i
        low_cut, high_cut = popt[1] - popt[2] * sigmas, popt[1] + popt[2] * sigmas
        # print(f'Fit #{i + 1} cut: {low_cut}, {high_cut}')
        hist, bin_edges = np.histogram(signal_events[(signal_events > low_cut) & (signal_events < high_cut)], bins=bins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        y_err = np.where(hist == 0, 1, np.sqrt(hist))

    if plot:
        ax.legend()
        ax.set_title(f'{title} Iterative Fits')
        fig.tight_layout()

    # fit_func = lambda x, a, mu, sigma, b: gaussian(x, a, mu, sigma) + b * (x - mu)
    p0 = [*popt[:3]]
    lower_bounds = [0.3 * np.max(hist), bin_edges[0], 0.1 * p0[2]]
    upper_bounds = [1.6 * np.max(hist), bin_edges[-1], 1.5 * p0[2]]
    bounds = (lower_bounds, upper_bounds)
    sigmas = 5
    low_cut, high_cut = popt[1] - popt[2] * sigmas, popt[1] + popt[2] * sigmas
    signal_events = signal_events[(signal_events > low_cut) & (signal_events < high_cut)]
    n_bins = max(len(signal_events) // 10, 5)
    hist, bin_edges = np.histogram(signal_events[(signal_events > low_cut) & (signal_events < high_cut)], bins=n_bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    y_err = np.where(hist == 0, 1, np.sqrt(hist))
    try:
        popt, pcov = cf(gaussian, bin_centers, hist, p0=p0, sigma=y_err, absolute_sigma=True, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
    except:
        print('Final fit failed. Using previous fit.')

    if final_plot:
        fig, ax = plt.subplots()
        ax.bar(bin_centers, hist, width=bin_width, edgecolor='black', align='center')
        ax.set_ylim(bottom=0, top=np.max(hist) * 1.2)
        x_plot = np.linspace(bin_edges[0], bin_edges[-1], 1000)
        ax.plot(x_plot, gaussian(x_plot, *popt[:3]), color='red', label='Fit')
        a, mu, sigma = [Measure(val, err) for val, err in zip(popt[:3], perr[:3])]
        fit_label = (fr'A = {a}' + '\n' + rf'$\mu$ = {mu}' + '\n' + rf'$\sigma$ = {sigma}')
        ax.annotate(fit_label, xy=(0.05, 0.05), xycoords='axes fraction', ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2),
                    fontsize=12, color='black')
        ax.set_xlabel('ADC')
        ax.set_ylabel('Events')
        ax.set_title(f'{title} Final Fit')
        ax.legend()
        fig.tight_layout()

    return Measure(popt[1], perr[1])


def plot_spectrum_noise_fit(signal_events_max_sum, bins=50, title='Test'):
    noise_bins = 200
    noise_range = np.mean(signal_events_max_sum) * 2.5
    noise_events = signal_events_max_sum[signal_events_max_sum < noise_range]
    hist_noise, bin_edges_noise = np.histogram(noise_events, bins=noise_bins)
    bin_centers_noise = (bin_edges_noise[1:] + bin_edges_noise[:-1]) / 2
    bin_width_noise = bin_edges_noise[1] - bin_edges_noise[0]

    p0 = [np.max(hist_noise) * 0.9, np.mean(noise_events), np.std(noise_events) * 0.8]
    y_err = np.where(hist_noise == 0, 1, np.sqrt(hist_noise))
    popt_noise, pcov_noise = cf(gaussian, bin_centers_noise, hist_noise, p0=p0, sigma=y_err, absolute_sigma=True)
    x_plot = np.linspace(bin_edges_noise[0], bin_edges_noise[-1], 1000)

    above_noise_events = signal_events_max_sum[signal_events_max_sum >= noise_range]
    hist_above_noise, bin_edges_above_noise = np.histogram(above_noise_events, bins=noise_bins)
    bin_centers_above_noise = (bin_edges_above_noise[1:] + bin_edges_above_noise[:-1]) / 2
    bin_width_above_noise = bin_edges_above_noise[1] - bin_edges_above_noise[0]

    fig_noise, ax_noise = plt.subplots()
    ax_noise.bar(bin_centers_noise, hist_noise, width=bin_width_noise, color='gray', edgecolor=None, align='center')
    ax_noise.bar(bin_centers_above_noise, hist_above_noise, width=bin_width_above_noise, color='green', edgecolor=None, align='center')
    ax_noise.plot(x_plot, gaussian(x_plot, *p0), color='gray', label='Guess')
    ax_noise.plot(x_plot, gaussian(x_plot, *popt_noise), color='red', label='Fit')
    ax_noise.set_xlabel('ADC')
    ax_noise.set_ylabel('Events')
    ax_noise.set_yscale('log')
    ax_noise.legend()
    ax_noise.set_ylim(bottom=0.5, top=np.max(hist_noise) * 1.2)
    ax_noise.set_title(f'{title} Noise Fit')
    fig_noise.tight_layout()


def analyze_file_qa(file_path, pedestals, noise_thresholds, num_detectors, connected_channels=None, edge_strips=None,
                    chunk_size=10000):
    # all_events = process_file(file_path, pedestals, noise_thresholds, num_detectors, connected_channels, chunk_size,
    #                           filer_noise_events=False)
    no_noise_events, event_numbers, total_events = process_file(file_path, pedestals, noise_thresholds, num_detectors,
                                                                connected_channels, chunk_size)
    print(f'Total events: {total_events}')

    # all_events = np.concatenate(all_events, axis=0)
    # all_events = all_events * connected_channels[np.newaxis, :, :, np.newaxis]
    # common_noise = get_common_noise(all_events, pedestals)
    # ped_sub_data = subtract_pedestal(all_events, pedestals)
    # ped_com_sub_data = ped_sub_data - common_noise[:, :, np.newaxis, :]
    # max_data = get_sample_max(ped_com_sub_data)
    # noise_mask = identify_noise(max_data, noise_threshold=noise_thresholds)

    print(f'no_noise_events.shape: {no_noise_events.shape}')

    # get_connected_channels(no_noise_events, noise_thresholds)  # Figure out which channels are connected
    max_edge_events_mask = get_max_edge_events(no_noise_events, edge_strips=edge_strips)
    max_edge_events = no_noise_events[max_edge_events_mask]
    max_edge_events_max = get_sample_max(max_edge_events)
    if len(max_edge_events) > 0:
        plot_combined_time_series(max_edge_events, max_events=10)
        plot_urw_position(max_edge_events_max, separate_event_plots=True, thresholds=noise_thresholds, max_events=10)

    flat_signals_mask = get_flat_signals(no_noise_events)
    print(flat_signals_mask)
    flat_signal_events = no_noise_events[flat_signals_mask]
    flat_signal_events_max = get_sample_max(flat_signal_events)
    if len(flat_signal_events) > 0:
        plot_combined_time_series(flat_signal_events, max_events=10)
        plot_urw_position(flat_signal_events_max, separate_event_plots=True, thresholds=noise_thresholds, max_events=10)

    no_noise_events_max = get_sample_max(no_noise_events)
    print(f'no_noise_events_max.shape: {no_noise_events_max.shape}')
    no_noise_events_adcs = no_noise_events_max.reshape(-1, num_detectors)
    print(f'no_noise_events_adcs.shape: {no_noise_events_adcs.shape}')
    plot_1d_sample_max_hist(no_noise_events_adcs, log=True, title='Max Sample per Strip ADC Spectrum No Noise Events')
    no_noise_events_max_strip = get_max_strip(no_noise_events_max)
    print(f'no_noise_events_max_strip.shape: {no_noise_events_max_strip.shape}')
    bins = np.arange(np.min(no_noise_events_max_strip) - 0.5, np.max(no_noise_events_max_strip) + 1.5, 1)
    plot_1d_sample_max_hist(no_noise_events_max_strip, bins=bins, title='Max Strip Number No Noise Events',
                            xlabel='Strip Number')
    no_noise_events_strip_max = get_strip_max(no_noise_events_max)
    print(f'no_noise_events_strip_max.shape: {no_noise_events_strip_max.shape}')
    plot_1d_sample_max_hist(no_noise_events_strip_max, log=True,
                            title='Max Sample Max Strip ADC Spectrum No Noise Events')

    spark_mask, spark_thresholds = identify_spark(no_noise_events_max, threshold_sigma=10)
    # all_data_spark_mask, spark_thresholds = identify_spark(max_data, spark_thresholds=spark_thresholds)
    plot_spark_metric(no_noise_events_max, spark_thresholds)
    # plot_spark_metric(max_data, spark_thresholds)
    neg_mask = identify_negatives(no_noise_events_max)
    print(f'spark_mask.shape: {spark_mask.shape}')
    spark_events = no_noise_events[spark_mask]
    print(f'spark_events.shape: {spark_events.shape}')
    # pure_signal_events_max = no_noise_events_max[(~spark_mask) & (~neg_mask)]
    pure_signal_events_max = no_noise_events_max[~neg_mask]
    print(f'pure_signal_events_max.shape: {pure_signal_events_max.shape}')

    if len(no_noise_events[neg_mask]) > 0:
        plot_combined_time_series(no_noise_events[neg_mask], max_events=10)
        plot_urw_position(no_noise_events_max[neg_mask], separate_event_plots=True, thresholds=noise_thresholds, max_events=10)

    signal_events_max_sum = np.sum(pure_signal_events_max, axis=(1, 2))
    print(f'signal_events_max_sum.shape: {signal_events_max_sum.shape}')
    plot_raw_fe_peak(signal_events_max_sum, bins=30)
    # fit_fe_peak2(signal_events_max_sum, bins=30)
    # mu, sigma, events = fit_fe_peak(signal_events_max_sum, plot_final=True)

    no_noise_max_sum = np.sum(no_noise_events_max, axis=(1, 2))
    # super_cut_events = no_noise_events[no_noise_max_sum > 4500]
    super_cut_mask = (no_noise_max_sum > 0) & (no_noise_max_sum < 500)
    super_cut_events = no_noise_events[super_cut_mask]
    super_cut_event_nums = event_numbers[super_cut_mask]
    print(f'super_cut_events.shape: {super_cut_events.shape}')
    super_cut_event_max = get_sample_max(super_cut_events)
    # plot_position_data(super_cut_event_max, event_nums=None)
    plot_urw_position(super_cut_event_max, separate_event_plots=True, thresholds=noise_thresholds, max_events=10, plot_avgs=False)

    # channel_sum = np.sum(no_noise_events_max, axis=0)
    # plot_p2_2d(channel_sum)


def analyze_spectra(file_path, pedestals, noise_thresholds, num_detectors, connected_channels=None, chunk_size=10000,
                    title='Test', save_path=None):
    # all_events = process_file(file_path, pedestals, noise_thresholds, num_detectors, connected_channels, chunk_size,
    #                           filer_noise_events=False)
    no_noise_events, event_numbers, total_events = process_file(file_path, pedestals, noise_thresholds, num_detectors,
                                                                connected_channels, chunk_size)
    print(f'{title} total events: {total_events}')
    no_noise_events_max = get_sample_max(no_noise_events)
    no_noise_events_max_sum = np.sum(no_noise_events_max, axis=(1, 2))

    # peak_mu = fit_fe_peak2(no_noise_events_max_sum, bins=30, title=title, plot=False, plot_raw=True, final_plot=True)
    plot_spectrum(no_noise_events_max_sum, bins=30, title=title, save_path=save_path)
    peak_mu = Measure(0, 0)

    return peak_mu


def analyze_file(file_path, pedestals, noise_thresholds, num_detectors, connected_channels, chunk_size=10000,
                 out_dir=None):
    events, event_numbers, total_events = process_file(file_path, pedestals, noise_thresholds, num_detectors,
                                                       connected_channels, chunk_size)

    no_noise_events = events

    no_noise_events_max = get_sample_max(no_noise_events)

    spark_mask, spark_thresholds = identify_spark(no_noise_events_max, threshold_sigma=10)
    neg_mask = identify_negatives(no_noise_events_max)
    pure_signal_events_max = no_noise_events_max[(~spark_mask) & (~neg_mask)]

    signal_events_max_sum = np.sum(pure_signal_events_max, axis=(1, 2))
    if len(signal_events_max_sum) < 500:
        print(f'Warning: Less than 500 signal events found in {file_path}.')
        return None
    out_fig_path = f'{out_dir}{os.path.basename(file_path).split(".")[0]}_fe_peak_fit' if out_dir is not None else None
    try:
        mu, sigma, events = fit_fe_peak(signal_events_max_sum, n_bin_vals=[75], save_fit_path=out_fig_path)
    except:
        print(f'Warning: Fit failed for {file_path}.')
        return None

    mesh_voltage, drift_voltage, run_date = interpret_file_name(file_path)

    return {'mu': mu, 'sigma': sigma, 'events': events,
            'mesh_voltage': mesh_voltage, 'drift_voltage': drift_voltage, 'run_date': run_date}


def peak_analysis(file_data, run_periods):
    drift_minus_mesh = 600

    df = pd.DataFrame(file_data)

    # Function to determine the run period for a given date
    def find_run_period(date):
        for index, run_periods_row in run_periods.iterrows():
            if run_periods_row['start_date'] <= date <= run_periods_row['end_date']:
                return index
        return None

    def split_measures(measure, col_name):
        return pd.Series({f'{col_name}_val': measure.val, f'{col_name}_err': measure.err})

    # Apply the function to create a new column 'run_period' in the dataframe
    df['run_period'] = df['run_date'].apply(find_run_period)
    df = df[df['drift_voltage'] - df['mesh_voltage'] == drift_minus_mesh]

    # Split measure columns into separate columns for val and err
    for column in ['mu', 'sigma', 'events']:
        df = df.join(df[column].apply(split_measures, col_name=column))

    print(df)

    run_period_event_avgs = df.groupby('run_period')['events_val'].mean().sort_values(ascending=False)

    fig_mus, ax_mus = plt.subplots()
    ax_mus.set_xlabel('Mesh Voltage (V)')
    ax_mus.set_ylabel('Peak ADC')
    for run_i in run_period_event_avgs.index:
        df_i = df[df['run_period'] == run_i]
        ax_mus.errorbar(df_i['mesh_voltage'], df_i['mu_val'], yerr=df_i['mu_err'], marker='o', alpha=0.6,
                        label=f'Run #{run_i} {np.average(df_i["events"])} Mean Events')
    ax_mus.legend()
    ax_mus.grid()
    fig_mus.tight_layout()

    fig_mus_vs_events, ax_mus_vs_events = plt.subplots()
    fig_mus_dev_vs_events, ax_mus_dev_vs_events = plt.subplots()
    ax_mus_vs_events.set_xlabel('Number of Fe Events')
    ax_mus_vs_events.set_ylabel('Peak ADC')
    ax_mus_dev_vs_events.set_xlabel('Number of Fe Events')
    ax_mus_dev_vs_events.set_ylabel('Percent Deviation of Peak ADC from Average')
    ax_mus_dev_vs_events.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_mus_dev_vs_events.axhline(0, ls='-', color='black')
    for mesh_v in sorted(pd.unique(df['mesh_voltage'])):
        df_mesh_v = df[df['mesh_voltage'] == mesh_v].sort_values(by='events_val')
        ax_mus_vs_events.errorbar(df_mesh_v['events_val'], df_mesh_v['mu_val'], xerr=df_mesh_v['events_err'],
                                  yerr=df_mesh_v['mu_err'], marker='o', alpha=1, label=f'{mesh_v} V Mesh Voltage')
        average_mu = np.average(df_mesh_v['mu'])
        mu_dev = ((df_mesh_v['mu'] - average_mu) / average_mu) * 100
        mu_dev_vals, mu_dev_errs = mu_dev.apply(lambda x: x.val), mu_dev.apply(lambda x: x.err)
        ax_mus_dev_vs_events.errorbar(df_mesh_v['events_val'], mu_dev_vals, xerr=df_mesh_v['events_err'],
                                      yerr=mu_dev_errs, marker='o', alpha=0.6, label=f'{mesh_v} V Mesh Voltage')

    ax_mus_vs_events.legend()
    ax_mus_vs_events.grid()
    ax_mus_dev_vs_events.legend()
    ax_mus_dev_vs_events.grid()
    fig_mus_vs_events.tight_layout()
    fig_mus_dev_vs_events.tight_layout()

    plt.show()


def analyze_file_p2_coverage(file_path, pedestals, noise_thresholds, num_detectors, connected_channels,
                             chunk_size=10000, title=None):
    no_noise_events, event_numbers, total_events = process_file(file_path, pedestals, noise_thresholds, num_detectors,
                                                                connected_channels, chunk_size)
    no_noise_events_max = get_sample_max(no_noise_events)
    channel_sum = np.sum(no_noise_events_max, axis=0)
    fig_2d = plot_p2_2d(channel_sum, title=title)
    return fig_2d


def get_run_periods(dir_path, ped_flag, plot=True):
    run_dates = []
    for file in os.listdir(dir_path):
        if not (file.endswith('.root') or file.endswith('.fdf')) or ped_flag in file:
            continue
        file_type = 'urw' if 'urw_' in file.lower() else 'micromega'
        mesh_v, drift_v, run_date = interpret_file_name(file, file_type)
        if mesh_v is None or drift_v is None or run_date is None:
            continue
        run_dates.append(run_date)

    run_periods = identify_run_periods(run_dates, plot)

    return run_periods


def get_run_period(run_date, run_periods):
    for index, run_periods_row in run_periods.iterrows():
        if run_periods_row['start_date'] <= run_date <= run_periods_row['end_date']:
            return index
    return None


def identify_run_periods(run_dates, plot=False):
    period_gap = 60 * 60 * 1  # Hours
    period_buffer = 10  # Minutes

    run_dates.sort()
    run_periods = []
    run_period = [run_dates[0] - timedelta(minutes=period_buffer)]
    for i, run_date in enumerate(run_dates[1:]):
        if (run_date - run_dates[i]).total_seconds() > period_gap:
            run_period.append(run_dates[i] + timedelta(minutes=period_buffer))
            run_periods.append(run_period)
            run_period = [run_date - timedelta(minutes=period_buffer)]
    run_period.append(run_dates[-1] + timedelta(minutes=period_buffer))
    run_periods.append(run_period)

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(run_dates, [1] * len(run_dates), alpha=0.3)
        for run_period in run_periods:
            ax.axvline(run_period[0], ls='--', color='black')
            ax.axvline(run_period[-1], ls='--', color='black')

        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        fig.tight_layout()

    run_periods_df = pd.DataFrame(run_periods, columns=['start_date', 'end_date'])

    return run_periods_df


def run_pedestal(ped_root_path, num_detectors, noise_sigmas=5, connected_channels=None, plot_pedestals=False):
    ped_data = read_det_data(ped_root_path, num_detectors)
    pedestals = get_pedestals(ped_data)
    ped_common_noise = get_common_noise(ped_data, pedestals)
    ped_rms = get_pedestals_rms(ped_data, pedestals)
    ped_fits_no_noise_sub = get_pedestal_fits(ped_data)
    ped_fits = get_pedestal_fits(ped_data, common_noise=ped_common_noise)
    if plot_pedestals:
        plot_combined_time_series(ped_data, max_events=10)
        means = {'Dumb Pedestals': {'vals': pedestals},
                 'Fit Peds No Noise Sub': {'vals': ped_fits_no_noise_sub['mean'],
                                           'errs': ped_fits_no_noise_sub['mean_err']},
                 'Fit Pedestals': {'vals': ped_fits['mean'], 'errs': ped_fits['mean_err']}}
        plot_pedestal_comp(means)
        rmses = {'Dumb RMSes': {'vals': ped_rms},
                 'Fit Sigma No Noise Sub': {'vals': ped_fits_no_noise_sub['sigma'],
                                            'errs': ped_fits_no_noise_sub['sigma_err']},
                 'Fit Sigma': {'vals': ped_fits['sigma'], 'errs': ped_fits['sigma_err']}}
        plot_pedestal_comp(rmses)
    pedestals, ped_rms = ped_fits['mean'], ped_fits['sigma']
    noise_thresholds = get_noise_thresholds(ped_rms, noise_sigmas=noise_sigmas)
    if connected_channels is not None:
        # connected_channels = np.ones(pedestals.shape, dtype=bool)
        pedestals = pedestals * connected_channels

    return pedestals, noise_thresholds


def get_connected_channels(data, noise_thresholds):
    """
    Figure out which channels are connected by looking at the average of the max strip for each event for each detector
    where the max strip is greater than the noise threshold.
    :param data:
    :param noise_thresholds:
    :return:
    """
    data_max = get_sample_max(data)
    sig_avg = np.mean(data_max * (data_max > noise_thresholds), axis=0)
    disconnect_thresholds = 0.2 * np.mean(sig_avg, axis=1)  # Hardcoded 20% of mean gets cut, ok for at least one run
    for det_num, det in enumerate(sig_avg):
        plot_1d_data(det, title=f'Strip Above Noise Threshold Sum Detector #{det_num}')
        plt.axhline(disconnect_thresholds[0], ls='--', color='black')
    connected_channels_found = sig_avg > disconnect_thresholds[:, np.newaxis]
    print(f"connected_channels = np.array({repr(connected_channels_found.tolist())})")


def interpret_file_name(file_path, detector_type='micromega'):
    file_name = file_path.split('.')[0].split('/')[-1]
    # Define the regular expressions for "_ME_", "_DR_", and the date
    if detector_type == 'urw':
        me_pattern = re.compile(r"_STRIPMESH_(\d+)")
        dr_pattern = re.compile(r"_STRIPDRIFT_(\d+)")
    else:
        me_pattern = re.compile(r"_ME_(\d+)")
        dr_pattern = re.compile(r"_DR_(\d+)")
    date_pattern = re.compile(r"_(\d{6}_\d{2}H\d{2}_\d{3}_\d{2})")

    # Find matches in the file name
    me_match = me_pattern.search(file_name)
    dr_match = dr_pattern.search(file_name)
    date_match = date_pattern.search(file_name)

    # Extract integers if matches are found
    me_integer = int(me_match.group(1)) if me_match else None
    dr_integer = int(dr_match.group(1)) if dr_match else None

    # Extract and convert date string to datetime object
    date_str = date_match.group(1) if date_match else None
    run_date = datetime.strptime(date_str, "%y%m%d_%HH%M_%f_%S")

    return me_integer, dr_integer, run_date


def load_connected_channels():
    connected_channels = np.array([[True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                    True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                    True, True, True, True, False, False, False, False, False, False, False, False,
                                    False, False, False, False, False, False, False, False, False, False, False, False,
                                    False, False, False, False, False, False, False, False, False, False, False, False],
                                   [True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                    True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                    True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                    True, True, True, True, True, True, True, True, False, False, False, False, False,
                                    False, False, False, False, False, False, False, False, False]])

    return connected_channels


def write_to_file(data, file_path, encoding='ISO-8859-1'):
    with open(file_path, 'w') as file:
        for run in data:
            for key, value in run.items():
                file.write(f'{key}: {value}\t')
            file.write('\n')


def read_from_file(file_path):
    results = []
    datetime_format = '%Y-%m-%d %H:%M:%S'
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            result = {}
            for element in line.strip().split('\t'):
                key, value = element.strip().split(': ')
                if ' ± ' in value:
                    result[key] = Measure(*(float(x) for x in value.split(' ± ')))
                elif key == 'run_date':
                    result[key] = datetime.strptime(value, datetime_format)
                else:
                    result[key] = eval(value)  # Using eval to convert the string back to its original data type
            results.append(result)
    return results
