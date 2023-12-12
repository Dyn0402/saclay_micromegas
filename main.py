#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 05 11:29 2023
Created in PyCharm
Created as saclay_micromegas/main

@author: Dylan Neff, dn277127
"""

import os
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import ROOT
import uproot
import awkward as ak

# Suppress the specified warning messages
# ROOT.gErrorIgnoreLevel = ROOT.kError + ROOT.kBreak
ROOT.gErrorIgnoreLevel = -1


def main():
    # fdf_dir = 'test_data/fdf/'
    # raw_root_dir = 'test_data/raw_root/'
    base_path = '/local/home/dn277127/Documents/TestBeamData/2023_July_Saclay/dec6/'
    fdf_dir = base_path
    raw_root_dir = f'{base_path}raw_root/'
    ped_flag = '_pedthr_'
    num_threads = 6
    free_memory = 2  # GB of memory to allocate (in theory, in reality needs a lot of wiggle room)
    chunk_size = f'{free_memory / num_threads} GB'
    print(f'{num_threads} threads, {chunk_size} chunk size')

    overwrite = False
    num_detectors = 2
    noise_sigmas = 5
    plot_pedestals = True

    ped_file = None
    fdf_files = [file for file in os.listdir(fdf_dir) if file[-4:] == '.fdf']
    fdf_data_list = [(file, fdf_dir, raw_root_dir, overwrite) for file in fdf_files]
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        with tqdm(total=len(fdf_files), desc='Processing fdfs') as pbar:
            for root_name in executor.map(process_fdf, *zip(*fdf_data_list)):
                if ped_flag in root_name:
                    if ped_file is not None:
                        print(f'Warning: Multiple ped files found: {ped_file}, {root_name}')
                    else:
                        ped_file = root_name
                pbar.update(1)

    # for file_num, file in enumerate(fdf_files):
    #     print(f'\nReading {file_num + 1}/{len(fdf_files)} {file} to root')
    #     root_name = file[:-4] + '.root'
    #     if ped_flag in root_name:
    #         if ped_file is not None:
    #             print(f'Warning: Multiple ped files found: {ped_file}, {root_name}')
    #         else:
    #             ped_file = root_name
    #     if not overwrite and root_name in os.listdir(raw_root_dir):
    #         print(f'{root_name} already exists in {raw_root_dir}, skipping')
    #         continue
    #     read_fdf_to_root(file, fdf_dir, raw_root_dir, root_name)

    # Deal with pedestals
    ped_root_path = os.path.join(raw_root_dir, ped_file)
    ped_data = read_det_data(ped_root_path, num_detectors)
    pedestals = get_pedestals(ped_data)
    ped_common_noise = get_common_noise(ped_data, pedestals)
    ped_rms = get_pedestals_rms(ped_data, pedestals)
    noise_thresholds = get_noise_thresholds(ped_rms, noise_sigmas=noise_sigmas)
    if plot_pedestals:
        plot_combined_time_series(ped_data, max_events=10)
        [plot_1d_data(pedestal, title=f'Detector {det_num} Pedestals') for det_num, pedestal in enumerate(pedestals)]
        plot_2d_data(*pedestals)
        [plot_1d_data(ped_rms_det, title=f'Detector {det_num} Ped STDs') for det_num, ped_rms_det in enumerate(ped_rms)]
        plt.show()

    data_files = [os.path.join(raw_root_dir, file) for file in os.listdir(raw_root_dir)
                  if file[-5:] == '.root' and file != ped_file]
    process_data = [(file, pedestals, noise_thresholds, num_detectors, chunk_size) for file in data_files]
    no_noise_events = []
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        with tqdm(total=len(process_data), desc='Processing Trees') as pbar:
            for noiseless_events_chunk in executor.map(process_file, *zip(*process_data)):
                no_noise_events.extend(noiseless_events_chunk)
                pbar.update(1)

    no_noise_events = np.concatenate(no_noise_events, axis=0)
    print(f'no_noise_events.shape: {no_noise_events.shape}')
    no_noise_events_max = get_sample_max(no_noise_events)
    no_noise_events_max_strip = get_max_strip(no_noise_events_max)
    plot_1d_sample_max_hist(no_noise_events_max_strip, bins=64, title='Sample Max Strip Number No Noise Events')
    no_noise_events_strip_max = get_strip_max(no_noise_events_max)
    plot_1d_sample_max_hist(no_noise_events_strip_max, bins=100, title='Sample Max ADC Spectrum No Noise Events')
    print(f'no_noise_events_max.shape: {no_noise_events_max.shape}')
    signal_mask = identify_common_signal(no_noise_events_max, signal_threshold=200)
    print(f'signal_mask.shape: {signal_mask.shape}')
    signal_events = no_noise_events[signal_mask]
    print(f'signal_events.shape: {signal_events.shape}')
    signal_events_max = no_noise_events_max[signal_mask]
    signal_events_max_strip = get_max_strip(signal_events_max)
    plot_1d_sample_max_hist(signal_events_max_strip, bins=64, title='Sample Max Strip Number Spectrum Signal Events')
    signal_events_strip_max = get_strip_max(signal_events_max)
    plot_1d_sample_max_hist(signal_events_strip_max, bins=100, title='Sample Max ADC Spectrum Signal Events')
    print(f'signal_events_max.shape: {signal_events_max.shape}')
    plot_high_noise_metric(signal_events_max, threshold=500)
    high_noise_mask = identify_high_noise(signal_events_max, avg_threshold=500)
    print(f'high_noise_mask.shape: {high_noise_mask.shape}')
    high_noise_events = signal_events[high_noise_mask]
    print(f'high_noise_events.shape: {high_noise_events.shape}')
    # [plot_2d_data(*event) for event in signal_events_max[~high_noise_mask][:10]]
    # plot_position_data(signal_events_max[~high_noise_mask], event_nums=None)
    plt.show()

    print('donzo')


def process_fdf(file, fdf_dir, raw_root_dir, overwrite):
    root_name = file[:-4] + '.root'
    if not overwrite and root_name in os.listdir(raw_root_dir):
        print(f'{root_name} already exists in {raw_root_dir}, skipping')
    else:
        read_fdf_to_root(file, fdf_dir, raw_root_dir, root_name)
    return root_name


def read_fdf_to_root(file, fdf_dir, raw_root_dir, root_name):
    os.system(f'./DreamDataReader {os.path.join(fdf_dir, file)}')
    raw_root_path = os.path.join(raw_root_dir, root_name)
    shutil.move('output.root', raw_root_path)
    return raw_root_path


def plot_adc_pyroot(file_path):
    # Create a ROOT TFile object to open the file
    root_file = ROOT.TFile(file_path)

    # Access the tree in the file
    tree = root_file.Get('T')

    # Create a canvas to draw the histogram
    canvas = ROOT.TCanvas("canvas", "Histogram Canvas", 800, 600)

    # Create a histogram from the tree variable
    variable_name = 'StripAmpl'
    tree.Draw(variable_name)
    histogram = ROOT.gPad.GetPrimitive("htemp")

    # Set histogram attributes
    histogram.SetTitle(f"Histogram of {variable_name}")
    histogram.GetXaxis().SetTitle(variable_name)
    histogram.GetYaxis().SetTitle("Frequency")

    # Draw the histogram on the canvas
    canvas.Draw()

    # Keep the program running to display the canvas
    ROOT.gApplication.Run()
    input()


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
    print(strip_means.shape)

    # This takes the median of samples over all events for each strip
    # data_event_concat = np.concatenate(data, axis=2)
    # strip_means = np.mean(data_event_concat, axis=2)
    # strip_medians = np.median(data_event_concat, axis=2)

    return strip_means


def get_common_noise(data, pedestals):
    return data - pedestals[np.newaxis, :, :, np.newaxis]


def get_pedestals_rms(ped_data, ped_means):
    ped_zeroed = subtract_pedestal(ped_data, ped_means)  # Subtract averages from pedestal data
    print(ped_zeroed.shape)
    ped_zeroed_concat = np.concatenate(ped_zeroed, axis=2)  # Concatenate all events
    print(ped_zeroed_concat.shape)
    ped_rms = np.std(ped_zeroed_concat, axis=2)  # Get RMS of pedestal data for each strip
    print(ped_rms.shape)
    # Existing code averages over all strips, but will see if we can get away with strip by strip
    return ped_rms


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


def subtract_pedestal(data, pedestal):
    return data - pedestal[np.newaxis, :, :, np.newaxis]


def get_sample_max(data):
    return np.max(data, axis=-1)


def plot_position_data(data, event_nums=None):
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


def plot_spectrum(signal_data):
    fig, ax = plt.subplots()
    for det_num, det in enumerate(signal_data):
        ax.hist(det, bins=50, edgecolor='black', label=f'Detector #{det_num}')
    ax.set_xlabel('ADC')
    ax.set_ylabel('Frequency')
    ax.set_title('ADC Spectrum')


def plot_1d_sample_max_hist(max_data, bins=100, title=None):
    fig, ax = plt.subplots()
    for det_num, det in enumerate(np.transpose(max_data)):
        ax.hist(det, bins=bins, edgecolor='black', label=f'Detector #{det_num}')
    ax.legend()
    ax.set_xlabel('ADC')
    ax.set_ylabel('Events')
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
    :return:
    """
    noise_strips = max_data < noise_threshold  # Compare strip maxima with the threshold
    noise_mask = np.all(noise_strips, axis=(1, 2))  # Mark event as noise if all strips on all detectors below threshold
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


def identify_high_noise(data_max, avg_threshold=1000):
    """
    Filter out events with high noise.
    :param data_max:
    :param avg_threshold:
    :return:
    """
    det_avg = np.mean(data_max, axis=2)
    event_max = np.max(det_avg, axis=1)
    high_noise_mask = event_max > avg_threshold

    return high_noise_mask


def plot_high_noise_metric(data_max, threshold=None):
    det_avg = np.transpose(np.mean(data_max, axis=2))
    fig, ax = plt.subplots()
    for det_num, det in enumerate(det_avg):
        ax.scatter(range(len(det)), det, label=f'Detector #{det_num}')
    if threshold is not None:
        ax.axhline(threshold, ls='--', color='black', label='High Noise Threshold')
    ax.set_xlabel('Event #')
    ax.set_ylabel('Detector Strip Averaged ADC')
    ax.set_title('High Noise Metric')
    ax.legend()
    fig.tight_layout()


def process_chunk(chunk, pedestals, noise_thresholds, num_detectors):
    data = read_det_data_chunk(chunk['StripAmpl'], num_detectors)
    ped_sub_data = subtract_pedestal(data, pedestals)
    max_data = get_sample_max(ped_sub_data)
    noise_mask = identify_noise(max_data, noise_threshold=noise_thresholds)
    data_no_noise = suppress_noise(data, noise_mask)
    # signal_mask = identify_common_signal(max_data, signal_threshold=400)
    # data_signal = select_signal(data, signal_mask)

    return data_no_noise


def process_file(file_path, pedestals, noise_thresholds, num_detectors, chunk_size=10000):
    with uproot.open(file_path) as file:
        tree_names = file.keys()

    noise_filtered_events = []
    for tree_name in tree_names:
        with uproot.open(file_path) as file:
            tree = file[tree_name]
            for chunk in uproot.iterate(tree, branches=['StripAmpl'], entrysteps=chunk_size):
                data_no_noise = process_chunk(chunk, pedestals, noise_thresholds, num_detectors)
                noise_filtered_events.append(data_no_noise)

    return noise_filtered_events


if __name__ == '__main__':
    main()
