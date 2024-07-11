#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 09 5:11 PM 2024
Created in PyCharm
Created as saclay_micromegas/DreamData.py

@author: Dylan Neff, Dylan
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
from datetime import datetime
from time import time

import uproot
import awkward as ak
import vector


class DreamData:
    def __init__(self, data_dir, feu_num, feu_connectors, ped_dir=None):
        self.data_dir = data_dir
        self.ped_dir = ped_dir
        self.feu_num = feu_num
        self.feu_connectors = feu_connectors

        self.array_flag = '_array'  # Ensure only files with array structure are read
        self.ped_flag = '_pedthr_'
        self.data_flag = '_datrun_'

        self.channels_per_connector = 64

        self.ped_data = None
        self.data = None

        self.ped_means = None
        self.ped_sigmas = None
        self.noise_thresholds = None

        self.data_amps = None
        self.data_mean_times = None

    def read_ped_data(self):
        ped_dir = self.ped_dir if self.ped_dir is not None else self.data_dir
        if ped_dir is None:
            print('Error: No ped directory specified.')
            return None
        ped_files = get_good_files(os.listdir(ped_dir), [self.ped_flag, self.array_flag], self.feu_num, '.root')

        if len(ped_files) == 0:
            print('Error: No ped files found.')
            return None
        if len(ped_files) > 1:
            print('Warning: Multiple ped files found, using first one.')
            print(f'Ped files found: {ped_files}')
        ped_file_path = f'{ped_dir}{ped_files[0]}'

        ped_data = read_det_data(ped_file_path)
        # self.ped_data = split_det_data(ped_data, self.feu_connectors, self.channels_per_connector)
        self.ped_data = split_det_data(ped_data, self.feu_connectors, self.channels_per_connector, to_connectors=False)

        self.get_pedestals()
        self.get_noise_thresholds()

    def get_pedestals(self):
        pedestals = get_pedestals_by_median(self.ped_data)
        # common_noise = self.get_common_noise(self.ped_data, pedestals)
        ped_common_noise_sub = self.subtract_common_noise(self.ped_data, pedestals)
        ped_fits = get_pedestal_fits(ped_common_noise_sub)
        self.ped_means = ped_fits['mean']
        self.ped_sigmas = ped_fits['sigma']
        # ped_connectors = split_det_data(self.ped_data, self.feu_connectors, self.channels_per_connector)
        # ped_means, ped_sigmas = [], []
        # for ped_connector_data, common_noise_connector in zip(ped_connectors, common_noise):
        #     ped_fits = get_pedestal_fits(ped_connector_data, common_noise_connector)
        #     ped_means.append(ped_fits['mean'])
        #     ped_sigmas.append(ped_fits['sigma'])
        # self.ped_means = np.concatenate(ped_means)
        # self.ped_sigmas = np.concatenate(ped_sigmas)

    def subtract_common_noise(self, data, pedestals):
        data_connectors = split_det_data(data, self.feu_connectors, self.channels_per_connector)
        peds_connectors = split_det_data(pedestals, self.feu_connectors, self.channels_per_connector)
        data_sub = []
        for data_connector, ped_connector in zip(data_connectors, peds_connectors):
            connector_common_noise = get_common_noise(data_connector, ped_connector)
            data_sub.append(data_connector - connector_common_noise[:, np.newaxis, :])
        return np.array(data_sub)

    # def subtract_common_noise(self, data, common_noise):
    #     data_connectors = split_det_data(data, self.feu_connectors, self.channels_per_connector)
    #     data_sub = []
    #     for data_connector, common_noise_connector in zip(data_connectors, common_noise):
    #         data_connector = data_connector - common_noise_connector[:, np.newaxis, :]
    #         data_sub.append(data_connector)
    #     return np.concatenate(data_sub, axis=1)

    def get_noise_thresholds(self, noise_sigmas=5):
        self.noise_thresholds = get_noise_thresholds(self.ped_sigmas, noise_sigmas)

    def read_data(self):
        if self.data_dir is None:
            print('Error: No data directory specified.')
            return None
        data_files = get_good_files(os.listdir(self.data_dir), [self.data_flag, self.array_flag], self.feu_num, '.root')

        if len(data_files) == 0:
            print('Error: No data files found.')
            return None

        self.data = []
        for data_file in data_files:
            data_file_path = f'{self.data_dir}{data_file}'
            data = read_det_data(data_file_path)
            self.data.append(split_det_data(data, self.feu_connectors, self.channels_per_connector, to_connectors=False))
            print(type(self.data[-1]))
            print(self.data[-1].dtype)

        self.data = np.concatenate(self.data)
        self.data = self.subtract_common_noise(self.data, self.ped_means)
        self.data = subtract_pedestal(self.data, self.ped_means)
        print(self.data.shape)

    def subtract_pedestals_from_data(self):
        self.data = subtract_pedestal(self.data, self.ped_means)

    def get_event_amplitudes(self):
        start = time()
        self.data_amps, self.data_mean_times = [], []
        for connector_i, data_connector in enumerate(self.data):
            fits = get_waveform_fits(self.data, self.noise_thresholds[connector_i])
            self.data_amps.append(fits['amplitude'])
            self.data_mean_times.append(fits['mean'])
        print(f'Fitting time: {time() - start} s')

    def plot_pedestals(self):
        fig_mean, ax_mean = plt.subplots()
        ax_mean.plot(self.ped_means)
        ax_mean.set_title('Pedestal Means')
        ax_mean.set_xlabel('Channel')
        ax_mean.set_ylabel('Mean')
        fig_mean.tight_layout()

        fig_sig, ax_sig = plt.subplots()
        ax_sig.plot(self.ped_sigmas)
        ax_sig.set_title('Pedestal Sigmas')
        ax_sig.set_xlabel('Channel')
        ax_sig.set_ylabel('Sigma')
        fig_sig.tight_layout()


def read_det_data(file_path, variable_name='amplitude', tree_name='nt'):
    vector.register_awkward()
    # Open the ROOT file with uproot
    root_file = uproot.open(file_path)

    # Access the tree in the file
    tree = root_file[tree_name]

    # Get the variable data from the tree
    variable_data = ak.to_numpy(tree[variable_name].array())
    variable_data = variable_data.astype(np.float32)
    root_file.close()

    return variable_data


def split_det_data(det_data, feu_connectors, channels_per_connector, to_connectors=True):
    channel_list = np.concatenate([np.arange(channels_per_connector) + channels_per_connector * (connector_num - 1)
                                   for connector_num in feu_connectors])
    if det_data.ndim == 1:
        det_data = det_data[channel_list]
        if to_connectors:
            det_data = np.array(np.split(det_data, len(feu_connectors)))
    elif det_data.ndim == 3:
        det_data = det_data[:, channel_list]
        if to_connectors:
            det_data = np.array(np.split(det_data, len(feu_connectors), axis=1))
    else:
        print('Error: Data shape not recognized.')
        return None

    return det_data


# Pedestal and Noise Functions

def get_pedestals_by_median(data):
    channel_medians = np.median(data, axis=2)  # Median of samples in each channel for each event
    channel_medians_transpose = np.transpose(channel_medians, axes=(1, 0))  # Concatenate channels for each event
    channel_means = np.mean(channel_medians_transpose, axis=1)  # Mean of channel medians over all events

    # This takes the median of samples over all events for each channel. Seems pretty equivalent
    # data_event_concat = np.concatenate(data, axis=1)
    # channel_means = np.mean(data_event_concat, axis=1)
    # channel_medians = np.median(data_event_concat, axis=1)

    return channel_means


def subtract_pedestal(data, pedestal):
    return data - pedestal[np.newaxis, :, np.newaxis]


def get_pedestals_rms(ped_data, ped_means):
    ped_zeroed = subtract_pedestal(ped_data, ped_means)  # Subtract averages from pedestal data
    ped_zeroed_concat = np.concatenate(ped_zeroed, axis=1)  # Concatenate all events
    ped_rms = np.std(ped_zeroed_concat, axis=1)  # Get RMS of pedestal data for each strip
    # Existing code averages over all strips, but will see if we can get away with strip by strip
    return ped_rms


def get_pedestal_fits(ped_data, common_noise=None):
    if common_noise is not None:
        ped_data = ped_data - common_noise[:, np.newaxis, :]
    ped_concat = np.concatenate(ped_data, axis=1)  # Concatenate all events
    ped_fits = np.apply_along_axis(fit_pedestals, -1, ped_concat)
    ped_fits = np.transpose(ped_fits)
    ped_fits = dict(zip(['mean', 'sigma', 'mean_err', 'sigma_err'], ped_fits))
    return ped_fits


def fit_pedestals(strip_samples):
    bin_edges = np.arange(-0.5, 4097.5, 1)
    mean = np.mean(strip_samples)
    sd = np.std(strip_samples)
    hist, _ = np.histogram(strip_samples, bins=bin_edges, density=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    popt, pcov = cf(gaussian_density, bin_centers, hist, p0=[mean, sd])
    perr = np.sqrt(np.diag(pcov))
    return *popt, *perr


def get_common_noise(data, pedestals):
    data_ped_sub = data - pedestals[np.newaxis, :, np.newaxis]
    common_noise = np.nanmedian(data_ped_sub, axis=1)

    return common_noise


def get_noise_thresholds(ped_rms, noise_sigmas=5):
    return noise_sigmas * ped_rms


def filter_noise_events(data, ped_thresholds, return_type='data'):
    """
    Filter out events where all channels are below the noise threshold.
    :param data:
    :param ped_thresholds:
    :param return_type:
    :return:
    """
    sample_maxes = get_sample_max(data)
    mask = np.any(sample_maxes > ped_thresholds, axis=1)
    if return_type == 'mask':
        return mask
    return data[mask]


def get_sample_max(data):
    return np.max(data, axis=-1)


def get_waveform_fits(data, noise_thresholds=None):
    if noise_thresholds is not None:
        data = filter_noise_events(data, noise_thresholds)
    fits = np.apply_along_axis(fit_waveform, -1, data)
    fits = np.transpose(fits)
    fits = dict(zip(['amplitude', 'mean', 'sigma', 'amplitude_err', 'mean_err', 'sigma_err'], fits))
    return fits


def fit_waveform(waveform):
    print(waveform)
    amplitude = np.max(waveform)
    mean = np.argmax(waveform)
    sd = len(waveform) / 5
    popt, pcov = cf(gaussian, np.arange(len(waveform)), waveform, p0=[amplitude, mean, sd])
    perr = np.sqrt(np.diag(pcov))
    return *popt, *perr


def gaussian_density(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def get_good_files(file_list, flags=None, feu_num=None, file_ext=None):
    good_files = []
    for file in file_list:
        if file_ext is not None and not file.endswith(file_ext):
            continue
        if flags is not None:
            continue_flag = False
            for flag in flags:
                if flag is not None and flag not in file:
                    continue_flag = True
            if continue_flag:
                continue
        if feu_num is not None and feu_num != get_num_from_fdf_file_name(file, -1):
            continue
        good_files.append(file)

    return good_files


# vvv Copied from Cosmic_Bench_DAQ_Control/common_functions.py 7-5-24 vvv

def get_date_from_fdf_file_name(file_name):
    """
    Get date from file name with format ...xxx_xxx_240212_11H42_000_01.xxx
    :param file_name:
    :return:
    """
    date_str = file_name.split('_')[-4] + ' ' + file_name.split('_')[-3]
    date = datetime.strptime(date_str, '%y%m%d %HH%M')
    return date


def get_num_from_fdf_file_name(file_name, num_index=-2):
    """
    Get fdf style file number from file name with format ...xxx_xxx_240212_11H42_000_01.xxx
    Updated to more robustly get first number from back.
    :param file_name:
    :param num_index:
    :return:
    """
    file_split = remove_after_last_dot(file_name).split('_')
    file_nums = []
    for x in file_split:
        try:
            file_nums.append(int(x))
        except ValueError:
            pass
    return file_nums[num_index]


def remove_after_last_dot(input_string):
    # Find the index of the last dot
    last_dot_index = input_string.rfind('.')

    # If there's no dot, return the original string
    if last_dot_index == -1:
        return input_string

    # Return the substring up to the last dot (not including the dot)
    return input_string[:last_dot_index]


def get_run_name_from_fdf_file_name(file_name):
    file_name_split = file_name.split('_')
    run_name_end_index = 0
    for i, part in enumerate(file_name_split):  # Find xxHxx in file name split
        if len(part) == 5 and part[2] == 'H' and is_convertible_to_int(part[:2]) and is_convertible_to_int(part[3:]):
            run_name_end_index = i
            break
    run_name = '_'.join(file_name_split[:run_name_end_index + 1])
    return run_name


def is_convertible_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# ^^^ Copied from Cosmic_Bench_DAQ_Control/common_functions.py ^^^