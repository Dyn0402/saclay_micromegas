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

        self.waveform_fit_func = 'waveform_func'

        self.channels_per_connector = 64

        self.ped_data = None
        self.data = None

        self.ped_means = None
        self.ped_sigmas = None
        self.noise_thresholds = None

        self.data_amps = None
        self.data_mean_times = None
        self.data_fit_success = None
        self.fit_params = None

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
        self.ped_data = split_det_data(ped_data, self.feu_connectors, self.channels_per_connector, to_connectors=False)

        self.get_pedestals()
        self.get_noise_thresholds()

    def get_pedestals(self):
        pedestals = get_pedestals_by_median(self.ped_data)
        ped_common_noise_sub = self.subtract_common_noise(self.ped_data, pedestals)
        ped_fits = get_pedestal_fits(ped_common_noise_sub)
        self.ped_means = ped_fits['mean']
        self.ped_sigmas = ped_fits['sigma']

    def subtract_common_noise(self, data, pedestals):
        feu_connectors = np.array(self.feu_connectors) - min(self.feu_connectors) + 1  # Ensure connectors start at 1
        data_connectors = split_det_data(data, feu_connectors, self.channels_per_connector)
        peds_connectors = split_det_data(pedestals, feu_connectors, self.channels_per_connector)
        data_sub = []
        for data_connector, ped_connector in zip(data_connectors, peds_connectors):
            connector_common_noise = get_common_noise(data_connector, ped_connector)
            data_sub.append(data_connector - connector_common_noise[:, np.newaxis, :])
        return np.concatenate(data_sub, axis=1)

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

        self.data = np.concatenate(self.data)
        self.data = self.subtract_common_noise(self.data, self.ped_means)
        self.subtract_pedestals_from_data()
        print(f'Read in data shape: {self.data.shape}')
        self.get_event_amplitudes()

    def subtract_pedestals_from_data(self):
        self.data = subtract_pedestal(self.data, self.ped_means)

    def get_event_amplitudes(self):
        start = time()
        fits = get_waveform_fits(self.data, self.noise_thresholds, self.waveform_fit_func)
        self.data_amps = fits['amplitude']
        self.data_mean_times = fits['mean']
        self.data_fit_success = fits['success'] != 0
        self.fit_params = fits
        print(f'Fitting time: {time() - start} s')

    def plot_event_amplitudes(self, channel=None):
        amps = np.ravel(self.data_amps) if channel is None else self.data_amps[:, channel]
        amps = amps[~np.isnan(amps)]
        fig, ax = plt.subplots()
        ax.hist(amps, bins=100)
        ax.set_title('Event Amplitudes')
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Counts')
        fig.tight_layout()

    def plot_event_mean_times(self, channel=None):
        mean_times = np.ravel(self.data_mean_times) if channel is None else self.data_mean_times[:, channel]
        mean_times = mean_times[~np.isnan(mean_times)]
        fig, ax = plt.subplots()
        ax.hist(mean_times, bins=100)
        ax.set_title('Event Mean Times')
        ax.set_xlabel('Time')
        ax.set_ylabel('Counts')
        fig.tight_layout()

    def plot_fit_param(self, param, params_ranges=None, channel=None):
        selection_data = self.get_selected_data(params_ranges, channel)
        # param_data = self.fit_params[param][self.data_fit_success]
        # param_data = np.ravel(param_data) if channel is None else param_data[:, channel]
        fig, ax = plt.subplots()
        ax.hist(selection_data, bins=100)
        ax.set_title(f'Fit Parameter: {param}')
        ax.set_xlabel(param)
        ax.set_ylabel('Counts')
        fig.tight_layout()

    def plot_event_fit_success(self):
        fig, ax = plt.subplots()
        # Count the number of channels/events where amplitude is zero
        amps, successes = np.ravel(self.data_amps), np.ravel(self.data_fit_success)
        amp_zero_counts = np.sum(np.isnan(amps))
        # Count the number of channels/events where success is false and amplitude is not zero
        # print(successes, np.sum(successes))
        # print(successes == 0, np.sum(successes == 0))
        # print(~successes, np.sum(~successes))
        # print(~np.isnan(amps), np.sum(~np.isnan(amps)))
        # print(np.isnan(amps), np.sum(np.isnan(amps)))
        # print(~successes & ~np.isnan(amps), np.sum(~successes & ~np.isnan(amps)))
        success_false_counts = np.sum(~successes & ~np.isnan(amps))
        # Count the number of channels/events where success is true
        success_true_counts = np.sum(successes)
        # Plot a bar chart of these three counts
        labels = ['Amplitude Zero', 'Fit Failed', 'Fit Success']
        counts = [amp_zero_counts, success_false_counts, success_true_counts]
        ax.bar(labels, counts)
        ax.set_yscale('log')
        ax.set_title('Event Fit Success')
        ax.set_ylabel('Counts')
        fig.tight_layout()

    def plot_fits(self, params_ranges, n_max=5, channel=None):
        """
        Plot fits where params are within ranges.
        :param params_ranges: Dictionary of param names and ranges
        :param n_max:
        :param channel:
        :return:
        """
        # data, success, fit_params = self.data, self.data_fit_success, self.fit_params
        # if channel is not None:
        #     data, success = data[:, channel], success[:, channel]
        #     params = {key: val[:, channel] for key, val in fit_params.items()}
        # else:
        #     success = np.ravel(success)
        #     params = {key: np.ravel(val) for key, val in fit_params.items()}
        #     # Reshape data to combine first two axes
        #     data = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
        # selection_mask = success
        # for param, param_range in params_ranges.items():
        #     selection_mask = selection_mask & (param_range[0] < params[param]) & (params[param] < param_range[1])
        # selection_data = data[selection_mask]
        selection_data = self.get_selected_data(params_ranges, channel)
        print(f'Found {len(selection_data)} event channels with fit params in ranges.')
        n_events = min(n_max, len(selection_data))
        selection_data = selection_data[:n_events]
        for event in selection_data:
            fig, ax = plt.subplots()
            hot_fit_params = fit_waveform_func(event)
            func_pars = hot_fit_params[:4]
            func_errs = hot_fit_params[4:-1]
            # amplitude = np.max(event)
            max_index = np.argmax(event)
            end_index = np.where(event[max_index:] < 0)[0][0] + max_index
            # start_index = np.where(event[:max_index] < amplitude * 0.1)[0][-1]
            fit_x = np.linspace(0, end_index, 1000)
            ax.plot(event, marker='o')
            ax.plot(fit_x, waveform_func_reparam(fit_x, *func_pars), color='red')
            fit_string = f'Amplitude: {func_pars[0]:.2f}±{func_errs[0]:.2f}\n' \
                            f'Mean: {func_pars[1]:.2f}±{func_errs[1]:.2f}\n' \
                            f'Time Shift: {func_pars[2]:.2f}±{func_errs[2]:.2f}\n' \
                            f'Q: {func_pars[3]:.2f}±{func_errs[3]:.2f}'
            ax.annotate(fit_string, (0.9, 0.9), xycoords='axes fraction', ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def get_selected_data(self, params_ranges=None, channel=None):
        """
        Get data that fits within specified parameter ranges.
        :return:
        """
        data, success, fit_params = self.data, self.data_fit_success, self.fit_params
        if channel is not None:
            data, success = data[:, channel], success[:, channel]
            params = {key: val[:, channel] for key, val in fit_params.items()}
        else:
            success = np.ravel(success)
            params = {key: np.ravel(val) for key, val in fit_params.items()}
            # Reshape data to combine first two axes
            data = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
        selection_mask = success
        if params_ranges is not None:
            for param, param_range in params_ranges.items():
                selection_mask = selection_mask & (param_range[0] < params[param]) & (params[param] < param_range[1])
        selection_data = data[selection_mask]

        return selection_data

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

    def plot_pedestal_fit(self, channel):
        ped_data = self.ped_data[channel]
        fit_pedestals(ped_data, plot=True)


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


def fit_pedestals(strip_samples, plot=False):
    bin_edges = np.arange(-0.5, 4097.5, 1)
    mean = np.mean(strip_samples)
    sd = np.std(strip_samples)
    hist, _ = np.histogram(strip_samples, bins=bin_edges, density=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    popt, pcov = cf(gaussian_density, bin_centers, hist, p0=[mean, sd])
    perr = np.sqrt(np.diag(pcov))
    if plot:
        fig, ax = plt.subplots()
        ax.bar(bin_centers, hist, width=1, align='center')
        ax.plot(bin_centers, gaussian_density(bin_centers, *popt), 'r-')
        fit_label = f'mu={popt[0]:.2f}±{perr[0]:.2f}\nsigma={popt[1]:.2f}±{perr[1]:.2f}'
        ax.annotate(fit_label, (0.9, 0.9), xycoords='axes fraction', ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title('Pedestal Fit')
        ax.set_xlabel('ADC')
        ax.set_ylabel('Density')
        fig.tight_layout()
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


def get_waveform_fits(data, noise_thresholds=None, func='gaus'):
    if noise_thresholds is not None:
        sample_maxes = get_sample_max(data)
        # If sample max below noise threshold, set to zero
        noise_mask = sample_maxes > noise_thresholds[np.newaxis, :]
        data = np.where(noise_mask[:, :, np.newaxis], data, 0)
    if func == 'gaus':
        fits = np.apply_along_axis(fit_waveform_gaus, -1, data)
        param_names = ['amplitude', 'mean', 'sigma', 'amplitude_err', 'mean_err', 'sigma_err', 'success']
    elif func == 'waveform_func':
        fits = np.apply_along_axis(fit_waveform_func, -1, data)
        param_names = ['amplitude', 'mean', 'time_shift', 'q', 'amplitude_err', 'mean_err', 'time_shift_err', 'q_err',
                       'success']
    else:
        print('Error: Fit function not recognized.')
        return None
    fits = fits.transpose((2, 0, 1))
    fits = dict(zip(param_names, fits))
    return fits


def fit_waveform_gaus(waveform):
    amplitude = np.max(waveform)
    if amplitude == 0:
        return 0, 0, 0, 0, 0, 0, False
    mean = np.argmax(waveform)
    sd = len(waveform) / 5
    try:
        popt, pcov = cf(gaussian, np.arange(len(waveform)), waveform, p0=[amplitude, mean, sd])
        perr = np.sqrt(np.diag(pcov))
        return *popt, *perr, True
    except RuntimeError:
        return amplitude, mean, sd, 0, 0, 0, False


def fit_waveform_func(waveform):
    amplitude = np.max(waveform)
    if amplitude == 0:
        # return 0, 0, 0, 0, 0, 0, 0, 0, False
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False
    mean = np.argmax(waveform)
    p0 = [amplitude, mean, 3, 2. / 3]
    p_bounds = [[0, 0.001, -len(waveform), 0], [20000, len(waveform) * 4, len(waveform), 2]]
    try:
        # Find first point after max that is below 0 and last point before max less than 10% of max. Only fit this range
        max_index = np.argmax(waveform)
        end_index = np.where(waveform[max_index:] < 0)[0][0] + max_index
        # start_index = np.where(waveform[:max_index] < amplitude * 0.1)[0][-1]
        waveform = waveform[:end_index]
        popt, pcov = cf(waveform_func_reparam, np.arange(len(waveform)), waveform, p0=p0, bounds=p_bounds)
        perr = np.sqrt(np.diag(pcov))
        return *popt, *perr, True
    except (RuntimeError, ValueError, IndexError):
        return amplitude, mean, 0, np.nan, np.nan, np.nan, np.nan, np.nan, False


def gaussian_density(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def waveform_func(t, a, w, q):
    if np.isscalar(t):
        # Handle scalar input
        if t <= 0:
            return 0
        term1 = np.sqrt((2 * q - 1) / (2 * q + 1)) * np.sin(w * t / 2 * np.sqrt(4 - 1 / q**2))
        term2 = -np.cos(w * t / 2 * np.sqrt(4 - 1 / q**2))
        return a * (np.exp(-w * t) + np.exp(-w * t / (2 * q)) * (term1 + term2))
    else:
        # Handle numpy array input
        result = np.zeros_like(t)
        positive_t_mask = t > 0
        term1 = np.sqrt((2 * q - 1) / (2 * q + 1)) * np.sin(w * t[positive_t_mask] / 2 * np.sqrt(4 - 1 / q**2))
        term2 = -np.cos(w * t[positive_t_mask] / 2 * np.sqrt(4 - 1 / q**2))
        result[positive_t_mask] = a * (np.exp(-w * t[positive_t_mask]) + np.exp(-w * t[positive_t_mask] / (2 * q)) * (term1 + term2))
        return result


def waveform_func_reparam(t, a, t_max, t_shift, q):
    t = t - t_shift
    t_max = t_max - t_shift
    w = 2 / t_max
    func = waveform_func(t, 1, w, q)
    func_max = waveform_func(t_max, 1, w, q)
    return a * func / func_max



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