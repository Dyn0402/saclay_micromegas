#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 09 5:11 PM 2024
Created in PyCharm
Created as saclay_micromegas/DreamData.py

@author: Dylan Neff, Dylan
"""

import os
import math
import concurrent.futures
from functools import partial
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit as cf
from datetime import datetime
from time import time
import copy

import uproot
import awkward as ak
import vector

from Measure import Measure


class DreamData:
    def __init__(self, data_dir, feu_num, feu_connectors, ped_dir=None, waveform_fit_func=None):
        self.data_dir = data_dir if data_dir.endswith('/') else f'{data_dir}/'
        self.ped_dir = ped_dir if ped_dir is not None and ped_dir.endswith('/') else f'{ped_dir}/'
        self.feu_num = feu_num
        self.feu_connectors = feu_connectors

        self.array_flag = '_array'  # Ensure only files with array structure are read
        self.ped_flag = '_pedthr_'
        self.data_flag = '_datrun_'

        if waveform_fit_func is None:
            # self.waveform_fit_func = 'waveform_func'
            self.waveform_fit_func = 'max_sample'
            # self.waveform_fit_func = 'parabola'
            # self.waveform_fit_func = 'parabola_vectorized'
            # self.waveform_fit_func = 'gaus'
        else:
            self.waveform_fit_func = waveform_fit_func
        self.noise_thresh_sigmas = 4

        self.fine_timestamp_constant = 1.0 / 6
        self.timestamp_frequency = 100e6  # Hz
        self.sample_period = 60  # ns

        self.channels_per_connector = 64
        self.starting_connector = min(self.feu_connectors)
        self.connector_channels = None  # Dictionary of connectors and channels to use for each connector

        self.ped_data = None
        self.data = None
        self.event_nums = None
        self.timestamps = None
        self.fine_timestamps = None
        self.fine_timestamp_corrected = False

        self.ped_means = None
        self.ped_sigmas = None
        self.noise_thresholds = None

        self.waveforms = None
        self.data_amps = None
        self.data_time_of_max = None
        self.data_fit_success = None
        self.fit_params = None
        self.event_amp_sums = None

        self.hits = None

        self.raw_amp_hist = None

    def copy(self):
        return copy.deepcopy(self)

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
            print('Warning: Multiple ped files found, using last one.')
            print(f'Ped files found: {ped_files}')
        ped_file_path = f'{ped_dir}{ped_files[-1]}'

        ped_data = read_det_data(ped_file_path)
        self.ped_data = self.split_det_data(ped_data, self.feu_connectors, starting_connector=1, to_connectors=False)

        self.get_pedestals()
        self.get_noise_thresholds(noise_sigmas=self.noise_thresh_sigmas)

    def get_pedestals(self):
        pedestals = get_pedestals_by_median(self.ped_data)
        ped_common_noise_sub = self.subtract_common_noise(self.ped_data, pedestals)
        ped_fits = get_pedestal_fits(ped_common_noise_sub)
        self.ped_means = ped_fits['mean']
        self.ped_sigmas = ped_fits['sigma']

    def subtract_common_noise(self, data, pedestals):
        # feu_connectors = np.array(self.feu_connectors) - min(self.feu_connectors) + 1  # Ensure connectors start at 1
        data_connectors = self.split_det_data(data, self.feu_connectors, to_connectors=True)
        peds_connectors = self.split_det_data(pedestals, self.feu_connectors, to_connectors=True)
        data_sub = []
        for data_connector, ped_connector in zip(data_connectors, peds_connectors):
            connector_common_noise = get_common_noise(data_connector, ped_connector)
            data_sub.append(data_connector - connector_common_noise[:, np.newaxis, :])
        return np.concatenate(data_sub, axis=1)

    def get_noise_thresholds(self, noise_sigmas=5):
        self.noise_thresholds = get_noise_thresholds(self.ped_sigmas, noise_sigmas)
        if self.connector_channels is not None:
            self.noise_thresholds = self.get_connector_channels(self.noise_thresholds)

    def read_data(self, file_nums=None, chunk_size=100, trigger_list=None, hist_raw_amps=False, save_waveforms=False):
        if self.data_dir is None:
            print('Error: No data directory specified.')
            return None
        data_files = get_good_files(os.listdir(self.data_dir), [self.data_flag, self.array_flag], self.feu_num, '.root',
                                    file_nums)

        if len(data_files) == 0:
            print('Error: No data files found.')
            return None

        def read_file(data_file, select_triggers=None, event_range=None):
            data_file_path = f'{self.data_dir}{data_file}'
            # data = read_det_data(data_file_path)
            # data_event_nums = read_det_data(data_file_path, tree_name='nt', variable_name='eventId')
            all_data = read_det_data_vars(data_file_path, ['amplitude', 'eventId', 'timestamp', 'ftst'], event_range=event_range)
            data_raw, data_event_nums = all_data['amplitude'], all_data['eventId']
            data_ft_stamp, data_timestamp = all_data['ftst'], all_data['timestamp']
            if select_triggers is not None:
                mask = np.isin(data_event_nums, select_triggers)
                data_raw, data_event_nums = data_raw[mask], data_event_nums[mask]
                data_ft_stamp, data_timestamp = data_ft_stamp[mask], data_timestamp[mask]
            data_raw = self.split_det_data(data_raw, self.feu_connectors, starting_connector=1, to_connectors=False)
            if self.ped_means is not None:
                data = self.subtract_common_noise(data_raw, self.ped_means)
                data = subtract_pedestal(data, self.ped_means)
            else:
                data = data_raw
            return data, data_raw, data_event_nums, data_timestamp, data_ft_stamp

        print('Reading in data...')

        n_sub_chunks = math.ceil(1 / chunk_size) if chunk_size < 1 else 1
        chunk_size = max(chunk_size, 1)
        for chunk_idx in range(0, len(data_files), chunk_size):
            chunk_files = data_files[chunk_idx:chunk_idx + chunk_size]

            n_tree_events = get_tree_entries(f'{self.data_dir}{chunk_files[0]}')  # Only matters if only 1 file, so get first
            for sub_chunk_i in range(n_sub_chunks):
                if n_sub_chunks == 1:
                    event_range_i = None
                else:
                    event_range_i = (int(sub_chunk_i * n_tree_events / n_sub_chunks), int((sub_chunk_i + 1) * n_tree_events / n_sub_chunks))
                    print(f'Getting sub chunk {sub_chunk_i + 1}/{n_sub_chunks}, events: {event_range_i[0]}-{event_range_i[1]} of {n_tree_events}')

                self.data = []
                if self.event_nums is None:
                    self.event_nums = []
                if self.timestamps is None:
                    self.timestamps = []
                if self.fine_timestamps is None:
                    self.fine_timestamps = []
                if save_waveforms and self.waveforms is None:
                    self.waveforms = []
                if self.event_amp_sums is None:
                    self.event_amp_sums = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    read_file_partial = partial(read_file, select_triggers=trigger_list, event_range=event_range_i)
                    for data_i, data_raw_i, event_nums, timestamps, ft_stamps in tqdm(executor.map(
                            read_file_partial, chunk_files), total=len(chunk_files)):
                        if self.connector_channels is not None:
                            data_i = self.get_connector_channels(data_i)
                        self.data.append(data_i)
                        self.event_nums.append(event_nums)
                        self.timestamps.append(timestamps)
                        self.fine_timestamps.append(ft_stamps)
                        self.event_amp_sums.append(np.nansum(data_i, axis=(1, 2)))
                        if save_waveforms:
                            self.waveforms.append(data_raw_i)
                        if hist_raw_amps:
                            n_events, n_channels, n_samples = data_raw_i.shape
                            data_raw_i = data_raw_i.transpose(1, 0, 2).reshape(n_channels, n_events * n_samples)
                            bins_y = np.arange(-0.5, 4096.5, 1)

                            histograms = np.array([np.histogram(data_raw_i[i], bins=bins_y)[0] for i in range(n_channels)])

                            if self.raw_amp_hist is None:
                                self.raw_amp_hist = histograms
                            else:
                                self.raw_amp_hist += histograms

                self.data = np.concatenate(self.data)

                print(f'Read in data shape: {self.data.shape}')
                print('Getting amplitudes...')
                self.get_event_amplitudes()

                # if save_waveforms:
                #     self.waveforms.append(self.data.copy())

                # Clear self.data to free up memory
                self.data = None
                print(f'Processed sub-chunk {sub_chunk_i + 1}/{n_sub_chunks}')
            print(f'Processed chunk {chunk_idx // chunk_size + 1}/{(len(data_files) + chunk_size - 1) // chunk_size}')

        self.event_nums = np.concatenate(self.event_nums)
        self.timestamps = np.concatenate(self.timestamps)
        self.timestamps = self.timestamps / self.timestamp_frequency  # Convert to seconds
        self.fine_timestamps = np.concatenate(self.fine_timestamps)
        self.event_amp_sums = np.concatenate(self.event_amp_sums)
        if save_waveforms:
            self.waveforms = np.concatenate(self.waveforms)
        # self.correct_for_fine_timestamps()
        print('Getting hits...')
        self.get_hits()

    def subtract_pedestals_from_data(self):
        self.data = subtract_pedestal(self.data, self.ped_means)

    def get_event_amplitudes(self):
        start = time()

        num_chunks = max(os.cpu_count() - 1, 1)
        data_chunks = np.array_split(self.data, num_chunks, axis=0)

        def process_chunk(chunk):
            return get_waveform_fits(chunk, self.noise_thresholds, self.waveform_fit_func)

        fits_list = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for fits in tqdm(executor.map(process_chunk, data_chunks), total=num_chunks):
                fits_list.append(fits)

        fit_params = {key: np.concatenate([fits[key] for fits in fits_list], axis=0) for key in fits_list[0].keys()}

        if self.data_amps is None:
            self.fit_params = fit_params
            self.data_amps = fit_params['amplitude']
            self.data_time_of_max = fit_params['time_max']
            self.data_fit_success = fit_params['success'] != 0
        else:
            for key in fit_params.keys():
                self.fit_params[key] = np.concatenate([self.fit_params[key], fit_params[key]], axis=0)
            self.data_amps = np.concatenate([self.data_amps, fit_params['amplitude']], axis=0)
            self.data_time_of_max = np.concatenate([self.data_time_of_max, fit_params['time_max']], axis=0)
            self.data_fit_success = np.concatenate([self.data_fit_success, fit_params['success'] != 0], axis=0)

        print(f'Fitting time: {time() - start} s')

    def correct_for_fine_timestamps(self, force=False):
        """
        Correct data_time_of_max and fit_params['time_max'] for fine timestamps.
        :param force: Force correction even if already corrected.
        :return:
        """
        if force or not self.fine_timestamp_corrected:
            self.data_time_of_max = self.data_time_of_max + self.fine_timestamps[:, np.newaxis] * self.fine_timestamp_constant
            self.fit_params['time_max'] = self.fit_params['time_max'] + self.fine_timestamps[:, np.newaxis] * self.fine_timestamp_constant
            self.fine_timestamp_corrected = True

    def uncorrect_for_fine_timestamps(self, force=False):
        """
        Uncorrect data_time_of_max and fit_params['time_max'] for fine timestamps.
        :param force: Force uncorrection even if already uncorrected.
        :return:
        """
        if force or self.fine_timestamp_corrected:
            self.data_time_of_max = self.data_time_of_max - self.fine_timestamps[:, np.newaxis] * self.fine_timestamp_constant
            self.fit_params['time_max'] = self.fit_params['time_max'] - self.fine_timestamps[:, np.newaxis] * self.fine_timestamp_constant
            self.fine_timestamp_corrected = False

    def get_hits(self, amp_min=0, time_max_range=None):
        """
        Get hits for each channel in each event.
        Currently, most of the work is handled by get_event_amplitudes, which masks noise.
        Here simply get channels in each event with amplitude above amp_min and with
        time_max within time_max_range.
        :param amp_min: Minimum amplitude for hit.
        :param time_max_range: Tuple of min and max time of max values.
        """
        self.hits = self.data_amps > amp_min
        if time_max_range is not None:
            good_time_max = (time_max_range[0] < self.data_time_of_max) & (self.data_time_of_max < time_max_range[1])
            self.hits = self.hits & good_time_max

    def get_channels_amps(self, connector, channels):
        """
        Get amplitudes for specified channels on connector in each event.
        :param connector: Connector number.
        :param channels: List or array of channels.
        :return: Amplitudes for specified channels in each event.
        """
        channels = np.array(channels)
        if isinstance(connector, int):
            channel_amps = self.split_det_data(self.data_amps, [connector], to_connectors=False)[:, channels]
        else:
            connector = np.array(connector)
            channel_amps = self.split_det_data_connectors(self.data_amps, connector, channels)
        return channel_amps

    def get_channels_hits(self, connector, channels):
        """
        Get hits for specified channels on connector in each event.
        :param connector: Connector number.
        :param channels: List or array of channels.
        :return: Hits for specified channels in each event.
        """
        channels = np.array(channels)
        if isinstance(connector, int):
            channel_hits = self.split_det_data(self.hits, [connector], to_connectors=False)[:, channels]
        else:
            channel_hits = self.split_det_data_connectors(self.hits, connector, channels)
        return channel_hits

    def get_channels_time_of_max(self, connector, channels):
        """
        Get time of max for specified channels on connector in each event.
        :param connector: Connector number.
        :param channels: List or array of channels.
        :return: Time of max for specified channels in each event.
        """
        channels = np.array(channels)
        if isinstance(connector, int):
            channel_time_of_max = self.split_det_data(self.data_time_of_max, [connector], to_connectors=False)[:, channels]
        else:
            channel_time_of_max = self.split_det_data_connectors(self.data_time_of_max, [connector], channels)
        return channel_time_of_max

    def split_det_data(self, det_data, feu_connectors, to_connectors=False, starting_connector=None):
        if starting_connector is None:
            starting_connector = self.starting_connector
        channel_list = np.concatenate([np.arange(self.channels_per_connector) +
                                       self.channels_per_connector * (connector_num - starting_connector)
                                       for connector_num in feu_connectors])
        if det_data.ndim == 1:
            det_data = det_data[channel_list]
            if to_connectors:
                det_data = np.array(np.split(det_data, len(feu_connectors)))
        elif det_data.ndim == 2:
            det_data = det_data[:, channel_list]
            if to_connectors:
                det_data = np.array(np.split(det_data, len(feu_connectors), axis=1))
        elif det_data.ndim == 3:
            det_data = det_data[:, channel_list]
            if to_connectors:
                det_data = np.array(np.split(det_data, len(feu_connectors), axis=1))
        else:
            print('Error: Data shape not recognized.')
            return None

        return det_data

    def split_det_data_connectors(self, det_data, connectors, channels):
        """
        Given list of connectors and channels, get det_data for these channels.
        Just add an offset to the channels based on the connector number.
        :param det_data:
        :param connectors:
        :param channels:
        :return:
        """
        channels = (np.array(connectors) - self.starting_connector) * self.channels_per_connector + np.array(channels)
        if det_data.ndim == 1:
            return det_data[channels]
        elif det_data.ndim >= 2:
            return det_data[:, channels]
        else:
            print('Error: Data shape not recognized.')
            return None

    def get_connector_channels(self, det_data):
        """
        Get specific channels for each connector.
        Args:
            det_data:

        Returns:

        """
        if self.connector_channels is not None:
            data_channels = []
            for feu_connector, channels in self.connector_channels.items():
                channel_offset = (feu_connector - self.starting_connector) * self.channels_per_connector
                if det_data.ndim == 1:
                    data_channels.append(det_data[channels + channel_offset])
                elif det_data.ndim >= 2:
                    data_channels.append(det_data[:, channels + channel_offset])
            return np.concatenate(data_channels)
        else:
            return det_data

    def get_event_nums(self):
        return self.event_nums

    def get_timestamps(self):
        return self.timestamps

    def filter_sparks(self, spark_filter_sigma=15, plot=True):
        """
        Filter sparks, defined as events with more than 10 hits.
        :param spark_filter_sigma: Number of standard deviations above mean to set spark threshold.
        :param plot: Whether to plot the spark threshold.
        :return:
        """
        spark_thresh = np.mean(self.event_amp_sums) + spark_filter_sigma * np.std(self.event_amp_sums)
        event_spark_filter = self.event_amp_sums < spark_thresh
        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(self.event_amp_sums, bins=100)
            ax.axvline(spark_thresh, color='black', ls='--', label='Spark Event Amplitude Sum Threshold')
            ax.legend()
            ax.set_yscale('log')
            ax.set_xlabel('Sum of All Samples of All Channels per Event')
            ax.set_ylabel('Number of Events')
            ax.set_title('Spark Metric: Events with large sample:channel sums are probably sparks')
            fig.tight_layout()
        self.filter_data(event_spark_filter)

    def filter_data(self, data_indices):
        """
        Filter data, keep only data at indices.
        :param data_indices:
        :return:
        """
        self.data_amps = self.data_amps[data_indices]
        self.data_time_of_max = self.data_time_of_max[data_indices]
        self.data_fit_success = self.data_fit_success[data_indices]
        self.event_nums = self.event_nums[data_indices]
        self.hits = self.hits[data_indices]
        self.event_amp_sums = self.event_amp_sums[data_indices]
        self.timestamps = self.timestamps[data_indices]
        self.fine_timestamps = self.fine_timestamps[data_indices]
        for key in self.fit_params.keys():
            self.fit_params[key] = self.fit_params[key][data_indices]
        if self.waveforms is not None:
            self.waveforms = self.waveforms[data_indices]

    def plot_event_amplitudes(self, channel=None, channels=None, logy=False):
        if channels is not None:
            amps = self.data_amps[:, channels]
        else:
            amps = self.data_amps
        amps = np.ravel(amps) if channel is None else amps[:, channel]
        amps = amps[~np.isnan(amps)]
        fig, ax = plt.subplots()
        ax.hist(amps, bins=100)
        if logy:
            ax.set_yscale('log')
        ax.set_title('Event Amplitudes')
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Counts')
        fig.tight_layout()

    def plot_event_time_maxes_extravagant(self, channel=None, channels=None, max_channel=False, min_amp=None):
        print(f'data_time_of_max.shape: {self.data_time_of_max.shape}')
        print(f'data_amps.shape: {self.data_amps.shape}')

        if channels is not None:
            time_maxes = self.data_time_of_max[:, channels]
            amps = self.data_amps[:, channels]
        else:
            time_maxes = self.data_time_of_max
            amps = self.data_amps

        print(f'time_maxes.shape: {time_maxes.shape}')
        print(f'amps.shape: {amps.shape}')

        fig, ax = plt.subplots()
        time_of_max = np.ravel(time_maxes)
        time_of_max = time_of_max[~np.isnan(time_of_max) & (time_of_max > -1) & (time_of_max < 40)]
        ax.hist(time_of_max, bins=100)
        ax.set_title('Event Time of Max')
        ax.set_xlabel('Time')
        ax.set_ylabel('Counts')
        fig.tight_layout()

        # plt.show()

        if min_amp is not None:
            time_maxes = time_maxes[np.max(amps, axis=1) > min_amp]
            amps = amps[np.max(amps, axis=1) > min_amp]

        print(f'time_maxes.shape: {time_maxes.shape}')
        print(f'amps.shape: {amps.shape}')

        if max_channel:
            # Get amp and time_max of the channel with the max amplitude in each event

            # Filter out events where all amplitudes are nan
            time_maxes = time_maxes[~np.all(np.isnan(amps), axis=1)]
            amps = amps[~np.all(np.isnan(amps), axis=1)]

            max_amp_channels = np.nanargmax(amps, axis=1)
            amps = np.nanmax(amps, axis=1)
            time_maxes = time_maxes[np.arange(len(time_maxes)), max_amp_channels]
        elif channel is not None:
            time_maxes = time_maxes[:, channel]
            amps = None  # Not Implemented
        else:
            time_maxes = np.ravel(time_maxes)
            amps = None  # Not Implemented

        print(f'time_maxes.shape: {time_maxes.shape}')
        print(f'amps.shape: {amps.shape}')

        print(amps)
        print(time_maxes)

        time_maxes = time_maxes[~np.isnan(time_maxes)]

        fig, ax = plt.subplots()
        time_of_max = np.ravel(time_maxes)
        time_of_max = time_of_max[~np.isnan(time_of_max) & (time_of_max > -1) & (time_of_max < 40)]
        ax.hist(time_of_max, bins=100)
        ax.set_title('Event Time of Max')
        ax.set_xlabel('Time')
        ax.set_ylabel('Counts')
        fig.tight_layout()

        # plt.show()

        time_maxes = np.ravel(time_maxes)
        amps = np.ravel(amps) if amps is not None else None

        # Filter out when time_max not > 0 and < 32
        time_max_range_filter = (0 <= time_maxes) & (time_maxes <= 32)
        time_maxes = time_maxes[time_max_range_filter]
        amps = amps[time_max_range_filter]

        fig, ax = plt.subplots()
        ax.hist(time_maxes, bins=100)
        ax.set_title('Event Mean Times')
        ax.set_xlabel('Time')
        ax.set_ylabel('Counts')
        fig.tight_layout()

        if amps is not None:
            # Plot 2D of time max vs amplitude
            fig, ax = plt.subplots()
            ax.hist2d(amps, time_maxes, bins=100, norm=LogNorm())
            ax.set_title('Event Mean Times vs Amplitudes')
            ax.set_ylabel('Time')
            ax.set_xlabel('Amplitude')
            fig.tight_layout()

            # Now bin time_maxes by amplitude. Get the mean and standard deviation for each bin and plot this as an
            # error bar plot.
            amp_bins = np.linspace(0, np.max(amps), 100)
            time_maxes_binned = [np.ravel(time_maxes[(amp_bins[i] < amps) & (amps < amp_bins[i + 1])]) for i in range(len(amp_bins) - 1)]
            n_points = np.array([len(time_maxes_bin) for time_maxes_bin in time_maxes_binned])
            time_maxes_mean = np.array([np.mean(time_maxes_bin) for time_maxes_bin in time_maxes_binned])
            time_maxes_std = np.array([np.std(time_maxes_bin) for time_maxes_bin in time_maxes_binned])

            time_maxes_mean = np.where(n_points > 10, time_maxes_mean, np.nan)

            fig, ax = plt.subplots()
            ax.errorbar(amp_bins[:-1], time_maxes_mean, yerr=time_maxes_std, fmt='o')
            ax.set_title('Event Mean Times vs Amplitudes')
            ax.set_ylabel('Time')
            ax.set_xlabel('Amplitude')
            fig.tight_layout()

    def get_event_time_maxes(self, channel=None, channels=None, max_channel=False, first_channel=False, min_amp=None, max_amp=None, filter_times=True, event_filter=None):
        if channels is not None:
            time_maxes = self.data_time_of_max[:, channels]
            amps = self.data_amps[:, channels]
        else:
            time_maxes = self.data_time_of_max
            amps = self.data_amps

        if event_filter is not None:
            time_maxes = time_maxes[event_filter]
            amps = amps[event_filter]

        if min_amp is not None:
            # Filter out events where all amplitudes are nan
            time_maxes = time_maxes[~np.all(np.isnan(amps), axis=1)]
            amps = amps[~np.all(np.isnan(amps), axis=1)]

            min_amp_mask = np.nanmax(amps, axis=1) > min_amp
            time_maxes = time_maxes[min_amp_mask]
            amps = amps[min_amp_mask]

        if max_amp is not None:
            # Filter out events where all amplitudes are nan
            time_maxes = time_maxes[~np.all(np.isnan(amps), axis=1)]
            amps = amps[~np.all(np.isnan(amps), axis=1)]

            max_amp_mask = np.nanmax(amps, axis=1) < max_amp
            time_maxes = time_maxes[max_amp_mask]
            amps = amps[max_amp_mask]

        if max_channel:
            # Filter out events where all amplitudes are nan
            # time_maxes = time_maxes[~np.all(np.isnan(amps), axis=1)]
            # amps = amps[~np.all(np.isnan(amps), axis=1)]
            amps_no_nan = np.where(np.isnan(amps), -np.inf, amps)

            # Get amp and time_max of the channel with the max amplitude in each event
            # max_amp_channels = np.nanargmax(amps, axis=1)
            max_amp_channels = np.nanargmax(amps_no_nan, axis=1)
            time_maxes = time_maxes[np.arange(len(time_maxes)), max_amp_channels]
        elif first_channel:
            # Fileter out events where all times are nan
            time_maxes = time_maxes[~np.all(np.isnan(time_maxes), axis=1)]

            # Get amp and time_max of the channel with the earliest time_max in each event
            first_time_channels = np.nanargmin(time_maxes, axis=1)
            time_maxes = time_maxes[np.arange(len(time_maxes)), first_time_channels]
        elif channel is not None:
            time_maxes = time_maxes[:, channel]
        else:
            time_maxes = np.ravel(time_maxes)

        return np.ravel(time_maxes)

        # if filter_times:
        #     time_maxes = time_maxes[~np.isnan(time_maxes)]
        #
        #     time_of_max = np.ravel(time_maxes)
        #     time_of_max = time_of_max[(time_of_max > -1) & (time_of_max < 40)] * self.sample_period
        #     original_num = len(time_of_max)
        #     time_of_max = time_of_max[(time_of_max > 3 * self.sample_period) & (time_of_max < 15 * self.sample_period)]
        #     cut_num = len(time_of_max)
        #     if cut_num / original_num < 0.9:
        #         print(f'Warning: Cut {1 - cut_num / original_num * 100:.2f}% of events.')
        #
        #     return time_of_max
        # else:
        #     return np.ravel(time_maxes)

    def filter_time_maxes(self, time_of_max, return_mask=False):
        # time_of_max = time_of_max[(time_of_max > -1) & (time_of_max < 40)] * self.sample_period
        # original_num = len(time_of_max)
        # time_of_max = time_of_max[(time_of_max > 3 * self.sample_period) & (time_of_max < 15 * self.sample_period)]
        # cut_num = len(time_of_max)
        # if cut_num / original_num < 0.9:
        #     print(f'Warning: Cut {1 - cut_num / original_num * 100:.2f}% of events.')

        time_of_max = np.where(np.isnan(time_of_max), -np.inf, time_of_max)
        loose_sample_mask = (time_of_max > -1) & (time_of_max < 40)
        original_num = len(time_of_max[loose_sample_mask])
        tight_sample_mask = (time_of_max > 3) & (time_of_max < 15)
        time_of_max = time_of_max[tight_sample_mask] * self.sample_period
        cut_num = len(time_of_max)
        if cut_num / original_num < 0.9:
            print(f'Warning: Cut {1 - cut_num / original_num * 100:.2f}% of events.')

        if return_mask:
            return time_of_max, tight_sample_mask

        return time_of_max

    def plot_event_time_maxes(self, channel=None, channels=None, max_channel=False, first_channel=False, min_amp=None, max_amp=None, event_filter=None, plot=True):
        # if channels is not None:
        #     time_maxes = self.data_time_of_max[:, channels]
        #     amps = self.data_amps[:, channels]
        # else:
        #     time_maxes = self.data_time_of_max
        #     amps = self.data_amps
        #
        # if min_amp is not None:
        #     # Filter out events where all amplitudes are nan
        #     time_maxes = time_maxes[~np.all(np.isnan(amps), axis=1)]
        #     amps = amps[~np.all(np.isnan(amps), axis=1)]
        #
        #     min_amp_mask = np.nanmax(amps, axis=1) > min_amp
        #     time_maxes = time_maxes[min_amp_mask]
        #     amps = amps[min_amp_mask]
        #
        # if max_channel:
        #     # Filter out events where all amplitudes are nan
        #     time_maxes = time_maxes[~np.all(np.isnan(amps), axis=1)]
        #     amps = amps[~np.all(np.isnan(amps), axis=1)]
        #
        #     # Get amp and time_max of the channel with the max amplitude in each event
        #     max_amp_channels = np.nanargmax(amps, axis=1)
        #     time_maxes = time_maxes[np.arange(len(time_maxes)), max_amp_channels]
        # elif channel is not None:
        #     time_maxes = time_maxes[:, channel]
        # else:
        #     time_maxes = np.ravel(time_maxes)
        #
        # time_maxes = time_maxes[~np.isnan(time_maxes)]

        # time_of_max = np.ravel(time_maxes)
        # time_of_max = time_of_max[(time_of_max > -1) & (time_of_max < 40)] * self.sample_period
        # original_num = len(time_of_max)
        # time_of_max = time_of_max[(time_of_max > 3 * self.sample_period) & (time_of_max < 15 * self.sample_period)]
        # cut_num = len(time_of_max)
        # if cut_num / original_num < 0.9:
        #     print(f'Warning: Cut {1 - cut_num / original_num * 100:.2f}% of events.')

        time_of_max = self.get_event_time_maxes(channel=channel, channels=channels, max_channel=max_channel, first_channel=False, min_amp=min_amp, max_amp=max_amp, event_filter=event_filter)
        time_of_max = self.filter_time_maxes(time_of_max)

        # Make numpy histogram and fit to gaussian
        hist, bins = np.histogram(time_of_max, bins=100)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        p0 = [np.max(hist), np.mean(time_of_max), np.std(time_of_max)]

        if plot:
            fig, ax = plt.subplots()
            ax.bar(bin_centers, hist, width=bins[1] - bins[0], align='center', color='blue', alpha=0.5)
            ax.set_title('Event Time of Max')
            ax.set_xlabel('Time [ns]')
            ax.set_ylabel('Counts')

        sigma, sigma_err = np.nan, np.nan
        try:
            popt, pcov = cf(gaussian, bin_centers, hist, p0=p0)
            fit_meas = [Measure(val, err) for val, err in zip(popt, np.sqrt(np.diag(pcov)))]
            if plot:
                fit_str = f'Amplitude: {fit_meas[0]}\nMean: {fit_meas[1]}\nSigma: {fit_meas[2]}'
                fit_xs = np.linspace(bins[0], bins[-1], 1000)
                ax.plot(fit_xs, gaussian(fit_xs, *popt), color='red')
                ax.annotate(fit_str, (0.9, 0.9), xycoords='axes fraction', ha='right', va='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            sigma, sigma_err = fit_meas[2].val, fit_meas[2].err
        except RuntimeError:
            print('Error: Gaussian fit failed.')
            if plot:
                ax.plot(bin_centers, gaussian(bin_centers, *p0), color='gray')
        if plot:
            fig.tight_layout()

        return sigma, sigma_err


    def plot_fit_param(self, param, params_ranges=None, channel=None):
        selection_data = self.get_selected_fit_data(param, params_ranges, channel)
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

    def plot_noise_metric(self):
        """
        Plot metric for noise in data.
        :return:
        """
        event_amp_sums = np.nansum(self.data_amps, axis=1)
        event_hits = np.nansum(self.hits, axis=1)
        event_max_amps = np.nanmax(self.data_amps, axis=1)
        nan_mask = np.isnan(event_amp_sums) | np.isnan(event_hits) | np.isnan(event_max_amps)
        event_amp_sums = event_amp_sums[~nan_mask]
        event_hits = event_hits[~nan_mask]
        event_max_amps = event_max_amps[~nan_mask]

        fig, ax = plt.subplots()
        ax.plot(event_amp_sums)
        ax.set_title('Event Amplitude Sums')
        ax.set_xlabel('Event')
        ax.set_ylabel('Amplitude Sum')
        fig.tight_layout()

        fig, ax = plt.subplots()
        ax.hist(event_amp_sums, bins=100)
        ax.set_title('Event Amplitude Sums')
        ax.set_xlabel('Amplitude Sum')
        ax.set_ylabel('Events')
        fig.tight_layout()

        fig, ax = plt.subplots()
        ax.scatter(event_hits, event_amp_sums, alpha=0.5)
        ax.set_title('Event Amplitude Sums vs Hits')
        ax.set_xlabel('Hits')
        ax.set_ylabel('Amplitude Sum')
        fig.tight_layout()


        fig, ax = plt.subplots()
        ax.scatter(event_max_amps, event_amp_sums, alpha=0.5)
        ax.set_title('Event Amplitude Sums vs Max Amplitudes')
        ax.set_xlabel('Max Amplitude')
        ax.set_ylabel('Amplitude Sum')
        fig.tight_layout()

        fig, ax = plt.subplots()
        ax.scatter(event_hits, event_max_amps, alpha=0.5)
        ax.set_title('Event Max Amplitudes vs Hits')
        ax.set_xlabel('Hits')
        ax.set_ylabel('Max Amplitude')
        fig.tight_layout()

        max_amp_div_sum = event_max_amps / event_amp_sums
        fig, ax = plt.subplots()
        ax.hist(max_amp_div_sum, bins=100)
        ax.set_title('Max Amplitude / Amplitude Sum')
        ax.set_xlabel('Max Amplitude / Amplitude Sum')
        ax.set_ylabel('Events')
        fig.tight_layout()

        fig, ax = plt.subplots()
        ax.scatter(range(len(max_amp_div_sum)), max_amp_div_sum, alpha=0.5)
        ax.set_title('Max Amplitude / Amplitude Sum')
        ax.set_xlabel('Event')
        ax.set_ylabel('Max Amplitude / Amplitude Sum')
        fig.tight_layout()

    def plot_hits_vs_strip(self, print_dead_strips=False):
        """
        Plot number of hits vs strip, separated by connector.
        :param print_dead_strips: Print strips with no hits.
        :return:
        """
        hits = np.sum(self.hits, axis=0)
        # Separate hits into groups of self.channels_per_connector
        hits = np.array(np.split(hits, len(hits) // self.channels_per_connector))
        fig, ax = plt.subplots()
        for i, hit_group in enumerate(hits):
            strip_nums = np.arange(len(hit_group)) + i * self.channels_per_connector
            ax.plot(strip_nums, hit_group, label=f'Connector {i + self.starting_connector}')
        ax.set_title('Hits vs Strip')
        ax.set_xlabel('Strip')
        ax.set_ylabel('Hits')
        ax.set_ylim(0, None)
        ax.legend()
        fig.tight_layout()

        if print_dead_strips:
            percent_max = 2.  # percent of max to call a strip dead
            max_hits = max((max(hit_group) for hit_group in hits))
            for i, hit_group in enumerate(hits):
                strip_nums = np.arange(len(hit_group)) + i * self.channels_per_connector
                dead_strips = strip_nums[hit_group < max_hits * percent_max / 100]
                print(f'Connector {i + self.starting_connector} Dead Strips: {dead_strips}')

    def plot_amplitudes_vs_strip(self, events=None):
        """
        Plot 2D histogram of amplitude (y-axis) vs strip (x-axis), separated by connector. For each strip, make a 1D
        histogram of amplitudes. Bins of 0 counts are removed. Then plot these histograms for each strip on the x-axis.
        :return:
        """
        bins = np.arange(1, 4401, 20)
        data_amps = self.data_amps if events is None else self.data_amps[np.isin(self.event_nums, events)]
        amps = np.ravel(data_amps)
        strip_nums = np.tile(np.arange(data_amps.shape[1]), (data_amps.shape[0], 1))
        strip_nums = np.ravel(strip_nums)
        fig, ax = plt.subplots(figsize=(10, 5))
        print(strip_nums.size)
        print(amps.size)
        h = ax.hist2d(strip_nums, amps, bins=[np.arange(-0.5, data_amps.shape[1] + 0.5, 1), bins],
                      cmin=1, cmap='jet', norm=LogNorm())
        ax.set_title('Amplitudes vs Strip')
        ax.set_xlabel('Strip')
        ax.set_ylabel('Amplitude')
        try:
            cbar = fig.colorbar(h[3], ax=ax)
            cbar.set_label('Counts')
        except ValueError:
            pass

        fig.tight_layout()

    def get_selected_data(self, params_ranges=None, channel=None):
        """
        Get data that fits within specified parameter ranges.
        :return:
        """
        data, success, fit_params = self.waveforms, self.data_fit_success, self.fit_params
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

    def get_selected_fit_data(self, param, params_ranges=None, channel=None):
        """
        Get data that fits within specified parameter ranges.
        :return:
        """
        success, fit_params = self.data_fit_success, self.fit_params
        if channel is not None:
            success = success[:, channel]
            params = {key: val[:, channel] for key, val in fit_params.items()}
        else:
            success = np.ravel(success)
            params = {key: np.ravel(val) for key, val in fit_params.items()}
        selection_mask = success
        if params_ranges is not None:
            for par, param_range in params_ranges.items():
                selection_mask = selection_mask & (param_range[0] < params[par]) & (params[par] < param_range[1])
        selection_data = params[param][selection_mask]

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

    def plot_common_noise(self, connector=0):
        """
        Plot common noise vs total sample number for a connector.
        :param connector:
        :return:
        """
        pedestals = get_pedestals_by_median(self.ped_data)
        data_connectors = self.split_det_data(self.ped_data, self.feu_connectors, to_connectors=True)
        peds_connectors = self.split_det_data(pedestals, self.feu_connectors, to_connectors=True)
        print(f'Connector {connector + self.starting_connector} Data: {data_connectors[connector].shape}')
        print(f'Connector {connector + self.starting_connector} Peds: {peds_connectors[connector].shape}')
        connector_common_noise = get_common_noise(data_connectors[connector], peds_connectors[connector])
        print(f'Connector {connector + self.starting_connector} Common Noise: {connector_common_noise.shape}')
        fig, ax = plt.subplots()
        ax.plot(connector_common_noise)
        ax.set_title(f'Connector {connector + self.starting_connector} Common Noise')
        ax.set_xlabel('Pedestal Event Number')
        ax.set_ylabel('Common Noise')
        fig.tight_layout()

        fig, ax = plt.subplots()
        ax.plot(connector_common_noise.flatten())
        ax.set_title(f'Connector {connector + self.starting_connector} Common Noise')
        ax.set_xlabel('Cumulative Sample Number')
        ax.set_ylabel('Common Noise')
        fig.tight_layout()

    # def plot_raw_amps_2d_hist(self):
    #     """
    #     Plot a 2D histogram of the raw amplitudes for each channel in the detector.
    #     :return:
    #     """
    #     if self.raw_amp_hist is None:
    #         print('Error: Raw amplitude histogram not created.')
    #         return
    #     fig, ax = plt.subplots()
    #     ax.set_title(f'Raw Amplitudes')
    #     ax.set_xlabel('Channel')
    #     ax.set_ylabel('Amplitude')
    #     # If there are 0 events make the color white
    #     cmap = plt.cm.jet
    #     cmap.set_under('w')
    #     # set log scale on color bar
    #     n_channels = self.raw_amp_hist.shape[0]
    #     h = ax.imshow(self.raw_amp_hist.T, origin='lower', aspect='auto', cmap=cmap,
    #                   extent=[-0.5, n_channels, -0.5, 4096.5], norm=LogNorm(vmin=1))
    #     cbar = fig.colorbar(h, ax=ax)
    #     fig.tight_layout()

    def plot_raw_amps_2d_hist(self, combine_y: int = 1):
        """
        Plot a 2D histogram of the raw amplitudes for each channel in the detector.

        :param combine_y: Number of bins to combine along the y-axis (amplitude).
                          Must be >= 1. Default is 1 (no combining).
        """
        if self.raw_amp_hist is None:
            print('Error: Raw amplitude histogram not created.')
            return

        hist = self.raw_amp_hist

        # --- Rebin along y if requested ---
        if combine_y > 1:
            n_channels, n_bins_y = hist.shape
            new_bins_y = n_bins_y // combine_y
            hist = hist[:, :new_bins_y * combine_y]  # trim to multiple of combine_y
            hist = hist.reshape(n_channels, new_bins_y, combine_y).sum(axis=2)

        fig, ax = plt.subplots()
        ax.set_title(f'Raw Amplitudes')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Amplitude')

        cmap = plt.cm.jet
        cmap.set_under('w')

        n_channels, n_bins_y = hist.shape
        h = ax.imshow(
            hist.T,
            origin='lower',
            aspect='auto',
            cmap=cmap,
            extent=[-0.5, n_channels - 0.5, -0.5, combine_y * n_bins_y - 0.5],
            norm=LogNorm(vmin=1)
        )

        cbar = fig.colorbar(h, ax=ax)
        fig.tight_layout()

    def plot_waveforms(self, event_num):
        """
        Plot waveforms for specific event
        :param self:
        :param event_num:
        :return:
        """
        num_connectors = len(self.feu_connectors)

        # Create a figure and axes
        fig, axes = plt.subplots(num_connectors, 1, figsize=(10, 8), sharex=True)
        if num_connectors == 1:
            axes = [axes]

        if self.connector_channels is not None:
            waveforms_split_hold = self.waveforms
            # split into connectors based on the length of each array in the connector_channels dictionary
            waveforms_split = []
            channel_index = 0
            for connector, channels in self.connector_channels.items():
                waveforms_split.append(waveforms_split_hold[:, channel_index:channel_index + len(channels)])
                channel_index += len(channels)
        else:
            waveforms_split = self.split_det_data(self.waveforms, self.feu_connectors, to_connectors=True)
        min_amp, max_amp = 0, 100

        for connector_i, (ax, waveform_connector) in enumerate(zip(axes, waveforms_split)):
            colors = plt.cm.brg(np.linspace(0, 0.6, len(waveform_connector[event_num])))  # Create colors for each strip
            for strip_i, waveform in enumerate(waveform_connector[event_num]):
                color = colors[strip_i]  # Use colormap color for each strip
                if strip_i == 0:
                    ax.plot(waveform, label=f'Connector {self.starting_connector + connector_i}', color=color)
                else:
                    ax.plot(waveform, color=color)
                min_amp = min(min_amp, np.min(waveform))
                max_amp = max(max_amp, np.max(waveform))

            # Add labels and text
            if connector_i == num_connectors - 1:
                ax.set_xlabel('Sample')
            if connector_i == 0:  # Print the event number
                ax.text(0.01, 0.98, f'Event {event_num}', ha='left', va='top', transform=ax.transAxes)
            ax.set_ylabel('Amplitude')
            ax.legend(loc='upper right')

        # Set consistent y-limits for all plots
        min_max_range = max_amp - min_amp
        for ax in axes:
            ax.set_ylim(min_amp - 0.05 * min_max_range, max_amp + 0.05 * min_max_range)

        # Adjust layout
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)

    def plot_fine_timestamp_hist(self):
        """
        Plot histogram of fine timestamps.
        :return:
        """
        fine_time_stamps = np.ravel(self.fine_timestamps)
        fig, ax = plt.subplots()
        ax.hist(fine_time_stamps, bins=100)
        ax.set_title('Fine Timestamps')
        ax.set_xlabel('Fine Timestamp')
        ax.set_ylabel('Counts')
        fig.tight_layout()


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


def read_det_data_vars(file_path, variables, tree_name='nt', event_range=None):
    vector.register_awkward()
    # Open the ROOT file with uproot
    root_file = uproot.open(file_path)

    # Access the tree in the file
    tree = root_file[tree_name]

    # Get the variable data from the tree
    if event_range is None:
        variable_data = tree.arrays(variables, library='np')
    else:
        variable_data = tree.arrays(variables, library='np', entry_start=event_range[0], entry_stop=event_range[-1])
    variable_data = {key: val.astype(np.float32) for key, val in variable_data.items()}
    root_file.close()

    return variable_data


def get_tree_entries(file_path, tree_name='nt'):
    # Open the ROOT file with uproot
    root_file = uproot.open(file_path)

    # Access the tree in the file
    tree = root_file[tree_name]

    # Get the number of entries in the tree
    num_entries = tree.num_entries

    return num_entries


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
        param_names = ['amplitude', 'time_max', 'sigma', 'amplitude_err', 'time_max_err', 'sigma_err', 'success']
    elif func == 'waveform_func':
        fits = np.apply_along_axis(fit_waveform_func, -1, data)
        param_names = ['amplitude', 'time_max', 'time_shift', 'q', 'amplitude_err', 'time_max_err', 'time_shift_err',
                       'q_err', 'success']
    elif func == 'parabola':
        fits = np.apply_along_axis(fit_waveform_parabola, -1, data)
        param_names = ['amplitude', 'time_max', 'success']
    elif func == 'parabola_vectorized':
        fits = fit_waveform_parabola_vectorized(data)
        param_names = ['amplitude', 'time_max', 'success']
    elif func == 'max_sample':
        # fits = np.apply_along_axis(get_waveform_max_sample, -1, data)
        amplitude = np.max(data, axis=-1)
        time_max = np.argmax(data, axis=-1)
        success = np.ones((data.shape[0], data.shape[1]))
        fits = np.stack([amplitude, time_max, success], axis=-1)
        param_names = ['amplitude', 'time_max', 'success']
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
    sd = len(waveform) / 2
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


def fit_waveform_cubic(waveform):
    """
    Get the maximum amplitude and the two adjacent points to the maximum amplitude. Using the cubic equation get the
    maximum amplitude and the time of the maximum amplitude.
    :param waveform:
    :return:
    """
    amplitude = np.max(waveform)
    if amplitude == 0:
        return np.nan, np.nan, False
    max_index = np.argmax(waveform)
    if max_index == 0 or max_index == len(waveform) - 1:
        return amplitude, max_index, False
    x = np.arange(max_index - 1, max_index + 2)
    y = waveform[x]

    # Unfinished


def fit_waveform_parabola(waveform):
    """
    Get the maximum amplitude and the two adjacent points to the maximum amplitude. Using the parabolic equation get the
    maximum amplitude and the time of the maximum amplitude.
    :param waveform:
    :return:
    """
    amplitude = np.max(waveform)
    if amplitude == 0:
        return np.nan, np.nan, False
    max_index = np.argmax(waveform)
    if max_index == 0 or max_index == len(waveform) - 1:
        return amplitude, max_index, False
    x = np.arange(max_index - 1, max_index + 2)
    y = waveform[x]

    x_vertex, y_vertex = calc_parabola_vertex(x[0], y[0], x[1], y[1], x[2], y[2])
    # print(f'points: {x} {y} vertex: {x_vertex} {y_vertex}')

    return y_vertex, x_vertex, True


def fit_waveform_parabola_vectorized(data):
    max_indices = np.argmax(data, axis=-1)
    amplitudes = np.max(data, axis=-1)

    # Identify valid indices (exclude borders)
    valid_mask = (max_indices > 0) & (max_indices < data.shape[-1] - 1)

    # Gather x and y values for parabola fitting
    x_left = max_indices - 1
    x_mid = max_indices
    x_right = max_indices + 1

    # Keep left and right indices within bounds
    x_left = np.clip(x_left, 0, data.shape[-1] - 1)
    x_right = np.clip(x_right, 0, data.shape[-1] - 1)

    y_left = np.take_along_axis(data, x_left[..., None], axis=-1).squeeze(-1)
    y_mid = amplitudes
    y_right = np.take_along_axis(data, x_right[..., None], axis=-1).squeeze(-1)

    # Fit parabola for valid points only
    xv, yv = np.full(y_mid.shape, np.nan), np.full(y_mid.shape, np.nan)
    xv[valid_mask], yv[valid_mask] = calc_parabola_vertex_new(
        x_left[valid_mask], y_left[valid_mask],
        x_mid[valid_mask], y_mid[valid_mask],
        x_right[valid_mask], y_right[valid_mask]
    )

    xv[~valid_mask], yv[~valid_mask] = np.nan, np.nan

    success = valid_mask & (yv > 0)  # Optional condition for valid fits
    return np.stack([yv, xv, success], axis=-1)


def calc_parabola_vertex_new(x1, y1, x2, y2, x3, y3):
    # denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    # # if denom == 0:  # Handle edge cases early
    # #     return None, None
    #
    # A = ((y2 - y1) * (x3 - x1) - (y3 - y1) * (x2 - x1)) / denom
    # B = ((y2 - y1) * (x3**2 - x1**2) - (y3 - y1) * (x2**2 - x1**2)) / denom
    # xv = -B / (2 * A)
    # yv = y1 + A * (x1 - xv)**2  # Reuse A for parabola height

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)

    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom

    B = (x3 ** 2 * (y1 - y2) + x2 ** 2 * (y3 - y1) + x1 ** 2 * (y2 - y3)) / denom

    C1 = x2 * x3 * (x2 - x3) * y1
    C2 = x3 * x1 * (x3 - x1) * y2
    C3 = x1 * x2 * (x1 - x2) * y3
    C4 = C1 + C2 + C3

    C = C4 / denom

    xv = -B / (2 * A)
    yv = C - B ** 2 / (4 * A)

    return xv, yv


def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    # Convert all inputs to float to ensure floating-point division
    x1, y1, x2, y2, x3, y3 = map(float, (x1, y1, x2, y2, x3, y3))

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)

    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom

    B = (x3 ** 2 * (y1 - y2) + x2 ** 2 * (y3 - y1) + x1 ** 2 * (y2 - y3)) / denom

    C1 = x2 * x3 * (x2 - x3) * y1
    C2 = x3 * x1 * (x3 - x1) * y2
    C3 = x1 * x2 * (x1 - x2) * y3
    C4 = C1 + C2 + C3

    C = C4 / denom

    xv = -B / (2 * A)
    yv = C - B ** 2 / (4 * A)

    if yv <= 0:
        print(f"Warning: denom={denom} a={A} b={B} c={C} c1={C1} c2={C2} c3={C3} c4={C4}")

    return xv, yv

def get_waveform_max_sample(waveform):
    amplitude = np.max(waveform)
    time_max = np.argmax(waveform)
    return amplitude, time_max, 1


def gaussian_density(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def waveform_func(t, a, w, q):
    if np.isscalar(t):
        # Handle scalar input
        if t <= 0:
            return 0
        term1 = np.sqrt((2 * q - 1) / (2 * q + 1)) * np.sin(w * t / 2 * np.sqrt(4 - 1 / q ** 2))
        term2 = -np.cos(w * t / 2 * np.sqrt(4 - 1 / q ** 2))
        return a * (np.exp(-w * t) + np.exp(-w * t / (2 * q)) * (term1 + term2))
    else:
        # Handle numpy array input
        result = np.zeros_like(t)
        positive_t_mask = t > 0
        term1 = np.sqrt((2 * q - 1) / (2 * q + 1)) * np.sin(w * t[positive_t_mask] / 2 * np.sqrt(4 - 1 / q ** 2))
        term2 = -np.cos(w * t[positive_t_mask] / 2 * np.sqrt(4 - 1 / q ** 2))
        result[positive_t_mask] = a * (
                    np.exp(-w * t[positive_t_mask]) + np.exp(-w * t[positive_t_mask] / (2 * q)) * (term1 + term2))
        return result


def waveform_func_reparam(t, a, t_max, t_shift, q):
    t = t - t_shift
    t_max = t_max - t_shift
    w = 2 / t_max
    func = waveform_func(t, 1, w, q)
    func_max = waveform_func(t_max, 1, w, q)
    return a * func / func_max


def get_good_files(file_list, flags=None, feu_num=None, file_ext=None, file_nums=None):
    good_files = []
    file_nums = None if file_nums == 'all' else file_nums
    for file in file_list:
        if file_ext is not None and not file.endswith(file_ext):
            continue
        if file_nums is not None and get_num_from_fdf_file_name(file, -2) not in file_nums:
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
