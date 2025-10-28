#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 05 8:31 PM 2024
Created in PyCharm
Created as saclay_micromegas/DreamDetector.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from collections import Counter
import copy

from Detector_Classes.Detector import Detector
from Detector_Classes.DreamData import DreamData
from Detector_Classes.DreamSubDetector import DreamSubDetector


class DreamDetector(Detector):
    def __init__(self, name=None, center=None, size=None, rotations=None, config=None):
        super().__init__(name=name, center=center, size=size, rotations=rotations, config=config)
        self.hv = {}
        self.feu_num = None
        self.feu_connectors = []
        self.det_map = None

        self.config = config
        if self.config is not None:
            self.load_from_config_dream()

        self.dream_data = None
        self.x_groups, self.y_groups = None, None
        self.sub_detectors = None

        self.x_hits, self.y_hits = None, None
        self.x_clusters, self.y_clusters = None, None
        self.x_cluster_sizes, self.y_cluster_sizes = None, None
        self.x_cluster_triggers, self.y_cluster_triggers = None, None
        self.x_cluster_centroids, self.y_cluster_centroids = None, None
        self.x_largest_clusters, self.y_largest_clusters = None, None
        self.x_largest_cluster_sizes, self.y_largest_cluster_sizes = None, None
        self.x_largest_cluster_amp_sums, self.y_largest_cluster_amp_sums = None, None
        self.x_largest_cluster_centroids, self.y_largest_cluster_centroids = None, None
        self.xy_largest_cluster_sums = None
        self.x_det_sum, self.y_det_sum, self.xy_det_sum = None, None, None
        self.x_largest_amp, self.y_largest_amp = None, None

        self.local_sub_centroids = None
        self.sub_centroids = None
        self.sub_triggers = None
        self.sub_det_corners_local = None
        self.sub_det_corners_global = None
        self.det_corners_local = None
        self.det_corners_global = None

    def copy(self):
        return copy.deepcopy(self)

    def load_from_config_dream(self):
        dream_feus = self.config['dream_feus']
        feu_nums = list(set([chan[0] for chan in dream_feus.values()]))
        if len(feu_nums) != 1:
            print(f'Error: {self.name} has multiple FEUs: {feu_nums}')
        else:
            self.feu_num = feu_nums[0]
        for axis_name, slot_chan in dream_feus.items():
            self.feu_connectors.append(slot_chan[1])

        if 'det_map' in self.config:
            self.det_map = split_neighbors(self.config['det_map'], starting_connector=min(self.feu_connectors))
            # Center gerber coordinates such that (0, 0) is bottom left corner on maxes/mins of xs_gerber and ys_gerber
            x_min = min(xs.min() for xs in self.det_map['xs_gerber'])
            x_max = max(xs.max() for xs in self.det_map['xs_gerber'])
            self.det_map['xs_gerber'] = self.det_map['xs_gerber'].apply(lambda x: x - x_min)

            y_min = min(ys.min() for ys in self.det_map['ys_gerber'])
            y_max = max(ys.max() for ys in self.det_map['ys_gerber'])
            self.det_map['ys_gerber'] = self.det_map['ys_gerber'].apply(lambda y: y - y_min)

            self.active_size = np.array([x_max - x_min, y_max - y_min, self.active_size[2]])

        if 'hvs' in self.config:
            self.hv = self.config['hvs']

    def load_dream_data(self, data_dir, ped_dir=None, noise_threshold_sigmas=None, file_nums=None, chunk_size=100,
                        trigger_list=None, hist_raw_amps=False, save_waveforms=False, waveform_fit_func=None,
                        connector_channels=None):
        self.dream_data = DreamData(data_dir, self.feu_num, self.feu_connectors, ped_dir, waveform_fit_func)
        self.dream_data.connector_channels = connector_channels
        if noise_threshold_sigmas is not None:
            self.dream_data.noise_thresh_sigmas = noise_threshold_sigmas
        if ped_dir:
            self.dream_data.read_ped_data()
        self.dream_data.read_data(file_nums, chunk_size=chunk_size, trigger_list=trigger_list,
                                  hist_raw_amps=hist_raw_amps, save_waveforms=save_waveforms)

    def make_sub_groups(self):
        x_group_dfs = self.det_map[self.det_map['axis'] == 'y']  # y-going strips give x position
        self.x_groups = []
        for x_index, x_group in x_group_dfs.iterrows():
            x_connectors, x_channels = x_group['connectors'], x_group['channels']
            x_amps = self.dream_data.get_channels_amps(x_connectors, x_channels)
            x_hits = self.dream_data.get_channels_hits(x_connectors, x_channels)
            x_times = self.dream_data.get_channels_time_of_max(x_connectors, x_channels)
            x_clusters, x_cluster_indices = find_clusters_all_events(x_hits)
            x_cluster_sizes = get_cluster_sizes(x_clusters)
            x_cluster_triggers = self.dream_data.get_event_nums()[x_cluster_indices]
            x_cluster_timestamps = self.dream_data.get_timestamps()[x_cluster_indices]
            x_cluster_centroids = get_cluster_centroids_all_events(x_clusters, x_cluster_indices,
                                                                   x_group['xs_gerber'], x_amps)
            xlarge_clusts = get_largest_clusters_all_events(x_clusters, x_cluster_indices, x_cluster_centroids, x_amps)
            x_largest_clusters, x_largest_cluster_centroids = xlarge_clusts
            x_largest_cluster_sizes = get_cluster_sizes(x_largest_clusters)
            self.x_groups.append({'df': x_group, 'amps': x_amps, 'hits': x_hits, 'times': x_times,
                                  'clusters': x_clusters,
                                  'cluster_sizes': x_cluster_sizes,
                                  'cluster_triggers': np.array(x_cluster_triggers),
                                  'cluster_timestamps': np.array(x_cluster_timestamps),
                                  'cluster_centroids': x_cluster_centroids,
                                  'largest_clusters': x_largest_clusters,
                                  'largest_cluster_sizes': x_largest_cluster_sizes,
                                  'largest_cluster_centroids': np.array(x_largest_cluster_centroids)})

        self.x_hits = np.hstack([x_group['hits'] for x_group in self.x_groups])

        y_group_dfs = self.det_map[self.det_map['axis'] == 'x']  # x-going strips give y position
        self.y_groups = []
        for y_index, y_group in y_group_dfs.iterrows():
            y_connectors, y_channels = y_group['connectors'], y_group['channels']
            y_amps = self.dream_data.get_channels_amps(y_connectors, y_channels)
            y_hits = self.dream_data.get_channels_hits(y_connectors, y_channels)
            y_times = self.dream_data.get_channels_time_of_max(y_connectors, y_channels)
            y_clusters, y_cluster_indices = find_clusters_all_events(y_hits)
            y_cluster_sizes = get_cluster_sizes(y_clusters)
            y_cluster_triggers = self.dream_data.get_event_nums()[y_cluster_indices]
            y_cluster_timestamps = self.dream_data.get_timestamps()[y_cluster_indices]
            y_cluster_centroids = get_cluster_centroids_all_events(y_clusters, y_cluster_indices,
                                                                   y_group['ys_gerber'], y_amps)
            ylarge_clusts = get_largest_clusters_all_events(y_clusters, y_cluster_indices, y_cluster_centroids, y_amps)
            y_largest_clusters, y_largest_cluster_centroids = ylarge_clusts
            y_largest_cluster_sizes = get_cluster_sizes(y_largest_clusters)
            self.y_groups.append({'df': y_group, 'amps': y_amps, 'hits': y_hits, 'times': y_times,
                                  'clusters': y_clusters,
                                  'cluster_sizes': y_cluster_sizes,
                                  'cluster_triggers': np.array(y_cluster_triggers),
                                  'cluster_timestamps': np.array(y_cluster_timestamps),
                                  'cluster_centroids': y_cluster_centroids,
                                  'largest_clusters': y_largest_clusters,
                                  'largest_cluster_sizes': y_largest_cluster_sizes,
                                  'largest_cluster_centroids': np.array(y_largest_cluster_centroids)})

        self.y_hits = np.hstack([y_group['hits'] for y_group in self.y_groups])

        all_cluster_triggers = [x_group['cluster_triggers'] for x_group in self.x_groups] + \
                               [y_group['cluster_triggers'] for y_group in self.y_groups]
        all_cluster_triggers = np.unique(np.concatenate(all_cluster_triggers))
        # print(all_cluster_triggers)
        trigger_data = {}
        for x_group in self.x_groups:
            triggers = x_group['cluster_triggers']
            for trigger in triggers:
                if trigger not in trigger_data:
                    trigger_data[trigger] = {'x': {}}

    def make_sub_detectors(self):
        self.make_sub_groups()
        self.sub_detectors = []
        for x_group in self.x_groups:
            x_group_df = x_group['df'].to_dict()
            for y_group in self.y_groups:
                if 'asacusa' in self.config['det_type']:  # Hack to only group same connectors.
                    input('Changes to connector -> channel mapping probably broke this for asacusa, check. Enter to continue.')
                    if x_group_df['connectors'] != y_group['df']['connectors']:  # Find better way in map file.
                        continue
                y_group_df = y_group['df'].to_dict()
                x_connectors, y_connectors = x_group_df['connectors'], y_group_df['connectors']
                x_channels, y_channels = x_group_df['channels'], y_group_df['channels']
                x_pos, y_pos = x_group_df['xs_gerber'], y_group_df['ys_gerber']
                x_pitch, y_pitch = x_group_df['pitch(mm)'], y_group_df['pitch(mm)']
                x_interpitch, y_interpitch = x_group_df['interpitch(mm)'], y_group_df['interpitch(mm)']

                if 'inter' in self.config['det_type'].split('_'):  # Get correct x_interpitch
                    x_interpitches = x_interpitch.split(':')
                    all_y_gerbers = []
                    for y_group_i in self.y_groups:
                        y_group_i_df = y_group_i['df'].to_dict()
                        all_y_gerbers.extend(y_group_i_df['ys_gerber'])
                    avg_y_gerbers = np.mean(all_y_gerbers)

                    sub_group_y_avg = np.mean(y_group_df['ys_gerber'])
                    if sub_group_y_avg < avg_y_gerbers:
                        x_interpitch = x_interpitches[0]
                    else:
                        x_interpitch = x_interpitches[1]

                sub_det = DreamSubDetector(sub_index=len(self.sub_detectors))
                sub_det.set_x(x_pos, x_group['amps'], x_group['hits'], x_group['times'], x_pitch, x_interpitch,
                              x_connectors, x_group['cluster_triggers'], x_group['clusters'], x_group['cluster_sizes'],
                              x_group['cluster_centroids'], x_group['largest_clusters'],
                              x_group['largest_cluster_sizes'], x_group['largest_cluster_centroids'], x_channels)
                sub_det.set_y(y_pos, y_group['amps'], y_group['hits'], y_group['times'], y_pitch, y_interpitch,
                              y_connectors, y_group['cluster_triggers'], y_group['clusters'], y_group['cluster_sizes'],
                              y_group['cluster_centroids'], y_group['largest_clusters'],
                              y_group['largest_cluster_sizes'], y_group['largest_cluster_centroids'], y_channels)
                self.sub_detectors.append(sub_det)

    def get_sub_centroids_coords(self, recalculate=True):
        if recalculate:
            self.local_sub_centroids, self.sub_centroids, self.sub_triggers = [], [], []
            for sub_det in self.sub_detectors:
                triggers, centroids = sub_det.get_event_centroids()
                if triggers.shape[0] != centroids.shape[0]:
                    print(f'Error: Triggers and centroids have different shapes: {triggers.shape}, {centroids.shape}')
                if len(centroids) == 0:
                    continue
                zs = np.full((len(centroids), 1), 0)  # Add z coordinate to centroids
                centroids = np.hstack((centroids, zs))  # Combine x, y, z
                self.local_sub_centroids = centroids
                centroids = self.convert_coords_to_global(centroids)
                self.sub_centroids.append(centroids)
                self.sub_triggers.append(triggers)
        else:
            for sub_det_i in range(len(self.local_sub_centroids)):
                self.sub_centroids[sub_det_i] = self.convert_coords_to_global(self.local_sub_centroids)

        return self.sub_centroids, self.sub_triggers

    def get_sub_det_corners(self):
        self.sub_det_corners_local, self.sub_det_corners_global = [], []
        for sub_det in self.sub_detectors:
            x_min, x_max = np.min(sub_det.x_pos), np.max(sub_det.x_pos)
            y_min, y_max = np.min(sub_det.y_pos), np.max(sub_det.y_pos)
            corners = np.array([[x_min, y_min, 0], [x_min, y_max, 0], [x_max, y_min, 0], [x_max, y_max, 0]])
            self.sub_det_corners_local.append(corners)
            corners_global = self.convert_coords_to_global(corners)
            self.sub_det_corners_global.append(corners_global)

    def get_det_corners(self):
       x_min, x_max = self.center[0] - self.size[0] / 2, self.center[0] + self.size[0] / 2
       y_min, y_max = self.center[1] - self.size[1] / 2, self.center[1] + self.size[1] / 2
       corners = np.array([[x_min, y_min, 0], [x_min, y_max, 0], [x_max, y_min, 0], [x_max, y_max, 0]])
       self.det_corners_local = corners
       self.det_corners_global = self.convert_coords_to_global(corners)

    def get_det_clusters(self):
        """
        Get clusters for the whole detector, without splitting into sub-detectors.
        :return:
        """
        xs_gerber, x_amps = [], []
        for row_i, x_group in self.det_map[self.det_map['axis'] == 'y'].iterrows():
            x_connector, x_channels = x_group['connectors'], x_group['channels']
            x_amps.append(self.dream_data.get_channels_amps(x_connector, x_channels))
            xs_gerber.extend(x_group['xs_gerber'])
        xs_gerber = np.array(xs_gerber)
        x_amps = np.concatenate(x_amps, axis=1)

        self.x_clusters, x_cluster_indices = find_clusters_all_events(self.x_hits)
        self.x_cluster_sizes = get_cluster_sizes(self.x_clusters)
        self.x_cluster_centroids = get_cluster_centroids_all_events(self.x_clusters, x_cluster_indices, xs_gerber, x_amps)
        large_clust = get_largest_clusters_all_events(self.x_clusters, x_cluster_indices,self.x_cluster_centroids, x_amps)
        self.x_largest_clusters, self.x_largest_cluster_centroids = large_clust
        self.x_largest_cluster_sizes = get_cluster_sizes(self.x_largest_clusters)
        self.x_cluster_triggers = self.dream_data.event_nums[x_cluster_indices]
        self.x_largest_cluster_amp_sums = get_cluster_amp_sums(self.x_largest_clusters, x_amps)
        self.x_largest_cluster_amp_sums = [cluster_sum[0] for cluster_sum in self.x_largest_cluster_amp_sums]
        self.x_det_sum = get_det_amp_sums(x_amps)
        self.x_largest_amp = get_det_largest_amp(x_amps)

        ys_gerber, y_amps = [], []
        for row_i, y_group in self.det_map[self.det_map['axis'] == 'x'].iterrows():
            y_connector, y_channels = y_group['connectors'], y_group['channels']
            y_amps.append(self.dream_data.get_channels_amps(y_connector, y_channels))
            ys_gerber.extend(y_group['ys_gerber'])
        ys_gerber = np.array(ys_gerber)
        y_amps = np.concatenate(y_amps, axis=1)

        self.y_clusters, y_cluster_indices = find_clusters_all_events(self.y_hits)
        self.y_cluster_sizes = get_cluster_sizes(self.y_clusters)
        self.y_cluster_centroids = get_cluster_centroids_all_events(self.y_clusters, y_cluster_indices, ys_gerber, y_amps)
        large_clust = get_largest_clusters_all_events(self.y_clusters, y_cluster_indices, self.y_cluster_centroids, y_amps)
        self.y_largest_clusters, self.y_largest_cluster_centroids = large_clust
        self.y_largest_cluster_sizes = get_cluster_sizes(self.y_largest_clusters)
        self.y_cluster_triggers = self.dream_data.event_nums[y_cluster_indices]
        self.y_largest_cluster_amp_sums = get_cluster_amp_sums(self.y_largest_clusters, y_amps)
        self.y_largest_cluster_amp_sums = [cluster_sum[0] for cluster_sum in self.y_largest_cluster_amp_sums]
        self.y_det_sum = get_det_amp_sums(y_amps)
        self.y_largest_amp = get_det_largest_amp(y_amps)

        self.xy_det_sum = self.x_det_sum + self.y_det_sum

        # For each event, sum the amplitudes of the largest cluster in x and y
        # Convert x and y triggers to sets to find common events
        common_events = set(self.x_cluster_triggers) & set(self.y_cluster_triggers)

        # Create mappings from event number to sum
        x_sums = dict(zip(self.x_cluster_triggers, self.x_largest_cluster_amp_sums))
        y_sums = dict(zip(self.y_cluster_triggers, self.y_largest_cluster_amp_sums))

        # Sum the x and y values for common events
        result = {event: x_sums[event] + y_sums[event] for event in common_events}

        # Convert to lists if needed
        common_events_list = list(result.keys())
        self.xy_largest_cluster_sums = list(result.values())

    def set_sub_det_event_filters(self, event_nums):
        """
        Set event filters for each sub-detector based on a list of event numbers.
        :param event_nums:
        :return:
        """
        for sub_det in self.sub_detectors:
            sub_det.set_event_filter(event_nums)

    def in_sub_det(self, sub_det_i, x, y, z, tolerance=0):
        """
        Check if a point is within a sub-detector x-y area using the sub-detector corners. Add tolerance to bounds.
        Assume sub detector is rectangular in local coordinates, rotate x,y from global to local.
        :param sub_det_i: Index of sub-detector to check.
        :param x: X coordinate to check.
        :param y: Y coordinate to check.
        :param tolerance: Tolerance to add to the sub-detector bounds.
        :return:
        """
        self.get_sub_det_corners()
        coords = np.array([x, y, z])
        local_coords = self.convert_global_coords_to_local(coords)
        corners = self.sub_det_corners_local[sub_det_i]
        x_min, x_max = np.min(corners[:, 0]) - tolerance, np.max(corners[:, 0]) + tolerance
        y_min, y_max = np.min(corners[:, 1]) - tolerance, np.max(corners[:, 1]) + tolerance
        return x_min < local_coords[0] < x_max and y_min < local_coords[1] < y_max

    def in_sub_det_mask(self, sub_det_i, x, y, z, tolerance=0):
        """
        Return a boolean mask of which points are inside the sub-detector.
        x, y, z: arrays of same shape (N,)
        """
        self.get_sub_det_corners()
        coords = np.stack([x, y, z], axis=1)  # shape (N, 3)
        local_coords = self.convert_global_coords_to_local(coords)

        corners = self.sub_det_corners_local[sub_det_i]
        x_min, x_max = np.min(corners[:, 0]) - tolerance, np.max(corners[:, 0]) + tolerance
        y_min, y_max = np.min(corners[:, 1]) - tolerance, np.max(corners[:, 1]) + tolerance

        x_local = local_coords[:, 0]
        y_local = local_coords[:, 1]

        return (x_min < x_local) & (x_local < x_max) & (y_min < y_local) & (y_local < y_max)

    def in_det(self, x, y, z, tolerance=0.0):
        """
        Check if a point is within the detector x-y area using the detector corners. Add tolerance to bounds.
        :param x:
        :param y:
        :param z:
        :param tolerance:
        :return:
        """
        self.get_det_corners()
        coords = np.array([x, y, z])
        local_coords = self.convert_global_coords_to_local(coords)
        corners = self.det_corners_local
        x_min, x_max = np.min(corners[:, 0]) - tolerance, np.max(corners[:, 0]) + tolerance
        y_min, y_max = np.min(corners[:, 1]) - tolerance, np.max(corners[:, 1]) + tolerance
        return x_min < local_coords[0] < x_max and y_min < local_coords[1] < y_max

    def in_det_mask(self, x, y, z, tolerance=0.0):
        """
        Return a boolean mask of which points are inside the detector.
        x, y, z: arrays of same shape (N,)
        """
        self.get_det_corners()
        coords = np.stack([x, y, z], axis=1)  # shape (N, 3)
        local_coords = self.convert_global_coords_to_local(coords)

        corners = self.det_corners_local
        x_min, x_max = np.min(corners[:, 0]) - tolerance, np.max(corners[:, 0]) + tolerance
        y_min, y_max = np.min(corners[:, 1]) - tolerance, np.max(corners[:, 1]) + tolerance

        x_local = local_coords[:, 0]
        y_local = local_coords[:, 1]

        return (x_min < x_local) & (x_local < x_max) & (y_min < y_local) & (y_local < y_max)


    def plot_event_1d(self, event_id):
        """
        Plot amplitude vs position for each group in the detector. Plot x and y on separate plots.
        :param event_id:
        :return:
        """
        fig, axs = plt.subplots(2, 1, figsize=(8, 10))
        fig.suptitle(f'{self.name} event {event_id} amplitudes')
        axs[0].set_title('X Amplitudes')
        axs[0].set_xlabel('X (mm)')
        axs[0].set_ylabel('Amplitude')
        axs[1].set_title('Y Amplitudes')
        axs[1].set_xlabel('Y (mm)')
        axs[1].set_ylabel('Amplitude')

        for x_group in self.x_groups:
            x_group_df = x_group['df']
            x_pos = x_group_df['xs_gerber']
            x_amps = x_group['amps'][event_id]
            axs[0].plot(x_pos, x_amps,
                        label=f'Pitch: {x_group_df["pitch(mm)"]}, Interpitch: {x_group_df["interpitch(mm)"]}')
        axs[0].legend()

        for y_group in self.y_groups:
            y_group_df = y_group['df']
            y_pos = y_group_df['ys_gerber']
            y_amps = y_group['amps'][event_id]
            axs[1].plot(y_pos, y_amps,
                        label=f'Pitch: {y_group_df["pitch(mm)"]}, Interpitch: {y_group_df["interpitch(mm)"]}')
        axs[1].legend()
        fig.tight_layout()

    def plot_event_2d(self, event_id):
        """
        Plot a 2D heatmap of the amplitudes in an event. Iterate over the x and y groups and plot the amplitudes.
        :param event_id:
        :return:
        """
        fig, ax = plt.subplots()
        ax.set_title(f'{self.name} event {event_id} amplitudes')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_aspect('equal')

        # Collect all x and y positions and amplitudes
        x_positions, x_amplitudes = [], []
        y_positions, y_amplitudes = [], []

        # Assume that self.x_groups and self.y_groups are of the same length and have corresponding entries
        for x_group in self.x_groups:
            x_group_df = x_group['df']
            x_pos = x_group_df['xs_gerber']
            x_amps = x_group['amps'][event_id]
            x_positions.extend(x_pos)
            x_amplitudes.extend(x_amps)
        for y_group in self.y_groups:
            y_group_df = y_group['df']
            y_pos = y_group_df['ys_gerber']
            y_amps = y_group['amps'][event_id]
            y_positions.extend(y_pos)
            y_amplitudes.extend(y_amps)

        x_pos_plot, y_pos_plot, amp_sums_plot = [], [], []
        for x_pos, x_amp in zip(x_positions, x_amplitudes):
            for y_pos, y_amp in zip(y_positions, y_amplitudes):
                amp_sum = x_amp + y_amp
                if amp_sum > 0:
                    x_pos_plot.append(x_pos)
                    y_pos_plot.append(y_pos)
                    amp_sums_plot.append(x_amp + y_amp)

        scatter = ax.scatter(x_pos_plot, y_pos_plot, c=amp_sums_plot, cmap='jet', s=10)
        fig.colorbar(scatter, ax=ax, label='Amplitude')
        fig.tight_layout()

    def plot_hits_1d(self):
        """
        Plot all hits for each strip, split by subdetector and x-y axes
        :return:
        """
        fig, axs = plt.subplots(2, 1, figsize=(8, 10))
        fig.suptitle(f'{self.name} Hits')
        axs[0].set_title('X Hits')
        axs[0].set_xlabel('Strip')
        axs[0].set_ylabel('Hits')
        axs[1].set_title('Y Hits')
        axs[1].set_xlabel('Strip')
        axs[1].set_ylabel('Hits')

        for x_group in self.x_groups:
            x_group_df = x_group['df']
            x_hits = np.sum(x_group['hits'], axis=0)
            x_channels = self.dream_data.get_flat_channel_nums(x_group_df['connectors'], x_group_df['channels'])
            axs[0].plot(x_channels, x_hits,
                        label=f'Pitch: {x_group_df["pitch(mm)"]}, Interpitch: {x_group_df["interpitch(mm)"]}')
        axs[0].set_ylim(bottom=0)
        axs[0].legend()

        for y_group in self.y_groups:
            y_group_df = y_group['df']
            y_hits = np.sum(y_group['hits'], axis=0)
            y_channels = self.dream_data.get_flat_channel_nums(y_group_df['connectors'], y_group_df['channels'])
            axs[1].plot(y_channels, y_hits,
                        label=f'Pitch: {y_group_df["pitch(mm)"]}, Interpitch: {y_group_df["interpitch(mm)"]}')
        axs[1].set_ylim(bottom=0)
        axs[1].legend()
        fig.tight_layout()

    def plot_centroids_2d(self, alpha=0.1, bin_size=1):
        """
        Plot the centroids of the largest clusters in each sub-detector.
        Includes both a scatter plot and a 2D histogram (log scale, white zero bins).
        :param alpha: Transparency for scatter points.
        :param bin_size: Bin size in mm for the 2D histogram.
        """
        all_centroids = []
        subs_centroids, subs_triggers = self.get_sub_centroids_coords()
        for sub_centroids, sub_triggers in zip(subs_centroids, subs_triggers):
            all_centroids.extend(list(sub_centroids))
        x_centroids, y_centroids, z_centroids = zip(*all_centroids)

        # --- Scatter plot ---
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f'{self.name} Largest Cluster Centroids (Scatter)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_aspect('equal')
        # ax.scatter(x_centroids, y_centroids, alpha=alpha)
        ax.scatter(x_centroids, y_centroids, alpha=alpha, color='red', s=5, edgecolor='none')
        fig.tight_layout()

        # --- 2D histogram plot ---
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f'{self.name} Largest Cluster Centroids (2D Histogram)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_aspect('equal')

        # Determine bin edges based on bin_size
        x_min, x_max = min(x_centroids), max(x_centroids)
        y_min, y_max = min(y_centroids), max(y_centroids)

        x_bins = np.arange(x_min, x_max + bin_size, bin_size)
        y_bins = np.arange(y_min, y_max + bin_size, bin_size)

        # Prepare colormap with white for zero counts
        cmap = plt.cm.jet.copy()
        cmap.set_under('white')

        # 2D histogram with log scale
        h = ax.hist2d(
            x_centroids,
            y_centroids,
            bins=[x_bins, y_bins],
            cmap=cmap,
            norm=mcolors.LogNorm(vmin=1, vmax=None)
        )

        plt.colorbar(h[3], ax=ax, label='Counts (log scale)')

        # Scatter overlay for clarity
        # ax.scatter(x_centroids, y_centroids, alpha=alpha, color='white', s=5, edgecolor='none')

        fig.tight_layout()

    def plot_xy_hit_map(self, log_scale=True):
        """
        Create a 2D histogram of coincident x-y strip hits.
        X and Y axes correspond to strip numbers (0–127).
        """
        x_hits = self.x_hits  # shape (n_events, 128), boolean
        y_hits = self.y_hits  # shape (n_events, 128), boolean
        n_strips = x_hits.shape[1]

        # Initialize hit map
        hit_map = np.zeros((n_strips, n_strips), dtype=int)

        # Fill coincidence map efficiently
        for x_event, y_event in zip(x_hits, y_hits):
            x_indices = np.where(x_event)[0]
            y_indices = np.where(y_event)[0]
            hit_map[np.ix_(x_indices, y_indices)] += 1

        # --- Plot ---
        fig, ax = plt.subplots()
        ax.set_title(f'{self.name} X-Y Strip Hit Map')
        ax.set_xlabel('X strip number')
        ax.set_ylabel('Y strip number')

        cmap = plt.cm.jet.copy()
        cmap.set_under('white')

        if log_scale:
            norm = mcolors.LogNorm(vmin=1, vmax=None)
            label = 'Counts (log scale)'
        else:
            norm = None
            label = 'Counts'

        im = ax.imshow(
            hit_map.T,  # transpose so y is vertical
            origin='lower',
            cmap=cmap,
            norm=norm,
            interpolation='nearest',
            aspect='auto'
        )

        plt.colorbar(im, ax=ax, label=label)
        fig.tight_layout()


    def plot_amplitude_sum_vs_event_num(self, print_threshold=None):
        """
        Plot the sum of the amplitudes in each event from dream_data.
        :return:
        """
        fig, ax = plt.subplots()
        ax.set_title(f'{self.name} Event Amplitudes')
        ax.set_xlabel('Event Number')
        ax.set_ylabel('Amplitude Sum')

        event_sums = np.sum(self.dream_data.data_amps, axis=1)
        ax.plot(event_sums)
        fig.tight_layout()

        if print_threshold is not None:
            print(f'Events with sum greater than {print_threshold}:')
            for i, event_sum in enumerate(event_sums):
                if event_sum > print_threshold:
                    print(f'Event {i}: {event_sum}')

    def plot_xy_amp_sum_vs_event_num(self, return_events=False, threshold=None, print=False, hit_threshold=None):
        """
        Plot the sum of the amplitudes in each event from x and y groups.
        :param print_threshold: Threshold for printing events with sum greater than threshold.
        :param hit_threshold: Threshold for printing events with more than hit_threshold hits.
        :return:
        """
        fig, ax = plt.subplots()
        ax.set_title(f'{self.name} Event Amplitudes')
        ax.set_xlabel('Event Number')
        ax.set_ylabel('Amplitude Sum')

        for x_group in self.x_groups:
            x_amps = np.nansum(x_group['amps'], axis=1)
            event_nums = self.dream_data.event_nums
            # Sort x_amps and event_nums by event_nums
            x_amps = x_amps[event_nums.argsort()]
            event_nums = np.sort(event_nums)
            ax.plot(event_nums, x_amps, label=f'x: {x_group["df"]["pitch(mm)"]}, {x_group["df"]["interpitch(mm)"]}')
        for y_group in self.y_groups:
            y_amps = np.nansum(y_group['amps'], axis=1)
            event_nums = self.dream_data.event_nums
            # Sort y_amps and event_nums by event_nums
            y_amps = y_amps[event_nums.argsort()]
            ax.plot(event_nums,y_amps, label=f'y: {y_group["df"]["pitch(mm)"]}, {y_group["df"]["interpitch(mm)"]}')
        ax.legend()
        fig.tight_layout()

        if (return_events or print) and threshold is not None:
            if print:
                print(f'Events with sum greater than {threshold}:')
            x_amp_sums = np.nansum([np.sum(x_group['amps'], axis=1) for x_group in self.x_groups], axis=0)
            y_amp_sums = np.nansum([np.sum(y_group['amps'], axis=1) for y_group in self.y_groups], axis=0)
            event_nums = []
            for i, (x_amp_sum, y_amp_sum) in enumerate(zip(x_amp_sums, y_amp_sums)):
                if x_amp_sum > threshold and y_amp_sum > threshold:
                    if hit_threshold is not None:
                        if np.nansum(self.x_hits[i]) < hit_threshold and np.nansum(self.y_hits[i]) < hit_threshold:
                            if print:
                                print(f'Event {i}: x: {x_amp_sum}, y: {y_amp_sum}')
                            event_nums.append(i)
                    else:
                        print(f'Event {i}: x: {x_amp_sum}, y: {y_amp_sum}')
                        event_nums.append(i)
            if return_events:
                return event_nums

    def plot_xy_amp_sum_vs_timestamp(self, x_range=None, t_start=None, fix=False):
        """
        Plot the sum of the amplitudes in each event from x and y groups.
        :return:
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f'{self.name} Event Amplitudes')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Amplitude Sum')

        for x_group in self.x_groups:
            x_amps = np.sum(x_group['amps'], axis=1)
            timestamps = self.dream_data.timestamps
            event_nums = self.dream_data.event_nums

            # Sort x_amps, event_nums, and timestamps by event_nums
            x_amps = x_amps[event_nums.argsort()]
            timestamps = timestamps[event_nums.argsort()]

            if t_start is not None:  # t_start is a datetime, convert timestamp from seconds to datetimes
                t_start = np.datetime64(t_start)
                if fix:
                    timestamps = np.where(timestamps < 260000, timestamps, 0)
                timestamps = t_start + timestamps.astype('timedelta64[s]')

            ax.scatter(timestamps, x_amps, alpha=0.2, s=3, label=f'x: {x_group["df"]["pitch(mm)"]}, {x_group["df"]["interpitch(mm)"]}')


        for y_group in self.y_groups:
            y_amps = np.sum(y_group['amps'], axis=1)
            timestamps = self.dream_data.timestamps
            event_nums = self.dream_data.event_nums

            # Sort y_amps, event_nums, and timestamps by event_nums
            y_amps = y_amps[event_nums.argsort()]
            timestamps = timestamps[event_nums.argsort()]

            if t_start is not None:  # t_start is a datetime, convert timestamp from seconds to datetimes
                t_start = np.datetime64(t_start)
                if fix:
                    timestamps = np.where(timestamps < 260000, timestamps, 0)
                timestamps = t_start + timestamps.astype('timedelta64[s]')

            ax.scatter(timestamps, y_amps, alpha=0.2, s=3, label=f'y: {y_group["df"]["pitch(mm)"]}, {y_group["df"]["interpitch(mm)"]}')

        if t_start is not None:
            # Format x-axis as datetime
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))  # Customize format as needed
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.tick_params(axis='x', rotation=45)

        ax.legend()
        if x_range is not None:
            ax.set_xlim(x_range)
        fig.tight_layout()

    def plot_num_hit_xy_hist(self):
        """
        Plot a histogram of the number of hits per event in the x and y directions.
        :return:
        """
        fig, axs = plt.subplots(2, 1, figsize=(8, 10))
        fig.suptitle(f'{self.name} Number of Hits per Event')
        axs[0].set_title('X Hits')
        axs[0].set_xlabel('Number of Hits')
        axs[0].set_ylabel('Frequency')
        axs[1].set_title('Y Hits')
        axs[1].set_xlabel('Number of Hits')
        axs[1].set_ylabel('Frequency')

        x_hit_nums = np.sum([np.sum(x_group['hits'], axis=1) for x_group in self.x_groups], axis=0)
        y_hit_nums = np.sum([np.sum(y_group['hits'], axis=1) for y_group in self.y_groups], axis=0)

        axs[0].hist(x_hit_nums, bins=range(0, 125, 1))
        axs[1].hist(y_hit_nums, bins=range(0, 125, 1))
        fig.tight_layout()

    def plot_cluster_sizes(self):
        """
        Plot the distribution of the sizes of the clusters in the detector.
        :return:
        """
        fig, ax = plt.subplots()
        fig.suptitle(f'{self.name} Cluster Sizes')
        ax.set_xlabel('Cluster Size')
        ax.set_ylabel('Counts')

        max_x, max_y = np.max(self.x_largest_cluster_sizes), np.max(self.y_largest_cluster_sizes)
        ax.hist(self.x_largest_cluster_sizes, bins=range(0, max_x + 1, 1), alpha=0.5, label='X Clusters')
        ax.hist(self.y_largest_cluster_sizes, bins=range(0, max_y + 1, 1), alpha=0.5, label='Y Clusters')

        ax.legend()
        fig.tight_layout()

    def plot_cluster_amps(self):
        """
        Plot the distribution of the amplitudes in the largest clusters in the detector.
        :return:
        """
        fig, ax = plt.subplots()
        fig.suptitle(f'{self.name} Largest Cluster Amplitudes')
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Counts')

        x_largest_cluster_amp_sums = np.array(self.x_largest_cluster_amp_sums)
        y_largest_cluster_amp_sums = np.array(self.y_largest_cluster_amp_sums)

        ax.hist(x_largest_cluster_amp_sums, bins=500, alpha=0.5, label='X Clusters')
        ax.hist(y_largest_cluster_amp_sums, bins=500, alpha=0.5, label='Y Clusters')

        ax.set_yscale('log')
        ax.legend()
        fig.tight_layout()

        # Add x and y cluster sums together
        fig, ax = plt.subplots()
        fig.suptitle(f'{self.name} Largest Cluster Amplitudes')
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Counts')
        ax.hist(self.xy_largest_cluster_sums, bins=500, alpha=0.5, label='X+Y Clusters')
        ax.set_yscale('log')
        ax.legend()
        fig.tight_layout()

    def plot_det_amp_sums(self, fit=False):
        """
        Plot amplitude sums for the whole detector for x, y and x+y
        Returns:
        """
        # Plot for x and y separately
        fig, ax = plt.subplots()
        fig.suptitle(f'{self.name} Detector Amplitude Sums (X and Y)')
        ax.set_xlabel('Amplitude Sum')
        ax.set_ylabel('Counts')

        x_det_sum = np.array(self.x_det_sum)
        y_det_sum = np.array(self.y_det_sum)

        ax.hist(x_det_sum, bins=500, alpha=0.5, label='X Detector')
        ax.hist(y_det_sum, bins=500, alpha=0.5, label='Y Detector')

        ax.legend()
        fig.tight_layout()

        # Plot for x+y sum
        fig, ax = plt.subplots()
        fig.suptitle(f'{self.name} Detector Amplitude Sums (X+Y)')
        ax.set_xlabel('Amplitude Sum')
        ax.set_ylabel('Counts')

        xy_det_sum = np.array(self.xy_det_sum)
        ax.hist(xy_det_sum, bins=500, alpha=0.5, label='X+Y Detector')

        ax.legend()
        fig.tight_layout()


    def plot_det_largest_amp_vs_amp_sums(self, bins=50, norm_per_strip=False, x_min=None, x_max=None, y_min=None, y_max=None):
        """
        Plot a 2D histogram of the largest amplitude strip per event vs the sum of all amplitudes in the event.
        For x and y separately.
        Returns:

        """
        fig, axs = plt.subplots(2, 1, figsize=(8, 10))
        fig.suptitle(f'{self.name} Largest Amplitude vs Amplitude Sum')
        axs[0].set_title('X Detector')
        axs[0].set_xlabel('Amplitude Sum')
        axs[0].set_ylabel('Largest Amplitude')
        axs[1].set_title('Y Detector')
        axs[1].set_xlabel('Amplitude Sum')
        axs[1].set_ylabel('Largest Amplitude')

        x_largest_amp = np.array(self.x_largest_amp)
        x_det_sum = np.array(self.x_det_sum)
        y_largest_amp = np.array(self.y_largest_amp)
        y_det_sum = np.array(self.y_det_sum)

        if norm_per_strip:
            n_x_strips = self.x_hits.shape[1]
            x_det_sum = self.x_det_sum / n_x_strips
            n_y_strips = self.y_hits.shape[1]
            y_det_sum = self.y_det_sum / n_y_strips
            axs[0].set_xlabel('Average Amplitude per Strip')
            axs[1].set_xlabel('Average Amplitude per Strip')

        axs[0].hist2d(x_det_sum, x_largest_amp, bins=bins, cmin=1, cmap='jet')
        axs[1].hist2d(y_det_sum, y_largest_amp, bins=bins, cmin=1, cmap='jet')
        fig.colorbar(axs[0].collections[0], ax=axs[0], label='Counts')
        fig.colorbar(axs[1].collections[0], ax=axs[1], label='Counts')
        axs[0].set_xlim(x_min if x_min is not None else np.min(x_det_sum),
                        x_max if x_max is not None else np.max(x_det_sum))
        axs[0].set_ylim(y_min if y_min is not None else np.min(x_largest_amp),
                        y_max if y_max is not None else np.max(x_largest_amp))

        axs[1].set_xlim(x_min if x_min is not None else np.min(y_det_sum),
                        x_max if x_max is not None else np.max(y_det_sum))
        axs[1].set_ylim(y_min if y_min is not None else np.min(y_largest_amp),
                        y_max if y_max is not None else np.max(y_largest_amp))

        fig.tight_layout()


def split_neighbors(df, starting_connector=0):
    """
    Split the detector map into groups of connectors based on the axis, pitch, and interpitch.
    Return a dataframe where each entry is a group with columns: axis, pitch, interpitch, connector, channels.
    :param df: Detector map dataframe.
    :param starting_connector: Connector number to start with.
    :return: DataFrame of groups with columns: axis, pitch, interpitch, connector, channels.
    """
    # Mark rows where group changes
    # df['group'] = ((df['axis'] != df['axis'].shift()) | (df['connector'] != df['connector'].shift()) |
    #                (df['pitch(mm)'] != df['pitch(mm)'].shift()) |
    #                (df['interpitch(mm)'] != df['interpitch(mm)'].shift()))

    # Remove connector from group change condition for RD542 homogeneous detectors, tested and looks ok for old detectors too
    df['group'] = ((df['axis'] != df['axis'].shift()) |
                   (df['pitch(mm)'] != df['pitch(mm)'].shift()) |
                   (df['interpitch(mm)'] != df['interpitch(mm)'].shift()))

    # Assign group number to each row using cumulative sum of group marks
    df['group'] = df['group'].cumsum()

    # Create a unique name for each group
    # df['group_name'] = df.apply(lambda row:
    #                             f"{row['axis']}_{row['connector']}_{row['pitch(mm)']}_{row['interpitch(mm)']}", axis=1)
    # This might break inter detector, need to test
    df['group_name'] = df.apply(lambda row:
                                f"{row['axis']}_{row['pitch(mm)']}_{row['interpitch(mm)']}", axis=1)

    # Group by the new group_name column
    grouped = df.groupby('group_name')
    print(f'Found {len(grouped)} groups in detector map')

    # Prepare the output dataframe
    result_data = []

    for group_name, group_data in grouped:  # Iterate through groups
        axis = group_data['axis'].iloc[0]
        pitch = group_data['pitch(mm)'].iloc[0]
        interpitch = group_data['interpitch(mm)'].iloc[0]
        # connector = int(group_data['connector'].iloc[0]) + starting_connector
        connector = np.array(list(map(int, group_data['connector']))) + starting_connector
        channels = np.array(list(map(int, group_data['connectorChannel'])))
        x_gerber = np.array(list(map(float, group_data['xGerber'])))
        y_gerber = np.array(list(map(float, group_data['yGerber'])))

        result_data.append({
            'axis': axis,
            'pitch(mm)': pitch,
            'interpitch(mm)': interpitch,
            'connectors': connector,
            'channels': channels,
            'xs_gerber': x_gerber,
            'ys_gerber': y_gerber
        })

    columns = ['axis', 'pitch(mm)', 'interpitch(mm)', 'connectors', 'channels', 'xs_gerber', 'ys_gerber']
    result_df = pd.DataFrame(result_data, columns=columns)
    return result_df


def find_clusters_all_events(hit_data):
    """
    Find the clusters for all events in a hit_data array.
    :param hit_data: Array of hit data with shape (n_events, n_channels).
    :return: List of clusters for each event.
    """
    clusters, cluster_triggers = [], []
    for trigger_id, event in enumerate(hit_data):
        event_clusters = find_true_clusters(event)
        if len(event_clusters) > 0:
            clusters.append(event_clusters)
            cluster_triggers.append(trigger_id)
    return clusters, cluster_triggers


def find_true_clusters(bool_array):
    # Ensure the input is a numpy array
    bool_array = np.asarray(bool_array)

    # Find the indices where the value is True
    true_indices = np.where(bool_array)[0]

    if len(true_indices) == 0:
        return []

    # Identify the breaks in the sequence of true_indices
    breaks = np.where(np.diff(true_indices) > 1)[0]

    # Initialize the list of clusters
    clusters = []

    # Append the first segment of indices
    start = 0
    for end in breaks:
        clusters.append(true_indices[start:end + 1])
        start = end + 1

    # Append the last segment of indices
    clusters.append(true_indices[start:])

    return clusters


def get_cluster_sizes(clusters):
    """
    Get the sizes of the clusters in all events.
    :param clusters: List of clusters.
    :return: Sizes of the clusters.
    """
    if len(clusters) == 0:
        return []
    if isinstance(clusters[0], list):  # Multiple clusters in each event, keep structure
        return [[len(cluster) for cluster in event_clusters] for event_clusters in clusters]
    else:  # Single cluster in each event
        return [len(cluster) for cluster in clusters]


def get_cluster_amp_sums(clusters, amp_array):
    """
    Get the sum of the amplitudes in each cluster in all events.
    :param clusters: List of clusters.
    :param amp_array: Array of amplitudes for each channel in the event.
    :return: Sum of the amplitudes in each cluster.
    """
    cluster_amp_sums = []
    for event_clusters, event_amps in zip(clusters, amp_array):
        event_cluster_amp_sums = []
        if isinstance(event_clusters, list):  # List of clusters
            for cluster in event_clusters:
                event_cluster_amp_sums.append(np.sum(event_amps[cluster]))
        else:  # Single cluster from largest clusters
            event_cluster_amp_sums.append(np.sum(event_amps[event_clusters]))
        cluster_amp_sums.append(event_cluster_amp_sums)
    return cluster_amp_sums


def get_det_amp_sums(amp_array):
    """
    Get sum of all channels in each event.
    Args:
        amp_array:

    Returns:

    """
    return np.sum(amp_array, axis=1)


def get_det_largest_amp(amp_array):
    """
    Get largest amplitude strip in each event.
    Args:
        amp_array:

    Returns:

    """
    return np.max(amp_array, axis=1)


def get_cluster_centroids_all_events(clusters, cluster_indices, pos_array, amp_array):
    """
    Get the centroids of the clusters in pos_array, weighted by amp_array.
    :param clusters: Array of cluster indices in the event.
    :param cluster_indices: Array of trigger ids for each event.
    :param pos_array: Array of positions for each channel.
    :param amp_array: Array of amplitudes for each channel in the event.
    :return:
    """
    centroids = []
    for event_clusters, cluster_index in zip(clusters, cluster_indices):
        centroids.append(get_cluster_centroids(event_clusters, pos_array, amp_array[cluster_index]))
    return centroids


def get_cluster_centroids(clusters, pos_array, amp_array):
    """
    Get the centroids of the clusters in pos_array, weighted by amp_array.
    :param clusters: Array of cluster indices in the event.
    :param pos_array: Array of positions for each channel.
    :param amp_array: Array of amplitudes for each channel in the event.
    :return:
    """
    centroids = []
    for cluster in clusters:
        cluster_pos = pos_array[cluster]
        cluster_amp = amp_array[cluster]
        centroid = np.average(cluster_pos, weights=cluster_amp)
        centroids.append(centroid)
    return centroids


def get_largest_clusters_all_events(clusters, cluster_indices, cluster_centroids, amp_array):
    """
    Get the largest cluster for each event.
    :param clusters: List of clusters.
    :param cluster_indices: List of trigger ids for each event.
    :param cluster_centroids: List of cluster centroids for each event.
    :param amp_array: Array of amplitudes for each channel in the event.
    :return: Largest clusters for each event.
    """
    largest_clusters, largest_cluster_centroids = [], []
    for event_clusters, cluster_index, event_centroids in zip(clusters, cluster_indices, cluster_centroids):
        if len(event_clusters) == 0:
            print(f'No clusters in event number {cluster_index}')
        largest_cluster, largest_cluster_centroid = get_largest_cluster(event_clusters, event_centroids,
                                                                        amp_array[cluster_index])
        largest_clusters.append(largest_cluster)
        largest_cluster_centroids.append(largest_cluster_centroid)
    return largest_clusters, largest_cluster_centroids


def get_largest_cluster(clusters, cluster_centroids, amp_array):
    """
    Get the largest cluster in the event based on the sum of the amplitude.
    :param clusters: List of clusters.
    :param cluster_centroids: List of cluster centroids.
    :param amp_array: Array of amplitudes for each channel in the event.
    :return: Largest cluster in the event along with its centroid.
    """
    largest_cluster = []
    largest_cluster_centroid = 0
    largest_cluster_amp = 0
    for cluster, centroid in zip(clusters, cluster_centroids):
        cluster_amp = np.sum(amp_array[cluster])
        if cluster_amp > largest_cluster_amp:
            largest_cluster = cluster
            largest_cluster_amp = cluster_amp
            largest_cluster_centroid = centroid

    return largest_cluster, largest_cluster_centroid
