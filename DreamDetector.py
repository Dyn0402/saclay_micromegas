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

from Detector import Detector
from DreamData import DreamData
from DreamSubDetector import DreamSubDetector


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
        self.x_cluster_centroids, self.y_cluster_centroids = None, None
        self.x_largest_clusters, self.y_largest_clusters = None, None
        self.x_largest_cluster_centroids, self.y_largest_cluster_centroids = None, None

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

        self.hv = self.config['hvs']

    def load_dream_data(self, data_dir, ped_dir=None):
        self.dream_data = DreamData(data_dir, self.feu_num, self.feu_connectors, ped_dir)
        self.dream_data.read_ped_data()
        self.dream_data.read_data()

    def make_sub_groups(self):
        x_group_dfs = self.det_map[self.det_map['axis'] == 'y']  # y-going strips give x position
        self.x_largest_cluster_centroids = []
        self.x_groups = []
        for x_index, x_group in x_group_dfs.iterrows():
            x_connector, x_channels = x_group['connector'], x_group['channels']
            x_amps = self.dream_data.get_channels_amps(x_connector, x_channels)
            x_hits = self.dream_data.get_channels_hits(x_connector, x_channels)
            x_clusters, x_cluster_indices = find_clusters_all_events(x_hits)
            x_cluster_triggers = self.dream_data.event_nums[x_cluster_indices]
            x_cluster_centroids = get_cluster_centroids_all_events(x_clusters, x_cluster_indices,
                                                                   x_group['xs_gerber'], x_amps)
            xlarge_clusts = get_largest_clusters_all_events(x_clusters, x_cluster_indices, x_cluster_centroids, x_amps)
            x_largest_clusters, x_largest_cluster_centroids = xlarge_clusts
            self.x_groups.append({'df': x_group, 'amps': x_amps, 'hits': x_hits, 'clusters': x_clusters,
                                  'cluster_triggers': np.array(x_cluster_triggers),
                                  'cluster_centroids': x_cluster_centroids,
                                  'largest_clusters': x_largest_clusters,
                                  'largest_cluster_centroids': np.array(x_largest_cluster_centroids)})
            self.x_largest_cluster_centroids.extend(x_largest_cluster_centroids)

        for x_group in self.x_groups:
            print(x_group['hits'].shape)
        self.x_hits = np.hstack([x_group['hits'] for x_group in self.x_groups])
        print(self.x_hits.shape)
        print('Hits event 1')
        print(self.x_hits[1])
        self.x_largest_cluster_centroids = np.array(self.x_largest_cluster_centroids)

        y_group_dfs = self.det_map[self.det_map['axis'] == 'x']  # x-going strips give y position
        self.y_largest_cluster_centroids = []
        self.y_groups = []
        for y_index, y_group in y_group_dfs.iterrows():
            y_connector, y_channels = y_group['connector'], y_group['channels']
            y_amps = self.dream_data.get_channels_amps(y_connector, y_channels)
            y_hits = self.dream_data.get_channels_hits(y_connector, y_channels)
            y_clusters, y_cluster_indices = find_clusters_all_events(y_hits)
            y_cluster_triggers = self.dream_data.event_nums[y_cluster_indices]
            y_cluster_centroids = get_cluster_centroids_all_events(y_clusters, y_cluster_indices,
                                                                   y_group['ys_gerber'], y_amps)
            ylarge_clusts = get_largest_clusters_all_events(y_clusters, y_cluster_indices, y_cluster_centroids, y_amps)
            y_largest_clusters, y_largest_cluster_centroids = ylarge_clusts
            self.y_groups.append({'df': y_group, 'amps': y_amps, 'hits': y_hits, 'clusters': y_clusters,
                                  'cluster_triggers': np.array(y_cluster_triggers),
                                  'cluster_centroids': y_cluster_centroids,
                                  'largest_clusters': y_largest_clusters,
                                  'largest_cluster_centroids': np.array(y_largest_cluster_centroids)})
            self.y_largest_cluster_centroids = np.array(self.y_largest_cluster_centroids)

        self.y_hits = np.hstack([y_group['hits'] for y_group in self.y_groups])

        all_cluster_triggers = [x_group['cluster_triggers'] for x_group in self.x_groups] + \
                               [y_group['cluster_triggers'] for y_group in self.y_groups]
        print(all_cluster_triggers)
        print(np.concatenate(all_cluster_triggers))
        print(np.concatenate(all_cluster_triggers).shape)
        all_cluster_triggers = np.unique(np.concatenate(all_cluster_triggers))
        print(all_cluster_triggers)
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
                    if x_group_df['connector'] != y_group['df']['connector']:  # Find better way in map file.
                        continue
                y_group_df = y_group['df'].to_dict()
                x_connector, y_connector = x_group_df['connector'], y_group_df['connector']
                x_channels, y_channels = x_group_df['channels'], y_group_df['channels']
                x_pos, y_pos = x_group_df['xs_gerber'], y_group_df['ys_gerber']
                x_pitch, y_pitch = x_group_df['pitch(mm)'], y_group_df['pitch(mm)']
                x_interpitch, y_interpitch = x_group_df['interpitch(mm)'], y_group_df['interpitch(mm)']

                sub_det = DreamSubDetector()
                sub_det.set_x(x_pos, x_group['amps'], x_group['hits'], x_pitch, x_interpitch, x_connector,
                              x_group['cluster_triggers'], x_group['clusters'], x_group['cluster_centroids'],
                              x_group['largest_clusters'], x_group['largest_cluster_centroids'], x_channels)
                sub_det.set_y(y_pos, y_group['amps'], y_group['hits'], y_pitch, y_interpitch, y_connector,
                              y_group['cluster_triggers'], y_group['clusters'], y_group['cluster_centroids'],
                              y_group['largest_clusters'], y_group['largest_cluster_centroids'], y_channels)
                self.sub_detectors.append(sub_det)

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

    def plot_centroids_2d(self):
        """
        Plot the centroids of the largest clusters in each subdetector.
        :return:
        """
        fig, ax = plt.subplots()
        ax.set_title(f'{self.name} Largest Cluster Centroids')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_aspect('equal')
        ax.scatter(self.x_largest_cluster_centroids, self.y_largest_cluster_centroids, alpha=0.5)
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

    def plot_xy_amp_sum_vs_event_num(self, print_threshold=None, hit_threshold=None):
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
            x_amps = np.sum(x_group['amps'], axis=1)
            ax.plot(x_amps, label=f'x: {x_group["df"]["pitch(mm)"]}, {x_group["df"]["interpitch(mm)"]}')
        for y_group in self.y_groups:
            y_amps = np.sum(y_group['amps'], axis=1)
            ax.plot(y_amps, label=f'y: {y_group["df"]["pitch(mm)"]}, {y_group["df"]["interpitch(mm)"]}')
        ax.legend()
        fig.tight_layout()

        if print_threshold is not None:
            print(f'Events with sum greater than {print_threshold}:')
            x_amp_sums = np.sum([np.sum(x_group['amps'], axis=1) for x_group in self.x_groups], axis=0)
            y_amp_sums = np.sum([np.sum(y_group['amps'], axis=1) for y_group in self.y_groups], axis=0)
            event_nums = []
            for i, (x_amp_sum, y_amp_sum) in enumerate(zip(x_amp_sums, y_amp_sums)):
                if x_amp_sum > print_threshold and y_amp_sum > print_threshold:
                    if hit_threshold is not None:
                        if np.sum(self.x_hits[i]) < hit_threshold and np.sum(self.y_hits[i]) < hit_threshold:
                            print(f'Event {i}: x: {x_amp_sum}, y: {y_amp_sum}')
                            event_nums.append(i)
                    else:
                        print(f'Event {i}: x: {x_amp_sum}, y: {y_amp_sum}')
                        event_nums.append(i)
            return event_nums

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


def split_neighbors(df, starting_connector=0):
    """
    Split the detector map into groups of connectors based on the axis, pitch, and interpitch.
    Return a dataframe where each entry is a group with columns: axis, pitch, interpitch, connector, channels.
    :param df: Detector map dataframe.
    :param starting_connector: Connector number to start with.
    :return: DataFrame of groups with columns: axis, pitch, interpitch, connector, channels.
    """
    # Mark rows where group changes
    df['group'] = ((df['axis'] != df['axis'].shift()) | (df['connector'] != df['connector'].shift()) |
                   (df['pitch(mm)'] != df['pitch(mm)'].shift()) |
                   (df['interpitch(mm)'] != df['interpitch(mm)'].shift()))

    # Assign group number to each row using cumulative sum of group marks
    df['group'] = df['group'].cumsum()

    # Create a unique name for each group
    df['group_name'] = df.apply(lambda row:
                                f"{row['axis']}_{row['connector']}_{row['pitch(mm)']}_{row['interpitch(mm)']}", axis=1)

    # Group by the new group_name column
    grouped = df.groupby('group_name')

    # Prepare the output dataframe
    result_data = []

    for group_name, group_data in grouped:  # Iterate through groups
        axis = group_data['axis'].iloc[0]
        pitch = group_data['pitch(mm)'].iloc[0]
        interpitch = group_data['interpitch(mm)'].iloc[0]
        connector = int(group_data['connector'].iloc[0]) + starting_connector
        channels = np.array(list(map(int, group_data['connectorChannel'])))
        x_gerber = np.array(list(map(float, group_data['xGerber'])))
        y_gerber = np.array(list(map(float, group_data['yGerber'])))

        result_data.append({
            'axis': axis,
            'pitch(mm)': pitch,
            'interpitch(mm)': interpitch,
            'connector': connector,
            'channels': channels,
            'xs_gerber': x_gerber,
            'ys_gerber': y_gerber
        })

    columns = ['axis', 'pitch(mm)', 'interpitch(mm)', 'connector', 'channels', 'xs_gerber', 'ys_gerber']
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
