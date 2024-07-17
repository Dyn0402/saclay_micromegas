#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 05 8:31 PM 2024
Created in PyCharm
Created as saclay_micromegas/DreamDetector.py

@author: Dylan Neff, Dylan
"""

import json

import numpy as np
import pandas as pd

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
        self.x_groups = None
        self.y_groups = None
        self.sub_detectors = None

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
            self.det_map = split_neighbors(self.config['det_map'])

        self.hv = self.config['hvs']

    def load_dream_data(self, data_dir, ped_dir=None):
        self.dream_data = DreamData(data_dir, self.feu_num, self.feu_connectors, ped_dir)
        self.dream_data.read_ped_data()
        self.dream_data.read_data()

    def make_sub_groups(self):
        x_group_dfs = self.det_map[self.det_map['axis'] == 'y']  # y-going strips give x position
        self.x_groups = []
        for x_index, x_group in x_group_dfs.iterrows():
            x_connector, x_channels = x_group['connector'], x_group['channels']
            x_amps = self.dream_data.get_channels_amps(x_connector, x_channels)
            x_hits = self.dream_data.get_channels_hits(x_connector, x_channels)
            x_clusters = find_clusters_all_events(x_hits)
            self.x_groups.append({'df': x_group, 'amps': x_amps, 'hits': x_hits, 'clusters': x_clusters})

        y_group_dfs = self.det_map[self.det_map['axis'] == 'x']  # x-going strips give y position
        self.y_groups = []
        for y_index, y_group in y_group_dfs.iterrows():
            y_connector, y_channels = y_group['connector'], y_group['channels']
            y_amps = self.dream_data.get_channels_amps(y_connector, y_channels)
            y_hits = self.dream_data.get_channels_hits(y_connector, y_channels)
            y_clusters = find_clusters_all_events(y_hits)
            self.y_groups.append({'df': y_group, 'amps': y_amps, 'hits': y_hits, 'clusters': y_clusters})

    def make_sub_detectors(self):
        self.make_sub_groups()
        self.sub_detectors = []
        for x_group in self.x_groups:
            x_group_df = x_group['df'].to_dict()
            for y_index, y_group in self.y_groups:
                y_group_df = y_group['df'].to_dict()
                x_connector, y_connector = x_group_df['connector'], y_group_df['connector']
                x_channels, y_channels = x_group_df['channels'], y_group_df['channels']
                x_pos, y_pos = x_group_df['xs_gerber'], y_group_df['ys_gerber']
                x_pitch, y_pitch = x_group_df['pitch(mm)'], y_group_df['pitch(mm)']
                x_interpitch, y_interpitch = x_group_df['interpitch(mm)'], y_group_df['interpitch(mm)']

                sub_det = DreamSubDetector()
                sub_det.set_x(x_pos, x_group['amps'], x_group['hits'], x_pitch, x_interpitch, x_connector,
                              x_group['clusters'], x_channels)
                sub_det.set_y(y_pos, y_group['amps'], y_group['hits'], y_pitch, y_interpitch, y_connector,
                              y_group['clusters'], y_channels)
                self.sub_detectors.append(sub_det)


def split_neighbors(df):
    """
    Split the detector map into groups of connectors based on the axis, pitch, and interpitch.
    Return a dataframe where each entry is a group with columns: axis, pitch, interpitch, connector, channels.
    :param df: Detector map dataframe.
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
        pitch = float(group_data['pitch(mm)'].iloc[0])
        interpitch = float(group_data['interpitch(mm)'].iloc[0])
        connector = int(group_data['connector'].iloc[0])
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
    clusters = []
    for event in hit_data:
        clusters.append(find_true_clusters(event))
    return clusters


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
