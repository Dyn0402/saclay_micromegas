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

    def make_sub_detectors(self):
        x_groups_df = self.det_map[self.det_map['axis'] == 'y']  # y-going strips give x position
        y_groups_df = self.det_map[self.det_map['axis'] == 'x']  # x-going strips give y position
        self.sub_detectors = []
        for x_index, x_group in x_groups_df.iterrows():
            x_group = x_group.to_dict()
            for y_index, y_group in y_groups_df.iterrows():
                y_group = y_group.to_dict()
                x_connector, y_connector = x_group['connector'], y_group['connector']
                x_channels, y_channels = x_group['channels'], y_group['channels']
                x_pos, y_pos = x_group['xs_gerber'], y_group['ys_gerber']
                x_pitch, y_pitch = x_group['pitch(mm)'], y_group['pitch(mm)']
                x_interpitch, y_interpitch = x_group['interpitch(mm)'], y_group['interpitch(mm)']
                x_amps = self.dream_data.get_channels_amps(x_connector, x_channels)
                y_amps = self.dream_data.get_channels_amps(y_connector, y_channels)
                x_hits = self.dream_data.get_channels_hits(x_connector, x_channels)
                y_hits = self.dream_data.get_channels_hits(y_connector, y_channels)

                sub_det = DreamSubDetector()
                sub_det.set_x(x_pos, x_amps, x_hits, x_pitch, x_interpitch, x_connector)
                sub_det.set_y(y_pos, y_amps, y_hits, y_pitch, y_interpitch, y_connector)
                self.sub_detectors.append(sub_det)
                sub_det.get_clusters()


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
