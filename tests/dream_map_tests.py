#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 10 4:12 PM 2024
Created in PyCharm
Created as saclay_micromegas/dream_map_tests.py

@author: Dylan Neff, Dylan
"""

import numpy as np

from DetectorConfigLoader import DetectorConfigLoader, load_det_map


def main():
    maps_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    det_types = ['strip', 'inter', 'asacusa']
    for det_type in det_types:
        print(det_type)
        det_map = load_det_map(f'{maps_dir}{det_type}_map.txt')
        print(det_map)
        print(split_neighbors(det_map))
    print('donzo')


def split_neighbors(df):
    """
    Split the detector map into groups of connectors based on the axis, pitch, and interpitch.
    Return a dictionary of groups with connectors and their channels.
    :param df: Detector map dataframe.
    :return: Dictionary of groups with connectors and their channels.
    """
    df['group'] = ((df['axis'] != df['axis'].shift()) | (df['pitch(mm)'] != df['pitch(mm)'].shift()) |
                   (df['interpitch(mm)'] != df['interpitch(mm)'].shift()))  # Mark rows where group changes.
    df['group'] = df['group'].cumsum()  # Assign group number to each row using cumulative sum of group marks.
    grouped = df.groupby('group')

    group_channels = {}
    for group_id, group_data in grouped:  # Iterate through groups
        connectors = group_data.groupby('connector')['connectorChannel'].apply(
            lambda x: np.array(list(map(int, x)))).to_dict()  # Group by connector and get channels as array.
        group_channels[group_id] = connectors

    return group_channels  # Return dictionary of groups with connectors and their channels.


if __name__ == '__main__':
    main()
