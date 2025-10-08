#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 10 4:12 PM 2024
Created in PyCharm
Created as saclay_micromegas/dream_map_tests.py

@author: Dylan Neff, Dylan
"""

from Detector_Classes.DetectorConfigLoader import load_det_map
from Detector_Classes.DreamDetector import split_neighbors
from Detector_Classes.DreamData import DreamData


def main():
    maps_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    # maps_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    det_types = ['strip', 'inter', 'asacusa', 'rd542']
    for det_type in det_types:
        print(det_type)
        det_map = load_det_map(f'{maps_dir}{det_type}_map.txt')
        # print(det_map)
        split_df = split_neighbors(det_map)
        print(split_df)
        print(split_df.iloc[0]['channels'])
        print(split_df.iloc[0]['connectors'])
        connectors = split_df.iloc[0]['connectors']
        connector_channels = split_df.iloc[0]['channels']
        channel_per_connector = 64
        channels = connectors * channel_per_connector + connector_channels
        print(channels)
        # test_data = DreamData('', 6, 4)
        # print(test_data.)
    print('donzo')


if __name__ == '__main__':
    main()
