#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 17 6:15 PM 2024
Created in PyCharm
Created as saclay_micromegas/BancoTelescope.py

@author: Dylan Neff, Dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from DetectorConfigLoader import DetectorConfigLoader
from BancoLadder_new import BancoLadder


class BancoTelescope:
    def __init__(self, det_config_loader=None, sub_run_name=None, data_dir=None, noise_dir=None):
        self.ladders = []
        self.det_config_loader = det_config_loader
        self.sub_run_name = sub_run_name
        self.data_dir = data_dir
        self.noise_dir = noise_dir

        if self.det_config_loader is not None:
            self.load_from_config(self.det_config_loader)

    def load_from_config(self, det_config_loader):
        for detector_name in det_config_loader.included_detectors:
            det_config = det_config_loader.get_det_config(detector_name, sub_run_name=self.sub_run_name)
            if det_config['det_type'] == 'banco':
                det = BancoLadder(config=det_config)
                print(f'Loading {det.name}')
                print(f'Center: {det.center}')
                print(f'Size: {det.size}')
                print(f'Rotations: {det.rotations}')
                print(f'Active Size: {det.active_size}')
                self.ladders.append(det)

    def read_data(self, ray_data=None):
        run_name = get_banco_run_name(self.data_dir)
        for ladder in self.ladders:
            print(f'\nReading data for {ladder.name}')
            data_path = f'{self.data_dir}{run_name}{ladder.ladder_num}.root'
            noise_path = f'{self.noise_dir}Noise_{ladder.ladder_num}.root'

            banco_traversing_triggers = None
            if ray_data is not None:
                banco_traversing_triggers = ladder.get_banco_traversing_triggers(ray_data)

            print('Reading banco_noise')
            ladder.read_banco_noise(noise_path)
            print('Reading banco_data')
            ladder.read_banco_data(data_path)
            print('Getting data noise pixels')
            ladder.get_data_noise_pixels()
            print('Combining data noise')
            ladder.combine_data_noise()
            print('Clustering data')
            ladder.cluster_data(min_pixels=1, max_pixels=8, chip=None, event_list=banco_traversing_triggers)
            print('Getting largest clusters')
            ladder.get_largest_clusters()
            print('Converting cluster coords')
            ladder.convert_cluster_coords()

    def align_ladders(self, ray_data=None):
        for ladder in self.ladders:
            ladder.align_ladder(ray_data)


def get_banco_run_name(base_dir, start_string='multinoiseScan', end_string='-ladder'):
    """
    Find the name of the banco run from the base directory. Run name starts with start string and ends with end string,
    with the date and time in between. Use the first file containing both strings.
    :param base_dir: Base directory to search for run name.
    :param start_string: String to start run name.
    :param end_string: String to end run name.
    :return: Run name.
    """
    run_name = None
    for file in os.listdir(base_dir):
        if start_string in file and end_string in file:
            # Run name is string up to end string
            run_name = file.split(end_string)[0] + end_string
            break
    return run_name