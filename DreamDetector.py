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

from Detector import Detector


class DreamDetector(Detector):
    def __init__(self, name=None, config=None, sub_run_name=None):
        super().__init__(config=config)
        self.detector_name = detector_name
        self.sub_run_name = sub_run_name
        self.hv = {}
        self.det_center = np.array([0, 0, 0])  # mm Center of detector
        self.det_orientation = np.array([0, 0, 0])
        self.feu_num = None
        self.feu_channels = {}

        self.config = config

    def load_from_run_config(self):
        if self.detector_name is None:  # Use first in included detectors
            self.detector_name = [name for name in self.run_config['included_detectors']
                                  if 'banco_ladder' not in name and 'm3_' not in name][0]

        detector_info = next((det for det in self.run_config['detectors']
                              if det.get('name') == self.detector_name), None)
        if detector_info is None:
            print(f'Error: Detector {self.detector_name} not found in run config.')
            return

        hv_channels = detector_info['hv_channels']
        center = detector_info.get('det_center_coords')
        self.det_center = np.array([center['x'], center['y'], center['z']])

        orientation = detector_info.get('det_orientation')
        self.det_orientation = np.array([orientation['x'], orientation['y'], orientation['z']])

        dream_feus = detector_info['dream_feus']
        feu_nums = list(set([chan[0] for chan in dream_feus.values()]))
        if len(feu_nums) != 1:
            print(f'Error: {self.detector_name} has multiple FEUs: {feu_nums}')
        else:
            self.feu_num = feu_nums[0]
        self.feu_channels = {name: chan[1] for name, chan in dream_feus.items()}

        if self.sub_run_name is None:  # Use first in sub runs
            self.sub_run_name = self.run_config['sub_runs'][0]['sub_run_name']

        sub_run_info = next((sub_run for sub_run in self.run_config['sub_runs']
                             if sub_run.get('sub_run_name') == self.sub_run_name), None)
        if sub_run_info is None:
            print(f'Error: Sub run {self.sub_run_name} not found in run config.')
            return

        hvs = sub_run_info['hvs']
        run_time = sub_run_info['run_time']

        self.hv = {hv_name: hvs[str(hv_channels[0])][str(hv_channels[1])]
                   for hv_name, hv_channels in hv_channels.items()}

