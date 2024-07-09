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
    def __init__(self, name=None, center=None, size=None, rotations=None, config=None):
        super().__init__(name=name, center=center, size=size, rotations=rotations, config=config)
        self.hv = {}
        self.feu_num = None
        self.feu_channels = []

        self.config = config
        if self.config is not None:
            self.load_from_config_dream()

    def load_from_config_dream(self):
        dream_feus = self.config['dream_feus']
        feu_nums = list(set([chan[0] for chan in dream_feus.values()]))
        if len(feu_nums) != 1:
            print(f'Error: {self.name} has multiple FEUs: {feu_nums}')
        else:
            self.feu_num = feu_nums[0]
        for axis_name, slot_chan in dream_feus.items():
            self.feu_channels.append(slot_chan[1])

        self.hv = self.config['hvs']


class DreamSubDetector:
    def __init__(self):
        pass
