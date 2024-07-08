#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 08 12:12 PM 2024
Created in PyCharm
Created as saclay_micromegas/Detector.py

@author: Dylan Neff, Dylan
"""

import numpy as np


class Detector:
    def __init__(self, name=None, config=None):
        self.name = None
        self.center = None  # mm Center of detector
        self.size = None  # mm Size of detector
        self.rotations = None
        self.config = config

        if self.config is not None:
            self.load_from_config()

    def load_from_config(self):
        self.name = self.config['name']
        self.center = self.config['center']
        self.size = self.config['size']
        self.rotations = self.config['rotations']

    def set_center(self, x=None, y=None, z=None):
        if isinstance(x, list) or isinstance(x, np.ndarray):
            x, y, z = x
        if isinstance(x, dict):
            x, y, z = x['x'], x['y'], x['z']

        if x is None:
            x = self.center[0]
        if y is None:
            y = self.center[1]
        if z is None:
            z = self.center[2]

        self.center = np.array([x, y, z])

    def set_size(self, x=None, y=None, z=None):
        if isinstance(x, list) or isinstance(x, np.ndarray):
            x, y, z = x
        if isinstance(x, dict):
            x, y, z = x['x'], x['y'], x['z']

        if x is None:
            x = self.size[0]
        if y is None:
            y = self.size[1]
        if z is None:
            z = self.size[2]

        self.size = np.array([x, y, z])
