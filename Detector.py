#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 08 12:12 PM 2024
Created in PyCharm
Created as saclay_micromegas/Detector.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R


class Detector:
    def __init__(self, name=None, center=None, size=None, rotations=None, config=None):
        self.name = name
        self.center = center  # mm Center of detector
        self.size = size  # mm Size of detector
        self.active_size = size  # mm Size of active area of detector. NOT YET IMPLEMENTED
        self.rotations = rotations  # List of rotations to apply to detector
        self.config = config  # Config dictionary to load detector from

        self.euler_rotation_order = 'zyx'  # Order in which 2D rotations are applied

        self.cluster_centroids = None

        if self.config is not None:
            self.load_from_config()

    def load_from_config(self):
        self.name = self.config.get('name')
        self.set_center(self.config.get('det_center_coords'))
        self.set_size(self.config.get('det_size'))

        self.rotations = [] if self.rotations is None else self.rotations
        self.load_rotations_from_config()

    def load_rotations_from_config(self):
        orient = self.config['det_orientation']
        for axis, angle in orient.items():  # Check for 180 deg rotations (flips) and do those first
            if angle == 180:
                self.rotations.append([angle if axis_i == axis else 0 for axis_i in self.euler_rotation_order])
        for axis, angle in orient.items():
            if angle != 180 and angle != 0:
                self.rotations.append([angle if axis_i == axis else 0 for axis_i in self.euler_rotation_order])

    def set_center(self, x=None, y=None, z=None):
        if x is None and y is None and z is None:
            self.center = None

        elif isinstance(x, list) or isinstance(x, np.ndarray):
            x, y, z = x
        elif isinstance(x, dict):
            x, y, z = x['x'], x['y'], x['z']

        else:
            if x is None:
                x = self.center[0] if self.center is not None else 0
            if y is None:
                y = self.center[1] if self.center is not None else 0
            if z is None:
                z = self.center[2] if self.center is not None else 0

        self.center = np.array([x, y, z])

    def set_size(self, x=None, y=None, z=None):
        if x is None and y is None and z is None:
            self.size = None

        elif isinstance(x, list) or isinstance(x, np.ndarray):
            x, y, z = x
        elif isinstance(x, dict):
            x, y, z = x['x'], x['y'], x['z']

        else:
            if x is None:
                x = self.size[0] if self.size is not None else 0
            if y is None:
                y = self.size[1] if self.size is not None else 0
            if z is None:
                z = self.size[2] if self.size is not None else 0

        self.size = np.array([x, y, z])
        self.active_size = np.array([x, y, z])  # Just set active size to size for now

    def set_rotations(self, rotations):
        self.rotations = rotations

    def add_rotation(self, angle, axis=None):
        if isinstance(angle, list) or isinstance(angle, np.ndarray):
            if axis is None:
                rotations = angle
            else:
                rotations = [[angle_i if axis_i == axis else 0 for axis_i in self.euler_rotation_order]
                             for angle_i in angle]
        else:
            rotations = [angle if axis_i == axis else 0 for axis_i in self.euler_rotation_order]

        if isinstance(rotations[0], list):
            self.rotations += rotations
        else:
            self.rotations.append(rotations)

    def replace_last_rotation(self, angle, axis=None):
        if self.rotations is None:
            self.rotations = []
        elif len(self.rotations) > 0:
            self.rotations.pop(-1)
        self.add_rotation(angle, axis)

    def remove_last_rotation(self):
        if len(self.rotations) > 0:
            self.rotations.pop(-1)

    def convert_coords_to_global(self, coords):
        # zs = np.full((len(self.cluster_centroids), 1), 0)  # Add z coordinate to centroids
        # self.cluster_centroids = np.hstack((self.cluster_centroids, zs))  # Combine x, y, z

        # Center coordinates around center of detector
        coords = coords - self.active_size / 2

        # Rotate cluster centroids to global coordinates
        coords = rotate_coordinates(coords, self.rotations, self.euler_rotation_order)

        # Translate cluster centroids to global coordinates
        coords = coords + self.center

        return coords


def rotate_coordinates(coords, rotations, rotation_order='zyx'):
    if rotations is None:
        return coords

    for rotation in rotations:
        coords = rotate_coordinates_single(coords, rotation, rotation_order)

    return coords


def rotate_coordinates_single(coords, rotation, rotation_order='zyx'):
    r = R.from_euler(rotation_order, rotation, degrees=True)
    coords = r.apply(coords)

    return coords
