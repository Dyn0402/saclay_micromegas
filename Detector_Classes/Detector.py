#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 08 12:12 PM 2024
Created in PyCharm
Created as saclay_micromegas/Detector.py

@author: Dylan Neff, Dylan
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import copy


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

    def copy(self):
        return copy.deepcopy(self)

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

        # Center coordinates around center of detector. Accuracy of active size not critical, will be aligned away
        coords = coords - self.active_size / 2

        # Rotate cluster centroids to global coordinates
        coords = rotate_coordinates(coords, self.rotations, self.euler_rotation_order)

        # Translate cluster centroids to global coordinates
        coords = coords + self.center

        return coords

    def convert_global_coords_to_local(self, coords):
        # Translate global coordinates to the detector's local frame by subtracting the detector center
        coords = coords - self.center

        # Rotate coordinates from global to local by applying the inverse of the detector's rotations
        coords = rotate_coordinates_inverse(coords, self.rotations, self.euler_rotation_order)

        # Adjust coordinates to be centered in the detector's local frame
        coords = coords + self.active_size / 2

        return coords

    def write_det_alignment_to_file(self, file_path):
        with open(file_path, 'w') as file:
            file.write(f'name: {self.name}\n')
            file.write(f'det_center_coords: {[float(center) for center in self.center]}\n')
            file.write(f'det_size: {[float(size) for size in self.size]}\n')
            file.write(f'det_active_size: {[float(size) for size in self.active_size]}\n')
            file.write('det_orientation:\n')
            file.write(f'euler_rotation_order: {self.euler_rotation_order}\n')
            for rotation_i, rotation in enumerate(self.rotations):
                file.write(f'rotation {rotation_i}: {[float(rot) for rot in rotation]}\n')

    def read_det_alignment_from_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'name:' in line:
                    self.name = line.split(':')[-1].strip()
                elif 'det_center_coords:' in line:
                    self.set_center([float(coord) for coord in line.split(':')[-1].strip()[1:-1].split(',')])
                elif 'det_size:' in line:
                    self.set_size([float(coord) for coord in line.split(':')[-1].strip()[1:-1].split(',')])
                elif 'det_active_size:' in line:
                    self.active_size = np.array([float(coord) for coord in line.split(':')[-1].strip()[1:-1].split(',')])
                elif 'det_orientation:' in line:
                    self.rotations = []
                elif 'euler_rotation_order:' in line:
                    self.euler_rotation_order = line.split(':')[-1].strip()
                else:
                    self.add_rotation([float(angle) for angle in line.split(':')[-1].strip()[1:-1].split(',')])


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


def rotate_coordinates_inverse(coords, rotations, rotation_order='zyx'):
    if rotations is None:
        return coords

    # Apply the rotations in reverse order and negate the angles to reverse the transformation
    for rotation in reversed(rotations):
        coords = rotate_coordinates_single_inverse(coords, rotation, rotation_order)

    return coords


def rotate_coordinates_single_inverse(coords, rotation, rotation_order='zyx'):
    r = R.from_euler(rotation_order, rotation, degrees=True).inv()  # Invert the rotation
    coords = r.apply(coords)

    return coords
