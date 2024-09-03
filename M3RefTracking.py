#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 07 7:26 PM 2024
Created in PyCharm
Created as saclay_micromegas/M3RefTracking.py

@author: Dylan Neff, Dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm

import uproot
import awkward as ak



class M3RefTracking:
    def __init__(self, ray_dir, file_nums='all', variables=None, single_track=True):
        self.ray_dir = ray_dir
        self.file_nums = file_nums
        self.single_track = single_track
        self.chi2_cut = 1.5
        self.detector_xy_extent_cuts = {'x': [-250, 250], 'y': [-250, 250]}
        if variables is None:
            self.variables = ['evn', 'evttime', 'rayN', 'Z_Up', 'X_Up', 'Y_Up', 'Z_Down', 'X_Down', 'Y_Down', 'Chi2X',
                              'Chi2Y']
        else:
            self.variables = variables

        self.ray_data = get_ray_data(ray_dir, file_nums, variables)
        if single_track:
            self.get_single_track_events()

    def get_xy_positions(self, z, event_list=None, multi_track_events=False, one_track=True):
        if multi_track_events:
            return get_xy_positions_multi_track_events(self.ray_data, z, event_list, one_track)
        else:
            return get_xy_positions(self.ray_data, z, event_list)

    def get_traversing_triggers(self, z, x_bounds, y_bounds, expansion_factor=1):
        """
        Get the event numbers of events that traverse the detector, given by the x and y bounds at altitude z.
        :param z: mm Altitude at which to get the traversing events.
        :param x_bounds: mm Tuple of x bounds of detector at altitude z.
        :param y_bounds: mm Tuple of y bounds of detector at altitude z.
        :param expansion_factor: Factor to expand the bounds by.
        :return: List of event numbers that traverse the detector.
        """
        x_positions, y_positions, event_nums = self.get_xy_positions(z)
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        x_min, x_max = x_min - (x_max - x_min) * expansion_factor, x_max + (x_max - x_min) * expansion_factor
        y_min, y_max = y_min - (y_max - y_min) * expansion_factor, y_max + (y_max - y_min) * expansion_factor
        mask = (x_min < x_positions) & (x_positions < x_max) & (y_min < y_positions) & (y_positions < y_max)
        return event_nums[mask]

    def cut_on_chi2(self, chi2_cut):
        chi2_x, chi2_y = self.ray_data['Chi2X'], self.ray_data['Chi2Y']
        mask = (chi2_x < chi2_cut) & (chi2_y < chi2_cut)
        for var in ['X_Up', 'Y_Up', 'X_Down', 'Y_Down', 'Chi2X', 'Chi2Y']:
            self.ray_data[var] = self.ray_data[var][mask]

    def cut_on_det_size(self):
        x_up, x_down, y_up, y_down = [self.ray_data[x_i] for x_i in ['X_Up', 'X_Down', 'Y_Up', 'Y_Down']]
        x_min, x_max, y_min, y_max = self.detector_xy_extent_cuts['x'] + self.detector_xy_extent_cuts['y']
        mask = ((x_min < x_up) & (x_up < x_max) & (x_min < x_down) & (x_down < x_max) &
                (y_min < y_up) & (y_up < y_max) & (y_min < y_down) & (y_down < y_max))
        for var in ['X_Up', 'Y_Up', 'X_Down', 'Y_Down', 'Chi2X', 'Chi2Y']:
            self.ray_data[var] = self.ray_data[var][mask]

    def get_single_track_events(self):
        """
        Find events with only one track with both chi2_x and chi2_y less than chi2_cut and within detector areas.
        :return:
        """
        self.cut_on_det_size()
        chi2_x, chi2_y = self.ray_data['Chi2X'], self.ray_data['Chi2Y']
        num_good_tracks = ak.sum((chi2_x < self.chi2_cut) & (chi2_y < self.chi2_cut), axis=1)
        mask = num_good_tracks == 1
        self.ray_data = self.ray_data[mask]

        # Flatten from here, picking track with min chi2_x + chi2_y
        chi2_x, chi2_y = self.ray_data['Chi2X'], self.ray_data['Chi2Y']
        chi2_sum = chi2_x + chi2_y
        min_chi2 = ak.min(chi2_sum, axis=1)
        mask = chi2_sum == min_chi2

        for var in ['X_Up', 'Y_Up', 'X_Down', 'Y_Down', 'Chi2X', 'Chi2Y']:
            if self.variables is None or var in self.variables:
                self.ray_data[var] = ak.ravel(self.ray_data[var][mask])

    def plot_xy(self, z, event_list=None, multi_track_events=False, one_track=True):
        x, y, event_nums = self.get_xy_positions(z, event_list, multi_track_events, one_track)
        fig, ax = plt.subplots()
        ax.scatter(x, y, alpha=0.5)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'xy Positions at z={z:.2f} mm')
        fig.tight_layout()


def get_ray_data(ray_dir, file_nums='all', variables=None):
    if variables is None:
        variables = ['evn', 'evttime', 'rayN', 'Z_Up', 'X_Up', 'Y_Up', 'Z_Down', 'X_Down', 'Y_Down', 'Chi2X', 'Chi2Y']

    def read_file(file_name):
        if not file_name.endswith('_rays.root'):
            return None

        if isinstance(file_nums, list):
            file_num = int(file_name.split('_')[-2])
            if file_num not in file_nums:
                return None

        with uproot.open(f'{ray_dir}{file_name}') as file:
            tree_name = f"{file.keys()[0].split(';')[0]};{max([int(key.split(';')[-1]) for key in file.keys()])}"
            tree = file[tree_name]  # Get tree with max ;# at end
            new_data = tree.arrays(variables, library='ak')
            return new_data

    # List of ROOT files in the directory
    root_files = os.listdir(ray_dir)

    data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:  # Use ThreadPoolExecutor for parallel file processing
        # Use tqdm for a progress bar and map the read_file function across all root_files
        for new_data in tqdm(executor.map(read_file, root_files), total=len(root_files)):
            if new_data is not None:
                data.append(new_data)
    data = ak.concatenate(data, axis=0)

    return data


def get_xy_positions_multi_track_events(ray_data, z, event_list=None, one_track=True):
    if isinstance(ray_data, ak.highlevel.Array):  # If ray data is awkward array, convert relevant entries to dict of
        variables = ['evn', 'Z_Up', 'Z_Down', 'X_Up', 'X_Down', 'Y_Up', 'Y_Down']  # numpy arrays
        ray_data_hold = ray_data
        ray_data = {}
        for var in variables:
            ray_data[var] = ak.to_numpy(ray_data_hold[var])

    mask = np.full(ray_data['evn'].size, True)
    if event_list is not None:
        mask = np.isin(ray_data['evn'], event_list)
    if one_track:
        one_track_mask = np.array([x.size == 1 for x in ray_data['X_Up']])
        mask = mask & one_track_mask

    z_up, z_down = ray_data['Z_Up'][mask], ray_data['Z_Down'][mask]
    x_up, x_down = ray_data['X_Up'][mask], ray_data['X_Down'][mask]
    y_up, y_down = ray_data['Y_Up'][mask], ray_data['Y_Down'][mask]
    event_nums = ray_data['evn'][mask]

    x_up, x_down = np.array([x[0] for x in x_up]), np.array([x[0] for x in x_down])
    y_up, y_down = np.array([y[0] for y in y_up]), np.array([y[0] for y in y_down])

    # Calculate the interpolation factors
    t = (z - z_up) / (z_down - z_up)
    t = np.mean(t)

    # Interpolate the x and y positions
    x_positions = x_up + t * (x_down - x_up)
    y_positions = y_up + t * (y_down - y_up)

    return x_positions, y_positions, event_nums


def get_xy_positions(ray_data, z, event_list=None):
    if isinstance(ray_data, ak.highlevel.Array):  # If ray data is awkward array, convert relevant entries to dict of
        variables = ['evn', 'Z_Up', 'Z_Down', 'X_Up', 'X_Down', 'Y_Up', 'Y_Down']  # numpy arrays
        ray_data_hold = ray_data
        ray_data = {}
        for var in variables:
            ray_data[var] = ak.to_numpy(ray_data_hold[var])

    mask = np.full(ray_data['evn'].size, True)
    if event_list is not None:
        mask = np.isin(ray_data['evn'], event_list)

    z_up, z_down = ray_data['Z_Up'][mask], ray_data['Z_Down'][mask]
    x_up, x_down = ray_data['X_Up'][mask], ray_data['X_Down'][mask]
    y_up, y_down = ray_data['Y_Up'][mask], ray_data['Y_Down'][mask]
    event_nums = ray_data['evn'][mask]

    # Calculate the interpolation factors
    t = (z - z_up) / (z_down - z_up)
    t = np.mean(t)

    # Interpolate the x and y positions
    x_positions = x_up + t * (x_down - x_up)
    y_positions = y_up + t * (y_down - y_up)

    return x_positions, y_positions, event_nums
