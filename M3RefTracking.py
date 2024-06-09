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
import uproot
import awkward as ak


class M3RefTracking:
    def __init__(self, ray_dir, file_nums='all', variables=None, single_track=True):
        self.ray_dir = ray_dir
        self.file_nums = file_nums
        self.single_track = single_track
        self.chi2_cut = 2
        if variables is None:
            self.variables = ['evn', 'evttime', 'rayN', 'Z_Up', 'X_Up', 'Y_Up', 'Z_Down', 'X_Down', 'Y_Down', 'Chi2X',
                              'Chi2Y']
        else:
            self.variables = variables

        self.ray_data = get_ray_data(ray_dir, file_nums, variables)
        if single_track:
            self.get_single_track_events()

    def get_xy_positions(self, z, event_list=None, one_track=True):
        return get_xy_positions(self.ray_data, z, event_list, one_track)

    def cut_on_chi2(self, chi2_cut):
        chi2_x, chi2_y = self.ray_data['Chi2X'], self.ray_data['Chi2Y']
        mask = (chi2_x < chi2_cut) & (chi2_y < chi2_cut)
        for var in self.ray_data.keys():
            self.ray_data[var] = self.ray_data[var][mask]

    def get_single_track_events(self):
        """
        Find events with only one track with both chi2_x and chi2_y less than chi2_cut.
        :return:
        """
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
                self.ray_data[var] = self.ray_data[var][mask]


def get_ray_data(ray_dir, file_nums='all', variables=None):
    if variables is None:
        variables = ['evn', 'evttime', 'rayN', 'Z_Up', 'X_Up', 'Y_Up', 'Z_Down', 'X_Down', 'Y_Down', 'Chi2X', 'Chi2Y']
    data = None
    for file_name in os.listdir(ray_dir):
        if not file_name.endswith('_rays.root'):
            continue

        if isinstance(file_nums, list):
            file_num = int(file_name.split('_')[-2])
            if file_num not in file_nums:
                continue

        with uproot.open(f'{ray_dir}{file_name}') as file:
            tree_name = f"{file.keys()[0].split(';')[0]};{max([int(key.split(';')[-1]) for key in file.keys()])}"
            tree = file[tree_name]  # Get tree with max ;# at end
            new_data = tree.arrays(variables, library='ak')
            if data is None:
                data = new_data
            else:
                data = ak.concatenate((data, new_data), axis=0)
                # for var in variables:
                #     data[var] = ak.concatenate((data[var], new_data[var]), axis=0)

    return data


def get_xy_positions(ray_data, z, event_list=None, one_track=True):
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
