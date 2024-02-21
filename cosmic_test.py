#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 13 5:30 PM 2024
Created in PyCharm
Created as saclay_micromegas/cosmic_test.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd

import uproot
import awkward as ak

from fe_analysis import *


def main():
    # ray_first_test()
    get_det_hits()
    # ray_plot_test()
    plt.show()
    print('donzo')


def get_det_hits():
    signal_ped_file_path = 'F:/Saclay/Cosmic_Data/CosTb_380V_stats_pedthr_240212_11H42_03.root'
    signal_file_path = 'F:/Saclay/Cosmic_Data/CosTb_380V_stats_datrun_240212_11H42_000_03.root'
    # signal_ped_file_path = '/local/home/dn277127/Documents/Cosmic_Data/CosTb_380V_stats_pedthr_240212_11H42_03.root'
    # signal_file_path = '/local/home/dn277127/Documents/Cosmic_Data/CosTb_380V_stats_datrun_240212_11H42_000_03.root'
    det_width = 130
    num_det_ranges = [[0, 4], [4, 8]]  # [Top, Bottom]
    # det_zs = [738, 278.6]  # mm [Top, Bottom] to bottom of the board, might want to raise a bit. Pre-stand data
    # det_x_centers = [31.5, 37.5]  # mm [Top, Bottom] Pre-stand data
    # det_y_centers = [-51, -55]  # mm [Top, Bottom] Pre-stand data
    # det_zs = [293.2 + 92 + 225, 293.2 + 225]  # mm [Top, Bottom] to bottom of the board maybe raise. For data with stand
    det_zs = [293.2 + 92 * 3 + 225, 293.2 + 92 * 2 + 225, 293.2 + 92 + 225, 293.2 + 225, 293.2 + 225 - 92]  # mm [Top, Bottom] to bottom of the board maybe raise. For data with stand
    det_x_centers = [0, 0, 0, 0, 0]  # mm [Top, Bottom] For data with stand
    det_y_centers = [0, 0, 0, 0, 0]  # mm [Top, Bottom] For data with stand
    det_defs = []
    for det_x_cent, det_y_cent, det_z in zip(det_x_centers, det_y_centers, det_zs):
        det_defs.append(extent_to_vertices([
            (-det_width / 2 + det_x_cent, -det_width / 2 + det_y_cent, det_z - 1),
            (det_width / 2 + det_x_cent, det_width / 2 + det_y_cent, det_z + 1)]))
    # det_bot_def = extent_to_vertices([
    #     (-det_width / 2 + det_x_centers[1], -det_width / 2 + det_y_centers[1], det_zs[0] - 1),
    #     (det_width / 2 + det_x_centers[1], det_width / 2 + det_y_centers[1], det_zs[0] + 1)])
    # det_top_def = extent_to_vertices([
    #     (-det_width / 2 + det_x_centers[0], -det_width / 2 + det_y_centers[0], det_zs[1] - 1),
    #     (det_width / 2 + det_x_centers[0], det_width / 2 + det_y_centers[0], det_zs[1] + 1)])

    hit_coords_dets = []
    for det_i, (num_dets, det_z) in enumerate(zip(num_det_ranges, det_zs)):
        pedestals, noise_thresholds = run_pedestal(signal_ped_file_path, num_dets, noise_sigmas=7, plot_pedestals=False)
        no_noise_events, event_numbers, total_events = process_file(signal_file_path, pedestals, noise_thresholds,
                                                                    num_dets)
        # print(no_noise_events)
        # print(f'Number of events: {total_events}')
        # print(f'no_noise_events shape: {no_noise_events.shape}')
        no_noise_events_max = get_sample_max(no_noise_events)
        # plot_urw_position(no_noise_events_max, separate_event_plots=True,
        #                   thresholds=noise_thresholds, max_events=10,
        #                   event_numbers=event_numbers, title='Signal Events')

        n_events, n_dets, n_strips = no_noise_events_max.shape
        no_noise_events_max = np.reshape(no_noise_events_max, (n_events, n_dets // 2, n_strips * 2))

        max_strips = get_max_strip(no_noise_events_max)
        hit_coords = max_strips / 128 * det_width - det_width / 2 + np.array([det_y_centers[1], det_x_centers[1]])
        # Flip the innermost dimension to match the order of the detectors
        print(max_strips)
        # print(hit_coords)
        hit_coords = np.flip(hit_coords, axis=1)  # y comes first in the data due to how I plugged them in
        print(hit_coords)
        print(event_numbers)
        hit_coords_dets.append(dict(zip(event_numbers, hit_coords)))

    all_events = set(hit_coords_dets[0].keys()).union(set(hit_coords_dets[1].keys()))

    for event_num in all_events:
        if event_num != 415:
            continue
        ax, got_track = ray_plot_test(event_num, plot_if_no_track=False)
        if not got_track:
            continue
        hit_coords = []
        for det_i, hit_coords_det in enumerate(hit_coords_dets):
            coords = None if event_num not in hit_coords_det else np.array([*hit_coords_det[event_num], det_zs[det_i]])
            coords = None  # Hack for now to just display boards
            hit_coords.append(coords)
        for det_i, det_def in enumerate(det_defs):
            if det_i >= len(hit_coords):
                plot_urw_hit(None, det_def, ax_in=ax)
            else:
                plot_urw_hit(hit_coords[det_i], det_def, ax_in=ax)
        # bot_hit_coord = None if event_num not in hit_coords_dets[0] \
        #     else np.array([*hit_coords_dets[0][event_num], det_zs[0]])
        # top_hit_coord = None if event_num not in hit_coords_dets[1] \
        #     else np.array([*hit_coords_dets[1][event_num], det_zs[1]])
        # plot_urw_hit(bot_hit_coord, det_bot_def, ax_in=ax)
        # plot_urw_hit(top_hit_coord, det_top_def, ax_in=ax)
        ax.set_title(f'Event {event_num}')
        ax.legend()
        plt.gcf().tight_layout()


def ray_plot_test(event_num=None, plot_if_no_track=True):
    ray_file_path = 'F:/Saclay/Cosmic_Data/CosTb_380V_stats_datrun_240212_11H42_rays.root'
    # ray_file_path = '/local/home/dn277127/Documents/Cosmic_Data/CosTb_380V_stats_datrun_240212_11H42_rays.root'
    variables = ['evn', 'evttime', 'rayN', 'Z_Up', 'X_Up', 'Y_Up', 'Z_Down', 'X_Down', 'Y_Down', 'Chi2X', 'Chi2Y']
    det_top_def = extent_to_vertices([(-250, -250, 1300), (250, 250, 1304)])
    det_bot_def = extent_to_vertices([(-250, -250, 22), (250, 250, 26)])

    ray_root_file = uproot.open(ray_file_path)
    ray_tree = ray_root_file['T;1']
    data = ray_tree.arrays(variables, library='np')
    # print(data)
    print(data.keys())
    # Turn into pandas dataframe
    df = pd.DataFrame(data)
    # print(df)

    if event_num is None:
        event_num = 1
    event_0 = df.iloc[event_num]
    print(event_0)
    missing_xy = any([len(x) == 0 for x in [event_0['X_Up'], event_0['Y_Up'], event_0['X_Down'], event_0['Y_Down']]])
    if missing_xy:
        top_hit, bottom_hit = None, None
        if not plot_if_no_track:
            return None, False
    else:
        top_hit = np.array([event_0['X_Up'][0], event_0['Y_Up'][0], event_0['Z_Up']])
        bottom_hit = np.array([event_0['X_Down'][0], event_0['Y_Down'][0], event_0['Z_Down']])
    ax = plot_ray(top_hit, bottom_hit, det_top_def, det_bot_def)

    return ax, top_hit is not None


def ray_first_test():
    signal_file_path = 'C:/Users/Dylan/Desktop/test/test_signal.root'
    ray_file_path = 'C:/Users/Dylan/Desktop/test/rays_CosTb_380V_stats_datrun_240212_11H42_000.root'
    # Open the ROOT file with uproot
    signal_root_file = uproot.open(signal_file_path)
    ray_root_file = uproot.open(ray_file_path)

    # Access the tree in the file
    signal_tree = signal_root_file['T;32']
    ray_tree = ray_root_file['T;1']

    # Get the variable data from the tree
    signal_evttime = ak.flatten(signal_tree['evttime'].array(), axis=None)
    ray_evttime = ak.flatten(ray_tree['evttime'].array(), axis=None)
    print(signal_evttime)
    print(ray_evttime)

    print(f'Min ray time: {min(ray_evttime)}')
    print(f'Min signal time: {min(signal_evttime)}')

    print(f'Signal time shift: {(signal_evttime - min(signal_evttime)) * 1000}')

    # n_plot = int(len(ray_evttime) / 8)
    n_plot = int(len(ray_evttime) / 1)
    fig, ax = plt.subplots()
    ax.scatter(signal_evttime[:n_plot], ray_evttime[:n_plot], s=1)
    ax.set_xlabel('Signal Time')
    ax.set_ylabel('Ray Time')

    fig, ax = plt.subplots()
    ax.plot(signal_evttime[:n_plot], marker='o', linestyle='None', label='Signal')

    fig, ax = plt.subplots()
    ax.plot(np.diff(signal_evttime[:n_plot]), marker='o', linestyle='None', label='Signal')

    fig, ax = plt.subplots()
    ax.hist(np.diff(signal_evttime), bins=100, histtype='step', label='Signal')
    ax.hist(np.diff(ray_evttime), bins=100, histtype='step', label='Ray')
    ax.set_yscale('log')
    ax.legend()

    signal_strip_ampl_corr = ak.to_numpy(signal_tree['StripAmpl_MGv2_corr'].array())
    print(signal_strip_ampl_corr)
    print(signal_strip_ampl_corr.shape)

    plt.show()


def plot_ray(ray_top_coord, ray_bot_coord, det_top_def, det_bot_def):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create Poly3DCollection for filled rectangles
    rect1 = Poly3DCollection([det_top_def], alpha=0.5, facecolors='gray', edgecolors='black', linewidths=1)
    rect2 = Poly3DCollection([det_bot_def], alpha=0.5, facecolors='gray', edgecolors='black', linewidths=1)

    # Add rectangles to the plot
    ax.add_collection3d(rect1)
    ax.add_collection3d(rect2)

    if ray_top_coord is not None and ray_bot_coord is not None:
        # Draw small circles at the points
        ax.scatter(ray_top_coord[0], ray_top_coord[1], ray_top_coord[2], s=50, color='red', label='Hit')
        ax.scatter(ray_bot_coord[0], ray_bot_coord[1], ray_bot_coord[2], s=50, color='red')

        # Draw a line connecting the circles
        line_points = np.array([ray_top_coord, ray_bot_coord])
        ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], color='orange', label='Track')

    # Set labels and legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_zlim(0, 2000)
    ax.legend()
    fig.tight_layout()

    return ax


def plot_urw_hit(hit_coords, det_urw_def, ax_in=None):
    """
    Plot the hit in the URW.
    :param hit_coords:
    :param det_urw_def:
    :param ax_in:
    :return:
    """
    if ax_in is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax_in

    # Create Poly3DCollection for filled rectangles
    rect = Poly3DCollection([det_urw_def], alpha=0.5, facecolors='blue', edgecolors='black', linewidths=1)

    # Add rectangles to the plot
    ax.add_collection3d(rect)

    if hit_coords is not None:
        # Draw small circles at the points
        ax.scatter(hit_coords[0], hit_coords[1], hit_coords[2], s=50, color='red')

    if ax_in is None:
        # Set labels and legend
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        ax.set_zlim(0, 2000)
        ax.legend()
        fig.tight_layout()


def extent_to_vertices(extent):
    """
    Convert rectangle extent to four vertices.

    Parameters:
    extent (tuple): Tuple containing two points representing the extent of the rectangle.
                    Each point is a tuple (x, y, z).

    Returns:
    numpy.ndarray: Array of shape (4, 3) containing the four vertices of the rectangle.
    """
    x_min, y_min, z_min = extent[0]
    x_max, y_max, z_max = extent[1]

    vertices = np.array([
        [x_min, y_min, z_min],
        [x_min, y_max, z_min],
        [x_max, y_max, z_min],
        [x_max, y_min, z_min]
    ])

    return vertices


if __name__ == '__main__':
    main()
