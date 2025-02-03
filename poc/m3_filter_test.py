#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 10 17:34 2024
Created in PyCharm
Created as saclay_micromegas/m3_filter_test

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
import json

import awkward as ak

from Detector_Classes.M3RefTracking import M3RefTracking


def main():
    # base_path = '/local/home/dn277127/Bureau/banco_test3/'
    # det_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    base_path = 'C:/Users/Dylan/Desktop/banco_test3/'
    det_info_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    run_config_file_name = 'run_config.json'

    m3_track_data = M3RefTracking(base_path)
    print(m3_track_data.ray_data)
    x_up = m3_track_data.ray_data['X_Up']
    print(len(x_up))
    print(x_up)
    x_up = ak.ravel(x_up)
    print(len(x_up))
    print(x_up)
    # return

    with open(f'{base_path}{run_config_file_name}', 'r') as file:
        run_config = json.load(file)

    detectors, included_detectors = run_config['detectors'], run_config['included_detectors']

    detector_geometries = get_detector_geometries(detectors, det_info_dir, included_detectors)
    traversing_event_ids = get_m3_det_traversing_events(base_path, detector_geometries)
    print(f'Traversing Event IDs: {traversing_event_ids}')
    print(f'Number of traversing events: {len(traversing_event_ids)}')
    print(f'Ballpark fraction of events kept: {len(traversing_event_ids) / traversing_event_ids[-1] * 100:.2f}%')

    plot_zs = [24, 600, 900, 1302]  # mm
    for z_i, plot_z in enumerate(plot_zs):
        m3_track_data = M3RefTracking(base_path)
        x_all, y_all, event_nums_all = m3_track_data.get_xy_positions(plot_z)
        x, y, event_nums = m3_track_data.get_xy_positions(plot_z, traversing_event_ids)

        if z_i == 0:
            fig_chi2_hist, ax_chi2_hist = plt.subplots()
            ax_chi2_hist.hist(m3_track_data.ray_data['Chi2X'], bins=100, label='X')
            ax_chi2_hist.hist(m3_track_data.ray_data['Chi2Y'], bins=100, label='Y')
            ax_chi2_hist.set_title(f'Chi2')
            ax_chi2_hist.legend()
            ax_chi2_hist.set_xlabel('Chi2')
            fig_chi2_hist.tight_layout()

        fig, ax = plt.subplots()
        ax.scatter(x_all, y_all, alpha=0.5, label='All Tracks')
        ax.scatter(x, y, alpha=0.5, label='Tracks Kept')
        ax.axvline(-250, color='gray', alpha=0.5)
        ax.axvline(250, color='gray', alpha=0.5)
        ax.axhline(-250, color='gray', alpha=0.5)
        ax.axhline(250, color='gray', alpha=0.5)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'xy Positions of Tracks Kept for z={plot_z:.2f}')
        ax.legend()
        fig.tight_layout()
    plt.show()

    print('donzo')


def get_m3_det_traversing_events(ray_directory, detector_geometries, det_bound_cushion=0.08):
    """
    Get event ids of events traversing any of the detectors in detector_geometries.
    :param ray_directory: Path to directory containing m3 tracking files
    :param detector_geometries: List of detector geometries to check for traversing events
    :param det_bound_cushion: Fractional cushion to add to detector bounds
    :return: List of event ids of events traversing any of the detectors
    """
    m3_track_data = M3RefTracking(ray_directory)
    masks = []
    for detector in detector_geometries:
        x, y, event_nums = m3_track_data.get_xy_positions(detector['z'])
        x_range, y_range = detector['x_max'] - detector['x_min'], detector['y_max'] - detector['y_min']
        x_min, x_max = detector['x_min'] - x_range * det_bound_cushion, detector['x_max'] + x_range * det_bound_cushion
        y_min, y_max = detector['y_min'] - y_range * det_bound_cushion, detector['y_max'] + y_range * det_bound_cushion
        masks.append((x > x_min) & (x < x_max) & (y > y_min) & (y < y_max))
    mask = np.any(masks, axis=0)
    event_ids = m3_track_data.ray_data['evn'][mask]
    return event_ids


def get_detector_geometries(detectors, det_info_dir, included_detectors=None):
    """
    Get detector geometries from run data in a format easier to check for traversing tracks.
    :param detectors: List of all detectors in config file with all run info, including geometry
    :param det_info_dir: Directory containing detector info files
    :param included_detectors: List of detectors to include in check. If None, all detectors are included.
    :return:
    """
    if included_detectors is None:
        included_detectors = [det['name'] for det in detectors]
    detector_geometries = []
    for det in detectors:
        if det['name'] in included_detectors and det['det_type'] != 'm3':
            det_info_path = f'{det_info_dir}{det["det_type"]}.json'
            with open(det_info_path, 'r') as file:
                det_info = json.load(file)
            x_size, y_size = det_info['det_size']['x'], det_info['det_size']['y']
            z = det['det_center_coords']['z']
            x_center, y_center = det['det_center_coords']['x'], det['det_center_coords']['y']
            x_angle, y_angle, z_angle = [det['det_orientation'][key] for key in ['x', 'y', 'z']]
            x_min, x_max, y_min, y_max = get_xy_max_min(x_size, y_size, x_center, y_center, x_angle, y_angle, z_angle)
            det_geom = {'z': z, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max, 'det_name': det['name']}
            detector_geometries.append(det_geom)
    return detector_geometries


def get_xy_max_min(x_size, y_size, x_center, y_center, x_angle, y_angle, z_angle):
    """
    Get the min and max x and y values of a detector given its size, center, and orientation.
    :param x_size: Size of detector in x direction
    :param y_size: Size of detector in y direction
    :param x_center: Center of detector in x direction
    :param y_center: Center of detector in y direction
    :param x_angle: Angle of detector in x direction
    :param y_angle: Angle of detector in y direction
    :param z_angle: Angle of detector in z direction
    :return:
    """
    # Calculate x, y, z coordinates of detector corners
    x_corners = np.array([-x_size / 2, x_size / 2, x_size / 2, -x_size / 2])
    y_corners = np.array([-y_size / 2, -y_size / 2, y_size / 2, y_size / 2])
    z_corners = np.array([0, 0, 0, 0])
    x_corners, y_corners, z_corners = rotate_3d(x_corners, y_corners, z_corners, x_angle, y_angle, z_angle)
    x_corners += x_center
    y_corners += y_center
    # Get min and max x, y values
    x_min, x_max = np.min(x_corners), np.max(x_corners)
    y_min, y_max = np.min(y_corners), np.max(y_corners)
    return x_min, x_max, y_min, y_max


def rotate_3d(x, y, z, x_angle, y_angle, z_angle):
    """
    Rotate 3d coordinates about the x, y, and z axes.
    :param x: x coordinates
    :param y: y coordinates
    :param z: z coordinates
    :param x_angle: Angle to rotate about x axis
    :param y_angle: Angle to rotate about y axis
    :param z_angle: Angle to rotate about z axis
    :return: Rotated x, y, z coordinates
    """
    # Rotate about x axis
    y, z = rotate_2d(y, z, x_angle)
    # Rotate about y axis
    x, z = rotate_2d(x, z, y_angle)
    # Rotate about z axis
    x, y = rotate_2d(x, y, z_angle)
    return x, y, z


def rotate_2d(x, y, angle):
    """
    Rotate 2d coordinates about the z axis.
    :param x: x coordinates
    :param y: y coordinates
    :param angle: Angle to rotate about z axis
    :return: Rotated x, y coordinates
    """
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    return x_rot, y_rot


if __name__ == '__main__':
    main()
