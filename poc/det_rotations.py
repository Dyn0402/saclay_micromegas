#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 08 3:16 PM 2024
Created in PyCharm
Created as saclay_micromegas/det_rotations.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    x_size, y_size = 10, 5
    x_center, y_center = 0, 0
    xy_cushion = 0.08
    x_angle, y_angle, z_angle = np.pi, 0, np.pi / 4 / 10
    x_min, x_max, y_min, y_max = get_xy_max_min(x_size, y_size, x_center, y_center, x_angle, y_angle, z_angle)
    print(x_min, x_max, y_min, y_max)
    fig, ax = plt.subplots()

    # Plot no rotation and rotated detector, along with x and y min and max
    ax.plot([-x_size / 2, x_size / 2, x_size / 2, -x_size / 2, -x_size / 2],
            [-y_size / 2, -y_size / 2, y_size / 2, y_size / 2, -y_size / 2], label='No Rotation')

    # Plot rotated detector
    x_corners = np.array([-x_size / 2, x_size / 2, x_size / 2, -x_size / 2])
    y_corners = np.array([-y_size / 2, -y_size / 2, y_size / 2, y_size / 2])
    z_corners = np.array([0, 0, 0, 0])
    x_corners, y_corners, z_corners = rotate_3d(x_corners, y_corners, z_corners, x_angle, y_angle, z_angle)
    x_corners += x_center
    y_corners += y_center
    x_corners = np.append(x_corners, x_corners[0])
    y_corners = np.append(y_corners, y_corners[0])
    ax.plot(x_corners, y_corners, label='Rotated Detector')

    # Plot x and y min and max with cushion
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * xy_cushion
    x_max += x_range * xy_cushion
    y_min -= y_range * xy_cushion
    y_max += y_range * xy_cushion
    ax.axvline(x_min, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x_max, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y_min, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y_max, color='gray', linestyle='--', alpha=0.5)

    # Make aspect ratio equal
    ax.set_aspect('equal', 'box')
    ax.legend()
    plt.show()
    print('donzo')


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
