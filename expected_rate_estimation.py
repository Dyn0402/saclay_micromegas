#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on January 29 1:47 PM 2024
Created in PyCharm
Created as saclay_micromegas/expected_rate_estimation.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def main():
    detector = {'x_min': -11.3875, 'x_max': 1.35625, 'y_min': -1.355, 'y_max': 11.37}  # Detector boundaries cm
    det_center = ((detector['x_max'] + detector['x_min']) / 2, (detector['y_max'] + detector['y_min']) / 2)
    n_rays = 100000  # Number of rays to generate

    # source_activity = 3.7e5  # Bq  (3.7e5 is 10 uCi, my ballpark guess for the source activity)
    fe55_half_life = 2.737  # years
    nominal_activity = 9.62e6  # Bq
    nominal_date = datetime(2023, 5, 22)
    experiment_date = datetime(2023, 12, 1)
    lam = np.log(2) / fe55_half_life  # Decay constant 1/years
    source_activity = nominal_activity * np.exp(-lam * (experiment_date - nominal_date).days / 365.25)
    print(f'Fe55 activity: {source_activity:.2e} Bq')

    transmission = 0.99999  # Probability of photon passing through detector without interacting

    ray_z_origins = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]  # cm z origins of rays

    hit_fractions, rates = [], []
    for ray_z_origin in ray_z_origins:
        ray_origin = (det_center[0] + 0.5, det_center[1] - 0.5, ray_z_origin)  # Ray origin (x, y, z) cm
        intercepts = generate_rays(n_rays, ray_origin)
        ray_hits = rays_hit_rectangular_detector(intercepts, detector['x_min'], detector['x_max'],
                                                 detector['y_min'], detector['y_max'])
        hit_fraction = len(ray_hits) / n_rays
        expected_rate = calc_expected_rate(source_activity, hit_fraction) * (1 - transmission)
        print(f'z={ray_z_origin}cm,  hit fraction: '
              f'{hit_fraction * 100:.2f}% +- {np.sqrt(hit_fraction * (1 - hit_fraction) / n_rays) * 100:.2f}%,  '
              f'expected rate: {expected_rate:.2f} Hz')
        plot_ray_hit_distribution(ray_hits, ray_origin, detector['x_min'], detector['x_max'], detector['y_min'],
                                  detector['y_max'], hit_fraction)
        hit_fractions.append(hit_fraction)
        rates.append(expected_rate)
    plot_hit_fractions_vs_z(ray_z_origins, hit_fractions)
    print('{' + ', '.join([f'{z}: {rate:.2f}' for z, rate in zip(ray_z_origins, rates)]) + '}')
    plt.show()
    print('donzo')


def generate_rays(n_rays, ray_origin):
    """
    Generate n_rays rays from the ray_origin and return their x, y coordinate where the ray intersects the z=0 plane
    :param n_rays:
    :param ray_origin:
    :return:
    """
    intercepts = np.zeros((n_rays, 2))
    for i in range(n_rays):
        phi, theta = get_random_ray()
        ray_unit_vector = calc_ray_unit_vector(phi, theta)
        intercepts[i] = calculate_ray_xy_intercept(ray_origin, ray_unit_vector)
    return intercepts


def get_random_ray():
    """
    Generate a random ray from the origin
    :return: phi, theta of ray
    """
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.arccos(np.random.uniform(-1, 1))
    return phi, theta


def calc_ray_unit_vector(phi, theta):
    """
    Calculate the unit vector of a ray from the origin
    :param phi:
    :param theta:
    :return:
    """
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return x, y, z


def calculate_ray_xy_intercept(ray_origin, ray_unit_vector):
    """
    Calculate the x and y coordinates at which the ray intercepts the z=0 plane
    :param ray_origin: 3D origin of ray (x, y, z)
    :param ray_unit_vector: Unit vector of ray (x, y, z)
    :return: x, y coordinates of ray intercept
    """
    if ray_unit_vector[2] == 0:
        return None, None
    if ray_origin[2] == 0:
        return ray_origin[0], ray_origin[1]
    if ray_unit_vector[2] * ray_origin[2] > 0:
        return None, None
    x = ray_origin[0] - ray_origin[2] * ray_unit_vector[0] / ray_unit_vector[2]
    y = ray_origin[1] - ray_origin[2] * ray_unit_vector[1] / ray_unit_vector[2]
    return x, y


def ray_hit_rectangular_detector(ray_x, ray_y, det_x_min, det_x_max, det_y_min, det_y_max):
    """
    Determine if a ray intercepts a rectangular detector on the z=0 plane
    :param ray_x: x position of the ray when it intercepts the z=0 plane
    :param ray_y: y position of the ray when it intercepts the z=0 plane
    :param det_x_min: Detector's lower x boundary
    :param det_x_max: Detector's upper x boundary
    :param det_y_min: Detector's lower y boundary
    :param det_y_max: Detector's upper y boundary
    :return: Boolean of whether the ray intercepts the detector
    """
    if det_x_min <= ray_x <= det_x_max and det_y_min <= ray_y <= det_y_max:
        return True
    else:
        return False


def rays_hit_rectangular_detector(ray_intercept_array, det_x_min, det_x_max, det_y_min, det_y_max):
    """
    Determine if a set of rays intercept a rectangular detector on the z=0 plane
    :param ray_intercept_array: Array of ray intercepts (x, y)
    :param det_x_min: Detector's lower x boundary
    :param det_x_max: Detector's upper x boundary
    :param det_y_min: Detector's lower y boundary
    :param det_y_max: Detector's upper y boundary
    :return: Boolean array of whether each ray intercepts the detector
    """
    rays = np.array(ray_intercept_array)
    ray_hits = rays[(det_x_min <= rays[:, 0]) & (rays[:, 0] <= det_x_max) &
                    (det_y_min <= rays[:, 1]) & (rays[:, 1] <= det_y_max)]
    return ray_hits


def calc_expected_rate(source_activity, hit_fraction):
    """
    Calculate the expected rate of hits on the detector
    :param source_activity: Activity of the source in Bq
    :param hit_fraction: Fraction of rays that hit the detector
    :return: Expected rate of hits on the detector
    """
    return source_activity * hit_fraction


def plot_ray_hit_distribution(ray_hits, ray_origin, det_x_min, det_x_max, det_y_min, det_y_max, hit_fraction):
    """
    Plot the distribution of ray intercepts on the z=0 plane
    :param ray_hits: Array of ray intercepts (x, y)
    :param ray_origin: Origin of the rays (x, y, z)
    :param det_x_min: Detector's lower x boundary
    :param det_x_max: Detector's upper x boundary
    :param det_y_min: Detector's lower y boundary
    :param det_y_max: Detector's upper y boundary
    :param hit_fraction: Fraction of rays that hit the detector
    :return:
    """
    det_x_len, det_y_len = det_x_max - det_x_min, det_y_max - det_y_min
    fig, ax = plt.subplots()
    plt.title(f'Ray Hit Distribution for Ray Origin {[round(x_i, 2) for x_i in ray_origin]} cm')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.axhline(det_y_min, color='black', label='Detector Boundary')
    plt.axhline(det_y_max, color='black')
    plt.axvline(det_x_min, color='black')
    plt.axvline(det_x_max, color='black')
    plt.xlim(det_x_min - det_x_len * 0.1, det_x_max + det_x_len * 0.1)
    plt.ylim(det_y_min - det_y_len * 0.2, det_y_max + det_y_len * 0.1)
    plt.text(det_x_min + det_x_len * 0.02, det_y_max + det_y_len * 0.08, f'Hit Percentage: {hit_fraction * 100:.1f}%',
             ha='left', va='top', fontsize=12)
    plt.plot([ray_origin[0]], ray_origin[1], marker='o', color='red', fillstyle='none', label='Ray Origin',
             ms=10, zorder=10)
    plt.plot(ray_hits[:, 0], ray_hits[:, 1], marker='o', ls='None', alpha=0.4, zorder=0)
    plt.legend(framealpha=1)
    fig.tight_layout()


def plot_hit_fractions_vs_z(ray_z_origins, hit_fractions):
    """
    Plot the fraction of rays that hit the detector vs the z origin of the rays
    :param ray_z_origins: Array of z origins of the rays
    :param hit_fractions: Array of fractions of rays that hit the detector
    :return:
    """
    fig, ax = plt.subplots()
    plt.title('Fraction of Rays that Hit Detector vs Ray Origin')
    plt.xlabel('Ray Origin z (cm)')
    plt.ylabel('Fraction of Rays that Hit Detector')
    plt.axhline(0.0, color='black', ls='-')
    plt.axvline(0.0, color='black', ls='-')
    plt.plot(ray_z_origins, hit_fractions, marker='o', ls='None')
    fig.tight_layout()


if __name__ == '__main__':
    main()
