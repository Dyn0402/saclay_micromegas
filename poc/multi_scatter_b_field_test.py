#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 05 18:14 2024
Created in PyCharm
Created as saclay_micromegas/multi_scatter_b_field_test

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from MultipleScattererBField import calculate_free_trajectory_xy, calculate_free_trajectory


def main():
    # plot_free_trajectory()
    plot_free_trajectory_3d()
    print('donzo')


def plot_free_trajectory_3d():
    x0, y0, z0 = 0.0, 0.0, 0.0  # initial position
    px, py, pz = 1.0, 0.0, 0.0  # initial momenta GeV/c
    rf = 100  # final radius (in centimeters)
    q = 1.0  # charge in units of electron charge
    B = 2  # magnetic field strength (in Tesla)
    plot_trajectory_3d(x0, y0, z0, px, py, pz, rf, q, B)


def plot_free_trajectory():
    x0, y0 = 0.0, 0.0  # initial position
    px, py = 1.0, 0.  # initial momenta (in arbitrary units)
    rf = 1.0  # final radius (in meters)
    q = 1.0  # charge (in arbitrary units)
    B = 3  # magnetic field strength (in Tesla)
    plot_trajectory(x0, y0, px, py, rf, q, B)


def plot_trajectory(x0, y0, px, py, rf, q, B, n_points=100):
    """
    Plot the trajectory of a charged particle in a magnetic field.

    Parameters:
    x0, y0 : float -> Initial position
    px, py : float -> Initial transverse momenta
    q : float -> Charge of the particle
    B : float -> Magnetic field strength in Tesla
    n_points : int -> Number of points to calculate and plot
    """
    r0 = np.sqrt(x0 ** 2 + y0 ** 2)
    x_plt_vals, y_plt_vals = [x0], [y0]
    for rf in np.linspace(r0, rf, n_points):
        result = calculate_free_trajectory_xy(x0, y0, px, py, rf, q, B)
        print(f'rf: {rf}, result: {result}')
        if result is not None:
            xf, yf, _, _, _ = result
            x_plt_vals.append(xf)
            y_plt_vals.append(yf)

    plt.figure(figsize=(6, 6))
    plt.plot(x_plt_vals, y_plt_vals, label='Trajectory')
    # Plot the final radius circle
    circle = plt.Circle((0, 0), rf, color='black', fill=False, linestyle='dashed', label='Final Radius')
    plt.gca().add_artist(circle)

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Particle Trajectory in Magnetic Field')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def plot_trajectory_3d(x0, y0, z0, px, py, pz, rf, q, B, n_points=100):
    """
    Plot the trajectory of a charged particle in a magnetic field.

    Parameters:
    x0, y0, z0 : float -> Initial position
    px, py, pz : float -> Initial momenta
    q : float -> Charge of the particle
    B : float -> Magnetic field strength in Tesla
    n_points : int -> Number of points to calculate and plot
    """
    r0 = np.sqrt(x0 ** 2 + y0 ** 2)
    x_plt_vals, y_plt_vals, z_plt_vals = [x0], [y0], [z0]
    x0, y0, z0, px, py, pz = [np.array([x]) for x in [x0, y0, z0, px, py, pz]]
    print(f'x0: {x0}, y0: {y0}, z0: {z0}, px: {px}, py: {py}, pz: {pz}')
    print(f'rf: {rf}, q: {q}, B: {B}')
    for rf in np.linspace(r0, rf, n_points):
        rf = np.array([rf])
        result = calculate_free_trajectory(x0, y0, z0, px, py, pz, rf, q, B)
        print(f'rf: {rf}, result: {result}')
        if result is not None:
            xf, yf, zf, px, py, pz, _ = result
            x0, y0, z0, px, py, pz = xf, yf, zf, px, py, pz
            x_plt_vals.append(xf[0])
            y_plt_vals.append(yf[0])
            z_plt_vals.append(zf)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_plt_vals, y_plt_vals, label='Trajectory')
    # Plot the final radius circle
    circle = plt.Circle((0, 0), rf, color='black', fill=False, linestyle='dashed', label='Final Radius')
    ax.set_xlim(-rf, rf)
    ax.set_ylim(-rf, rf)
    plt.gca().add_artist(circle)

    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title('Particle Trajectory in Magnetic Field')
    ax.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
