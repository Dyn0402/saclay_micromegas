#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 07 18:41 2024
Created in PyCharm
Created as saclay_micromegas/plot_eic_visualization

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt

from MultipleScattererBField import relativistic_larmor_radius


def main():
    plot_test()
    print('donzo')


def plot_test():
    # Define particle and simulation parameters
    p_T = 0.4  # Transverse momentum in GeV/c
    q = 1.0  # Charge in multiples of the elementary charge
    B = 2.0  # Magnetic field strength in Tesla

    # Calculate the Larmor radius
    r_L = relativistic_larmor_radius(p_T, q, B) * 100  # Convert to cm
    print(f"Larmor Radius: {r_L:.2e} cm")

    # Define rings as a list of dictionaries
    rings = [
        {'r': 3.2, 'color': 'orange'},
        {'r': 3.2, 'color': 'orange'},
        {'r': 4.8, 'color': 'orange'},
        {'r': 12, 'color': 'orange'},
        {'r': 27, 'color': 'orange'},
        {'r': 42, 'color': 'orange'},
        {'r': 55, 'color': 'blue'},
        {'r': 65, 'color': 'red'},
        {'r': 72.5, 'color': 'cyan'},
        {'r': 80, 'color': 'white'}
    ]

    r_max = rings[-1]['r']

    # Plotting setup
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')

    # Plot the rings
    for ring in rings:
        circle = plt.Circle((0, 0), ring['r'], color=ring['color'], fill=False, linewidth=2)
        ax.add_artist(circle)

    # Particle starting at origin

    # Plot a red x at the origin
    # ax.plot(0, 0, 'ro', label='Origin')

    # The particle's trajectory will be shifted so it starts at (0, 0)
    theta = np.linspace(0, 2 * np.pi, 1000)  # Create an array of angles from 0 to 2pi
    x_center = r_L  # Circle's center is at (r_L, 0)
    y_center = 0

    x_trajectory = x_center + r_L * np.cos(theta)  # Shift the x-coordinates of the particle
    y_trajectory = y_center + r_L * np.sin(theta)  # y-coordinates of the particle

    # Only plot the particle's trajectory up to the radius of the last ring
    mask = (x_trajectory ** 2 + y_trajectory ** 2 <= r_max ** 2) & (y_trajectory >= 0)
    x_trajectory = x_trajectory[mask]
    y_trajectory = y_trajectory[mask]

    # Plot the particle's trajectory starting from the origin
    ax.plot(x_trajectory, y_trajectory, color='black')

    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
