#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 04 21:50 2024
Created in PyCharm
Created as saclay_micromegas/epic_tracking_sim

@author: Dylan Neff, dn277127
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit as cf

from MultipleScattererBField import MultipleScattererBField, calculate_free_trajectory


def main():
    scatter_eic_test()
    print('donzo')


def scatter_eic_test():
    # Define simulation parameters
    energy = 2.  # GeV
    num_particles = 1
    initial_radius = 0  # Starting at the origin
    magnetic_field = 5  # 1 T magnetic field
    charge = 1 # Charge of the particle

    # Configuration for the material
    configuration = {
        'scattering_objects': [
            {
                'type': 'vacuum',
                'thickness': 10,  # cm to reach 100 cm total (50cm kapton + 50cm vacuum)
                'radius': 0,
                'material': MultipleScattererBField.materials['vacuum']
            },
            {
                'type': 'kapton',
                'thickness': 0.2,  # cm
                'radius': 10,  # cm
                'material': MultipleScattererBField.materials['kapton']
            },
            {
                'type': 'vacuum',
                'thickness': 10,  # cm to reach 100 cm total (50cm kapton + 50cm vacuum)
                'radius': 10.2,
                'material': MultipleScattererBField.materials['vacuum']
            }
        ]
    }

    # Run the simulation
    simulation = MultipleScattererBField(energy, num_particles, initial_radius, configuration, magnetic_field, energy)
    simulation.charge = charge
    simulation.simulate()
    trajectory = simulation.trajectories

    plt.figure(figsize=(10, 5))

    particle_plot_index = 0
    for point in trajectory:
        x_traj, y_traj, z_traj = point['x'], point['y'], point['z']
        px_traj, py_traj, pz_traj = point['px'], point['py'], point['pz']
        rf, r0 = point['rf'], np.sqrt(x_traj**2 + y_traj**2)
        rs_plot = np.linspace(r0, rf, 100)
        xs_plot, ys_plot = [], []
        for rf_i in rs_plot:
            free_res = calculate_free_trajectory(x_traj, y_traj, z_traj, px_traj, py_traj, pz_traj, rf_i,
                                                 charge, magnetic_field)
            xs_plot.append(free_res[0][particle_plot_index])
            ys_plot.append(free_res[1][particle_plot_index])

        plt.plot(xs_plot, ys_plot)
    plt.title('Particle Trajectory in 1T Magnetic Field')
    plt.xlabel('X Position (cm)')
    plt.ylabel('Y Position (cm)')

    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
