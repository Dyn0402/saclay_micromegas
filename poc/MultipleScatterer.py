#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 04 22:03 2024
Created in PyCharm
Created as saclay_micromegas/MultipleScatterer

@author: Dylan Neff, dn277127
"""


import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class MultipleScatterer:
    # Class variable: Materials database
    materials = {
        'air': {'radiation_length': 36.62, 'density': 1.205 / 1000},
        'mylar': {'radiation_length': 39.95, 'density': 1.40},
        'copper': {'radiation_length': 137.3, 'density': 8.96},
        'kapton': {'radiation_length': 85.5, 'density': 1.42},
        'polypropylene': {'radiation_length': 78.5, 'density': 2.041},
        'argon': {'radiation_length': 19.55, 'density': 1.519 / 1000},
        'aluminum': {'radiation_length': 24.01, 'density': 2.699},
    }

    def __init__(self, energy, num_particles, initial_radius, configuration):
        self.energy = energy
        self.num_particles = num_particles
        self.init_r = initial_radius  # Initial radius in cm
        self.configuration = configuration
        self.particles = self.gen_beam_particles()

    def gen_beam_particles(self):
        # Generate gaussian random x and y displacements from init_r
        x = np.random.normal(loc=0, scale=self.init_r, size=self.num_particles)
        y = np.random.normal(loc=0, scale=self.init_r, size=self.num_particles)
        angles_x, angles_y = np.zeros(self.num_particles), np.zeros(self.num_particles)
        particles = {'x': x, 'y': y, 'angles_x': angles_x, 'angles_y': angles_y, 'z': 0}
        return particles

    def simulate(self):
        banco_particles, test_det_particles = [], []
        for obj in self.configuration['scattering_objects']:
            self.particles = self.scatter_object(obj)
            if obj['type'] == 'banco':
                banco_particles.append(deepcopy(self.particles))
            elif obj['type'] == 'test_det':
                test_det_particles.append(deepcopy(self.particles))
        # Additional analysis and plotting can be added here
        return banco_particles, test_det_particles

    def scatter_object(self, scattering_object):
        if 'material' in scattering_object:
            angles, dxys = self.simulate_scattering(scattering_object['material'], scattering_object['thickness'])
        else:
            angles, dxys = np.zeros((2, self.num_particles)), np.zeros((2, self.num_particles))

        self.particles['x'] += dxys[0] + np.tan(self.particles['angles_x']) * scattering_object['thickness']
        self.particles['y'] += dxys[1] + np.tan(self.particles['angles_y']) * scattering_object['thickness']
        self.particles['z'] += scattering_object['thickness']
        self.particles['angles_x'] += angles[0]
        self.particles['angles_y'] += angles[1]
        return self.particles

    def simulate_scattering(self, material, thickness):
        # Simplified scattering simulation based on material properties
        material_data = self.materials[material]
        theta_0 = self.calc_scattering_angle(material_data, thickness)
        angles = np.random.normal(0, theta_0, size=(2, self.num_particles))
        displacements = np.random.normal(0, thickness * theta_0, size=(2, self.num_particles))
        return angles, displacements

    def calc_scattering_angle(self, material_data, thickness):
        radiation_length = material_data['radiation_length']
        # Simplified formula for the Highland scattering angle
        return (13.6 / self.energy) * np.sqrt(thickness / radiation_length)

    def plot_configuration(self):
        # Plot configuration similar to plot_config_diagram()
        fig, ax = plt.subplots()
        z = 0
        color_map = {'banco': 'red', 'test_det': 'green'}
        for material in self.configuration['scattering_objects']:
            dz = material['thickness']
            if material['type'] != 'air':
                ax.add_patch(Rectangle((z, 0.35), dz, 0.3, color=color_map.get(material['type'], 'blue')))
            z += dz
        ax.set_title(f'{self.configuration["name"]} Configuration')
        ax.set_xlabel('Distance (cm)')
        plt.show()
