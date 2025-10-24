#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 22 19:32 2024
Created in PyCharm
Created as saclay_micromegas/multiple_scattering

@author: Dylan Neff, dn277127
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit as cf

materials = {
    # 'material_name', {'radiation_length': g/cm^2, 'density': g/cm^3}
    'air': {'radiation_length': 36.62, 'density': 1.205 / 1000},
    'mylar': {'radiation_length': 39.95, 'density': 1.40},
    'copper': {'radiation_length': 137.3, 'density': 8.96},
    'kapton': {'radiation_length': 85.5, 'density': 1.42},
    'polypropylene': {'radiation_length': 78.5, 'density': 2.041},
    'argon': {'radiation_length': 19.55, 'density': 1.519 / 1000},
    'aluminum': {'radiation_length': 24.01, 'density': 2.699},
    # 'pcb': 18.8,
}

test_detector = [
    # {'material': material_name, 'thickness': thickness in cm},
    {'material': 'mylar', 'thickness': 5 / 10000},  # cm from microns
    {'material': 'steel', 'thickness': 18 * 0.4 / 10000},  # cm from microns, 40% filled
    {'material': 'steel', 'thickness': 18 * 0.4 / 10000},  # cm from microns, 40% filled twice?
    {'material': 'argon', 'thickness': 0.5},  # cm
    {'material': 'kapton', 'thickness': 50 / 10000},  # cm from microns
    {'material': 'copper', 'thickness': 15 / 10000},  # cm from microns
    {'material': 'kapton', 'thickness': 50 / 10000},  # cm from microns
    {'material': 'copper', 'thickness': 15 / 10000},  # cm from microns
    {'material': 'mylar', 'thickness': 5 / 10000},  # cm from microns
    # {'material': 'pcb', 'thickness': 0.5},  # cm  ???
]

test_detector_x0 = [
    # {'x0': fraction of radiation length},
    {'x0': 0.003, 'thickness': 150 / 10000},  # Ballpark
]

banco_arm_x0 = [
    {'x0': 0.003, 'thickness': 150 / 10000},  # Guess, same as test
]

test_det_air_gap = [
    {'material': 'air', 'thickness': 10},  # cm
]

banco_arm_air_gap = [
    {'material': 'air', 'thickness': 13},  # cm
]

init_air_gap = [
    {'material': 'air', 'thickness': 5},  # cm
]


def main():
    # initial_test()
    configuration_optimization()
    # cosmic_muons()
    print('donzo')


def cosmic_muons():
    incident_energy = 2000  # MeV
    num_particles = 10000
    init_r = 200 / 10000  # cm from microns

    configurations = [
        # {
        #     'name': r'Cosmic Muons One Banco Arm 10$\mu$m Resolution',
        #     'banco_res': 10 / 10000,  # um to cm
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        # {
        #     'name': r'Cosmic Muons One Banco Arm 5$\mu$m Resolution',
        #     'banco_res': 5 / 10000,  # um to cm
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        {
            'name': r'Cosmic Muons Two Banco Arms',
            'banco_res': 5 / 10000,  # um to cm
            'scattering_objects':
                [
                    {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
                    {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
                    {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
                    {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
                    {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
                    {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
                    {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                ]
        },
        # {
        #     'name': r'Cosmic Muons Two Banco Arms Tracking Resolution Contribution',
        #     'banco_res': 5 / 10000,  # um to cm
        #     'scatter': False,
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        # {
        #     'name': r'Cosmic Muons Two Banco Arms Scattering Contribution',
        #     'scatter': True,
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
    ]
    marker_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    marker = iter(marker_shapes)

    fig, ax = plt.subplots(dpi=144)
    for configuration in configurations:
        # Generate gaussian random x and y displacements from init_r
        particles = gen_beam_particles(num_particles, init_r)
        banco_particles, test_det_particles = [], []
        for scattering_object in configuration['scattering_objects']:
            if 'scatter' in configuration and configuration['scatter'] is False:
                scattering_object['x0'] = 0
            particles = scatter_object(incident_energy, scattering_object, particles)
            if scattering_object['type'] == 'banco':
                banco_particles.append(deepcopy(particles))
            elif scattering_object['type'] == 'test_det':
                test_det_particles.append(deepcopy(particles))

        banco_res = configuration['banco_res'] if 'banco_res' in configuration else 0
        banco_tracks = calc_banco_tracks(*deepcopy(banco_particles)[:2], resolution=banco_res)
        banco_tracks2 = None
        if len(banco_particles) == 4:
            banco_tracks2 = calc_banco_tracks(*deepcopy(banco_particles)[2:], resolution=banco_res)
        if 'banco_max_angle' in configuration:
            plot_banco_angles(banco_tracks)
            straight_mask = select_straight_tracks(banco_tracks, configuration['banco_max_angle'])
        else:
            straight_mask = None
        banco_track_sets = [banco_tracks] if banco_tracks2 is None else [banco_tracks, banco_tracks2]
        for banco_tracks_i_num, banco_tracks_i in enumerate(banco_track_sets):
            distances, sigmas, sigma_errs = [], [], []
            for det_i, det_particles in enumerate(test_det_particles):
                x_res, y_res = calc_det_residuals(det_particles, banco_tracks_i)
                if straight_mask is not None:
                    x_res, y_res = x_res[straight_mask], y_res[straight_mask]
                popt_x, pcov_x, x_hist, x_bin_centers = fit_residuals(x_res * 10)
                sigma_x, sigma_x_err = abs(popt_x[2]), np.sqrt(np.diag(pcov_x)[2])
                distances.append(det_particles['z'])
                sigmas.append(sigma_x * 1000)
                sigma_errs.append(sigma_x_err)
                # plot_res_xy_hists(x_res, y_res, f'Detector {det_i + 1} Residuals')
            lab = configuration["name"]
            lab = lab + f' Arm {banco_tracks_i_num + 1}' if len(banco_track_sets) > 1 else lab
            ax.errorbar(distances, sigmas, yerr=sigma_errs, alpha=0.8, marker=next(marker), label=lab, markersize=5)
            plot_config_diagram(configuration)
    ax.set_title('Detector Resolution vs Distance')
    ax.set_xlabel('Distance from Beam Exit (cm)')
    ax.set_ylabel(r'Resolution ($\mu$m)')
    ax.grid()
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend()
    fig.tight_layout()
    plt.show()


def configuration_optimization():
    # incident_energy = 880  # MeV
    incident_energy = 880  # MeV
    num_particles = 10000
    init_r = 200 / 10000  # cm from microns

    configurations = [
        # {
        #     'name': 'banco_first',
        #     'banco_res': 0,
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        {
            'name': r'2023 Test Beam Config 10$\mu$m Resolution 150 GeV Muons',
            'banco_res': 10 / 10000,  # um to cm
            'incident_energy': 150e3,  # MeV
            'scattering_objects':
                [
                    {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
                    {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                ]
        },
        {
            'name': r'Sandwiched 10$\mu$m Resolution 150 GeV Muons',
            'banco_res': 10 / 10000,  # um to cm
            'incident_energy': 150e3,  # MeV
            'scattering_objects':
                [
                    {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
                ]
        },
        {
            'name': r'Sandwiched 150$\mu$m (uRW) Resolution 150 GeV Muons',
            'banco_res': 150 / 10000,  # um to cm
            'incident_energy': 150e3,  # MeV
            'scattering_objects':
                [
                    {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
                ]
        },
        {
            'name': r'2023 Test Beam Config 10$\mu$m Resolution 80 GeV Muons',
            'banco_res': 10 / 10000,  # um to cm
            'incident_energy': 80e3,  # MeV
            'scattering_objects':
                [
                    {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
                    {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                ]
        },
        {
            'name': r'2023 Test Beam Config 10$\mu$m Resolution 10 GeV Muons',
            'banco_res': 10 / 10000,  # um to cm
            'incident_energy': 10e3,  # MeV
            'scattering_objects':
                [
                    {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
                    {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                ]
        },
        {
            'name': r'2025 Test Beam Config 10$\mu$m Resolution 80 GeV Muons',
            'banco_res': 10 / 10000,  # um to cm
            'incident_energy': 80e3,  # MeV
            'scattering_objects':
                [
                    {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
                    {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
                    {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                    {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
                    {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
                ]
        },
        # {
        #     'name': r'2023 Test Beam Config 10$\mu$m Resolution 5 GeV Muons',
        #     'banco_res': 10 / 10000,  # um to cm
        #     'incident_energy': 5e3,  # MeV
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        # {
        #     'name': r'2023 Test Beam Config 10$\mu$m Resolution 1 GeV Muons',
        #     'banco_res': 10 / 10000,  # um to cm
        #     'incident_energy': 1e3,  # MeV
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        # {
        #     'name': r'2023 Test Beam Config 10$\mu$m Resolution 880 MeV Electrons',
        #     'banco_res': 10 / 10000,  # um to cm
        #     'incident_energy': 880,  # MeV
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        # {
        #     'name': r'2023 Test Beam 5$\mu$m Resolution',
        #     'banco_res': 5 / 10000,  # um to cm
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        # {
        #     'name': r'First Test Beam 5$\mu$m Resolution',
        #     'banco_res': 5 / 10000,  # um to cm
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        # {
        #     'name': '2023 Test Beam Tracking Resolution Contribution',
        #     'banco_res': 10 / 10000,  # um to cm
        #     'scatter': False,  # Turn off scattering if False
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        # {
        #     'name': '2023 Test Beam Scattering Contribution',
        #     'scatter': True,  # Turn off scattering if False
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        # {
        #     'name': 'banco_edges',
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #         ]
        # },
        # {
        #     'name': 'banco_edges_straight_0.001_degree',
        #     'banco_max_angle': 0.001,  # Degrees
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #         ]
        # },
        # {
        #     'name': 'banco_edges_straight_0.0002_degree',
        #     'banco_max_angle': 0.0002,  # Degrees
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #         ]
        # },
        # {
        #     'name': r'Next Test Beam 10$\mu$m Resolution',
        #     'banco_res': 10 / 10000,  # um to cm
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        # {
        #     'name': r'Next Test Beam 5$\mu$m Resolution',
        #     'banco_res': 5 / 10000,  # um to cm
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },

        # {
        #     'name': r'2024 Test Beam Single Arm',
        #     # 'name': r'2024 Test Beam Single Arm 10 $\mu$m Tracking Resolution',
        #     'banco_res': 10 / 10000,  # um to cm
        #     'incident_energy': 880,  # MeV
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        #
        # {
        #     'name': r'2024 Test Beam Single Arm Muons',
        #     # 'name': r'2024 Test Beam Single Arm 10 $\mu$m Tracking Resolution',
        #     'banco_res': 10 / 10000,  # um to cm
        #     'incident_energy': 80e3,  # MeV
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },

        # {
        #     'name': r'2024 Test Beam Single Arm 5 $\mu$m Tracking Resolution',
        #     'banco_res': 5 / 10000,  # um to cm
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             # {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             # {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             # {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        # {
        #     'name': r'2024 Test Beam Single Arm Tracking Resolution Contribution',
        #     'banco_res': 10 / 10000,  # um to cm
        #     'scatter': False,  # Turn off scattering if False
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             # {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             # {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             # {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        # {
        #     'name': r'2024 Test Beam Single Arm Scattering Contribution',
        #     'scatter': True,  # Turn off scattering if False
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             # {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             # {'x0': 0.003, 'thickness': 10.6 / 10, 'type': 'banco_seperator'},  # Guess, same as test
        #             # {'x0': 0.0001, 'thickness': 5 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 7, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },

        # {
        #     'name': 'banco_edge_sandwich',
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
        # {
        #     'name': 'banco_second_sandwich',
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 10, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # }
        # {
        #     'name': r'2023 Test Beam 5$\mu$m Resolution',
        #     'banco_res': 5 / 10000,  # um to cm
        #     'scattering_objects':
        #         [
        #             {'material': 'air', 'thickness': 5, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 13, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 150 / 10000, 'type': 'banco'},  # Guess, same as test
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #             {'material': 'air', 'thickness': 8, 'type': 'air'},  # cm
        #             {'x0': 0.003, 'thickness': 1.58e-2, 'type': 'test_det'},  # Ballpark
        #         ]
        # },
    ]
    marker_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    marker = iter(marker_shapes)

    fig, ax = plt.subplots(dpi=144)
    for configuration in configurations:
        # Generate gaussian random x and y displacements from init_r
        particles = gen_beam_particles(num_particles, init_r)
        banco_particles, test_det_particles = [], []
        for scattering_object in configuration['scattering_objects']:
            if 'scatter' in configuration and configuration['scatter'] is False:
                scattering_object['x0'] = 0
            energy = configuration['incident_energy'] if 'incident_energy' in configuration else incident_energy
            particles = scatter_object(energy, scattering_object, particles)
            if scattering_object['type'] == 'banco':
                banco_particles.append(deepcopy(particles))
            elif scattering_object['type'] == 'test_det':
                test_det_particles.append(deepcopy(particles))

        banco_res = configuration['banco_res'] if 'banco_res' in configuration else 0
        banco_tracks = calc_banco_tracks(*deepcopy(banco_particles)[:2], resolution=banco_res)
        banco_tracks2 = None
        if len(banco_particles) == 4:
            banco_tracks2 = calc_banco_tracks(*deepcopy(banco_particles)[2:], resolution=banco_res)
        if 'banco_max_angle' in configuration:
            plot_banco_angles(banco_tracks)
            straight_mask = select_straight_tracks(banco_tracks, configuration['banco_max_angle'])
        else:
            straight_mask = None
        banco_track_sets = [banco_tracks] if banco_tracks2 is None else [banco_tracks, banco_tracks2]
        for banco_tracks_i_num, banco_tracks_i in enumerate(banco_track_sets):
            distances, sigmas, sigma_errs = [], [], []
            for det_i, det_particles in enumerate(test_det_particles):
                x_res, y_res = calc_det_residuals(det_particles, banco_tracks_i)
                if straight_mask is not None:
                    x_res, y_res = x_res[straight_mask], y_res[straight_mask]
                popt_x, pcov_x, x_hist, x_bin_centers = fit_residuals(x_res * 10)
                sigma_x, sigma_x_err = abs(popt_x[2]), np.sqrt(np.diag(pcov_x)[2])
                distances.append(det_particles['z'])
                sigmas.append(sigma_x * 1000)
                sigma_errs.append(sigma_x_err)
                # plot_res_xy_hists(x_res, y_res, f'Detector {det_i + 1} Residuals')
            lab = configuration["name"]
            lab = lab + f' Arm {banco_tracks_i_num + 1}' if len(banco_track_sets) > 1 else lab
            ax.errorbar(distances, sigmas, yerr=sigma_errs, alpha=0.8, marker=next(marker), label=lab, markersize=5)
            plot_config_diagram(configuration)
    ax.set_title('Detector Resolution vs Distance')
    ax.set_xlabel('Distance from Beam Exit (cm)')
    ax.set_ylabel(r'Resolution ($\mu$m)')
    ax.grid()
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend()
    fig.tight_layout()
    plt.show()


def initial_test():
    n_test_detectors = 3
    incident_energy = 880  # MeV
    num_particles = 1000
    init_r = 200 / 10000  # cm from microns

    print(f'Air gap x0: {calc_x0_percent(test_det_air_gap[0]["thickness"], materials["air"]["radiation_length"], materials["air"]["density"]) * 100:.2f}%')

    # Generate gaussian random x and y displacements from init_r
    particles = gen_beam_particles(num_particles, init_r)
    plot_particles(particles, 'Initial')

    print(f'Particle 0: {particles["x"][0]}, {particles["y"][0]}')
    particles = scatter_object(incident_energy, init_air_gap, particles)

    banco_arm1_particles = deepcopy(particles)
    particles = scatter_object(incident_energy, banco_arm_x0, particles)
    particles = scatter_object(incident_energy, banco_arm_air_gap, particles)
    banco_arm2_particles = deepcopy(particles)
    particles = scatter_object(incident_energy, banco_arm_x0, particles)

    det_particles = []
    for det_i in range(n_test_detectors):
        particles = scatter_object(incident_energy, test_det_air_gap, particles)
        # plot_particles(particles, f'Hitting Detector #{det_i + 1}')
        det_particles.append(deepcopy(particles))
        print(f'Particle 0: {particles["x"][0]}, {particles["y"][0]}')
        particles = scatter_object(incident_energy, test_detector_x0, particles)

    banco_tracks = calc_banco_tracks(banco_arm1_particles, banco_arm2_particles)

    for det_i in range(n_test_detectors):
        print(f'Det {det_i} Particle 0 pos: {det_particles[det_i]["x"][0]}, {det_particles[det_i]["y"][0]}')
        x_res, y_res = calc_det_residuals(det_particles[det_i], banco_tracks)
        print(f'Det {det_i} Particle 0 Residual: {x_res[0]}, {y_res[0]}')
        # plot_res_r(x_res, y_res, f'Detector {det_i + 1} Residuals')
        plot_res_xy_hists(x_res, y_res, f'Detector {det_i + 1} Residuals')

    plt.show()


def gen_beam_particles(num_particles, init_r):
    # Generate gaussian random x and y displacements from init_r
    x = np.random.normal(loc=0, scale=init_r, size=num_particles)
    y = np.random.normal(loc=0, scale=init_r, size=num_particles)
    angles_x, angles_y = np.zeros(num_particles), np.zeros(num_particles)
    particles = {'x': x, 'y': y, 'angles_x': angles_x, 'angles_y': angles_y, 'z': 0}
    return particles


def scatter_object(energy, scattering_object, particles):
    if isinstance(scattering_object, dict):
        scattering_object = [scattering_object]
    for material in scattering_object:
        if 'x0' in material and material['x0'] is not None:
            if material['x0'] <= 0:
                angles, dxys = np.zeros((2, len(particles['x']))), np.zeros((2, len(particles['x'])))
            else:
                angles, dxys = simulate_electron_scattering(energy, material['x0'], material['thickness'], None,
                                                            len(particles['x']))
        else:
            angles, dxys = simulate_electron_scattering(energy, None, material['thickness'], material['material'],
                                                        len(particles['x']))
        particles['x'] += dxys[0] + np.tan(particles['angles_x']) * material['thickness']
        particles['y'] += dxys[1] + np.tan(particles['angles_y']) * material['thickness']
        particles['z'] += material['thickness']
        particles['angles_x'] += angles[0]
        particles['angles_y'] += angles[1]
    return particles


def simulate_electron_scattering(energy, x0=None, thickness=None, material=None, num_particles=1000):
    # Constants
    m = 0.511  # MeV/c^2
    e = -1.0  # Elementary charge
    p = calc_momentum(energy, m)
    b = calc_beta(energy, m)

    # Convert thickness to radiation lengths
    if x0 is None:
        # Material properties (e.g., aluminum)
        radiation_length = materials[material]['radiation_length']  # Radiation length in cm
        density = materials[material]['density']  # Density in g/cm^3

        t_rad_lens = calc_x0_percent(thickness, radiation_length, density)
    else:
        t_rad_lens = x0

    # Calculate standard deviation of scattering angle using Highland formula
    theta_0 = (13.6 / (b * p)) * abs(e) * np.sqrt(t_rad_lens) * (1 + 0.038 * np.log(t_rad_lens * e ** 2 / b ** 2))

    # Simulate scattering angles
    z1_x = np.random.normal(loc=0, scale=1, size=num_particles)
    z2_x = np.random.normal(loc=0, scale=1, size=num_particles)
    z1_y = np.random.normal(loc=0, scale=1, size=num_particles)
    z2_y = np.random.normal(loc=0, scale=1, size=num_particles)

    t, o = thickness, theta_0

    x_plane = t * o * (z1_x / np.sqrt(12) + z2_x / 2)
    y_plane = t * o * (z1_y / np.sqrt(12) + z2_y / 2)
    theta_x_plane = z2_x * o
    theta_y_plane = z2_y * o

    # Simulate energy loss (not included in this simplified model)

    # Return scattering angles and displacements
    return np.array([theta_x_plane, theta_y_plane]), np.array([x_plane, y_plane])


def plot_particles(particles, title=''):
    fig, ax = plt.subplots()
    ax.scatter(particles['x'], particles['y'])
    ax.set_title(title)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')


def plot_res_r(x_res, y_res, title=''):
    r_res = np.sqrt(x_res ** 2 + y_res ** 2)
    fig, ax = plt.subplots()
    ax.hist(r_res, bins=50)
    ax.set_title(title)
    ax.set_xlabel('r residual (cm)')
    ax.set_ylabel('Tracks')


def plot_res_xy_hists(x_res, y_res, title=''):
    x_res *= 10
    y_res *= 10
    popt_x, pcov_x, x_hist, x_bin_centers = fit_residuals(x_res)
    popt_y, pcov_y, y_hist, y_bin_centers = fit_residuals(y_res)
    # x_hist, x_bins = np.histogram(x_res, bins=50)
    # x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
    # y_hist, y_bins = np.histogram(y_res, bins=50)
    # y_bin_centers = (y_bins[:-1] + y_bins[1:]) / 2
    # popt_x, pcov_x = cf(gauss, x_bin_centers, x_hist, p0=[1, 1, 1])
    # popt_y, pcov_y = cf(gauss, y_bin_centers, y_hist, p0=[1, 1, 1])
    fig, ax = plt.subplots()
    # ax.hist(x_res, bins=50, alpha=0.5, color='r', label='x')
    x_bin_width, y_bin_width = x_bin_centers[1] - x_bin_centers[0], y_bin_centers[1] - y_bin_centers[0]
    ax.bar(x_bin_centers, x_hist, width=x_bin_width, alpha=0.5, color='r', align='center', label='x')
    x_plt_points = np.linspace(min(x_res), max(x_res), 100)
    ax.plot(x_plt_points, gauss(x_plt_points, *popt_x), 'r-')
    # ax.hist(y_res, bins=50, alpha=0.5, color='g', label='y')
    ax.bar(y_bin_centers, y_hist, width=y_bin_width, alpha=0.5, color='g', align='center', label='y')
    ax.plot(x_plt_points, gauss(x_plt_points, *popt_y), 'g-')
    ax.set_title(title)
    ax.set_xlabel('Residual (mm)')
    ax.set_ylabel('Number of Tracks')
    ax.annotate(fr'$\sigma_x$: {abs(popt_x[2]):.2f} mm', xy=(0.05, 0.9), xycoords='axes fraction', color='r')
    ax.annotate(fr'$\sigma_y$: {abs(popt_y[2]):.2f} mm', xy=(0.05, 0.85), xycoords='axes fraction', color='g')
    ax.legend()


def plot_banco_angles(banco_tracks):
    x0, y0, z0, x_slope, y_slope, dz = banco_tracks
    angles = calc_angles(x_slope, y_slope, dz)
    fig, ax = plt.subplots()
    ax.hist(angles, bins=50)


def plot_config_diagram(configuration):
    """
    Plot a diagram of the configuration, with the beam exit at x=0 adn each material represented by a vertical box
    with the x width corresponding to the thickness. Make air white (skip it), banco red, and test_det green.
    :param configuration:
    :return:
    """
    fig, ax = plt.subplots()
    z = 0
    color_map = {'banco': 'red', 'banco_seperator': 'black', 'test_det': 'green'}
    name_map = {'banco': 'Banco', 'banco_seperator': 'Banco Seperator', 'test_det': 'Test Detector'}
    print(f'Configuration: {configuration["name"]}')
    for material in configuration['scattering_objects']:
        dz = material['thickness']
        if material['type'] != 'air' and material['type'] != 'banco_seperator':
            ax.add_patch(Rectangle((z, 0.35), dz, 0.3, color=color_map[material['type']]))
        print(f'{material["type"]} at z={z} with thickness {material["thickness"]}')
        z += dz
    for material, color in color_map.items():
        if material == 'banco_seperator':
            continue
        ax.add_patch(Rectangle((0, 0), 0, 0, color=color, label=name_map[material]))
    ax.legend()
    ax.set_title(f'{configuration["name"]} Configuration')
    ax.set_xlabel('Distance from Beam Exit (cm)')
    ax.set_xlim(left=0, right=65)
    ax.set_ylim(bottom=0, top=1)
    fig.tight_layout()


def calc_banco_tracks(particles1, particles2, resolution=0):
    # Calculate tracks of particles from banco arm 1 to banco arm 2
    x0, y0, z0 = particles1['x'], particles1['y'], particles1['z']
    x1, y1, z1 = particles2['x'], particles2['y'], particles2['z']
    if resolution > 0:
        x0, y0 = np.random.normal(loc=x0, scale=resolution), np.random.normal(loc=y0, scale=resolution)
        x1, y1 = np.random.normal(loc=x1, scale=resolution), np.random.normal(loc=y1, scale=resolution)
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    x_slope, y_slope = dx / dz, dy / dz
    return x0, y0, z0, x_slope, y_slope, dz


def select_straight_tracks(banco_tracks, max_angle=0.1):
    x0, y0, z0, x_slope, y_slope, dz = banco_tracks
    angles = calc_angles(x_slope, y_slope, dz)
    return angles < max_angle


def calc_angles(x_slope, y_slope, dz):
    r_slope = np.sqrt(x_slope ** 2 + y_slope ** 2)
    angles = np.rad2deg(np.arctan(r_slope / dz))
    return angles


def calc_det_residuals(det_particles, banco_tracks):
    x0, y0, z0, x_slope, y_slope, dz = banco_tracks
    x, y = det_particles['x'], det_particles['y']
    x_res, y_res = x - (x0 + x_slope * (det_particles['z'] - z0)), y - (y0 + y_slope * (det_particles['z'] - z0))
    return x_res, y_res


def fit_residuals(residuals, bins=50):
    hist, bins = np.histogram(residuals, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    try:
        p0 = [max(hist) * 0.9, np.mean(residuals), np.std(residuals)]
        popt, pcov = cf(gauss, bin_centers, hist, p0=p0)
    except:
        print(f'Residual fit failed!')
        popt, pcov = [0, 0, 0], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    return popt, pcov, hist, bin_centers


def calc_beta(energy, mass):
    return np.sqrt(1 - (mass / energy) ** 2)


def calc_momentum(energy, mass):
    return energy * calc_beta(energy, mass)


def calc_x0_percent(thickness, radiation_length, density):
    return thickness / radiation_length * density


def gauss(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


if __name__ == '__main__':
    main()
