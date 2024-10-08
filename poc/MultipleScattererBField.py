#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 05 18:08 2024
Created in PyCharm
Created as saclay_micromegas/MultipleScattererBField

@author: Dylan Neff, dn277127
"""


import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class MultipleScattererBField:
    materials = {
        'vacuum': {'radiation_length': 0, 'density': 0},
        'air': {'radiation_length': 36.62, 'density': 1.205 / 1000},
        'mylar': {'radiation_length': 39.95, 'density': 1.40},
        'copper': {'radiation_length': 137.3, 'density': 8.96},
        'kapton': {'radiation_length': 85.5, 'density': 1.42},
        'polypropylene': {'radiation_length': 78.5, 'density': 2.041},
        'argon': {'radiation_length': 19.55, 'density': 1.519 / 1000},
        'aluminum': {'radiation_length': 24.01, 'density': 2.699},
    }

    def __init__(self, energy, num_particles, initial_radius, configuration, magnetic_field=0, pt=0, mass=1):
        self.energy = energy  # Particle energy in GeV
        self.num_particles = num_particles
        self.init_r = initial_radius  # Initial radius in cm
        self.configuration = configuration
        self.magnetic_field = magnetic_field  # Constant B-field in Tesla
        self.pt = pt  # Transverse momentum in GeV/c
        self.mass = mass  # Particle mass in GeV/c^2
        self.charge = 1  # Particle charge in multiples of elementary charge
        self.trajectories = []
        self.particles = self.gen_collision_particles()

    def gen_collision_particles(self):
        # Start particles at origin with distribution of transverse momenta
        x = np.zeros(self.num_particles)
        y = np.zeros(self.num_particles)
        z = np.zeros(self.num_particles)
        pz = np.full(self.num_particles, np.sqrt(self.energy ** 2 - self.pt ** 2))  # Calculate pz from pt and energy
        px = np.random.uniform(-self.pt, self.pt, self.num_particles)
        py = np.sqrt(self.pt ** 2 - px ** 2)  # Calculate py from px and pt, for each particle pt^2 = px^2 + py^2
        py *= np.random.choice([-1, 1], self.num_particles)  # Multiply py by random sign to get random direction
        particles = {'x': x, 'y': y, 'z': z, 'px': px, 'py': py, 'pz': pz}
        return particles

    def simulate(self):
        banco_particles, test_det_particles = [], []
        for obj in self.configuration['scattering_objects']:
            self.particles = self.scatter_object(obj)
            if obj['type'] == 'banco':
                banco_particles.append(deepcopy(self.particles))
            elif obj['type'] == 'test_det':
                test_det_particles.append(deepcopy(self.particles))
        return banco_particles, test_det_particles

    def scatter_object(self, scattering_object):
        # Calculate free trajectories through object
        x0s, y0s, z0s = self.particles['x'], self.particles['y'], self.particles['z']
        px0s, py0s, pz0s = self.particles['px'], self.particles['py'], self.particles['pz']
        t = scattering_object['thickness']
        rf = t + scattering_object['radius']
        free_trajec = calculate_free_trajectory(x0s, y0s, z0s, px0s, py0s, pz0s, rf, self.charge, self.magnetic_field)
        if free_trajec is None:
            return self.particles
        trajectory = deepcopy(self.particles)
        trajectory.update({'rf': np.full(len(x0s), rf)})
        self.trajectories.append(trajectory)
        print(scattering_object)

        xf, yf, zf, pxf, pyf, pzf, path_length = free_trajec

        if 'x0' in scattering_object:
            n_rad_lens = scattering_object['x0']
        else:
            radiation_length = scattering_object['material']['radiation_length']  # Radiation length in cm
            density = scattering_object['material']['density']  # Density in g/cm^3

            n_rad_lens = calc_x0_percent(path_length, radiation_length, density)

        print(f'Number of Radiation Length: {n_rad_lens}, path length: {path_length}')
        angles, displacements = self.simulate_scattering(n_rad_lens, path_length)
        print(f'Angles: {angles}, Displacements: {displacements}')

        # Get random orientation for displacement and angle relative to xf, yf, zf, and pxf, pyf, pzf; respectively
        pxfs, pyfs, pzfs = rotate_vector_array(pxf, pyf, pzf, angles).T
        xfs, yfs, zfs = random_displacement_on_cylinder(xf, yf, zf, displacements)

        self.particles['x'] = xfs
        self.particles['y'] = yfs
        self.particles['z'] = zfs
        self.particles['px'] = pxfs
        self.particles['py'] = pyfs
        self.particles['pz'] = pzfs

        return self.particles

    def simulate_scattering(self, n_radiation_lengths, thickness):
        if n_radiation_lengths == 0:
            angles = np.full(self.num_particles, 0)
            displacements = np.full(self.num_particles, 0)
        else:
            angles, displacements = simulate_highland_scattering(self.energy, n_radiation_lengths, thickness, self.mass,
                                                                 self.charge, self.num_particles)
            theta_0 = calc_scattering_angle(n_radiation_lengths, thickness, self.energy)
            angles = np.random.normal(0, theta_0, size=self.num_particles)
            displacements = np.random.normal(0, thickness * theta_0, size=self.num_particles)
        return angles, displacements


def calculate_free_trajectory_xy(x0, y0, px, py, r_f, q, B):
    """
    Calculate the final position and transverse momenta of a charged particle moving in a magnetic field.

    Returns the final position and momenta, or None if the particle doesn't reach the final radius.
    """
    r0 = np.sqrt(x0 ** 2 + y0 ** 2)
    pT = np.sqrt(px ** 2 + py ** 2)
    rL = relativistic_larmor_radius(pT, q, B)

    if r_f < r0 or r_f > r0 + 2 * rL:
        return None

    theta0 = np.arctan2(py, px)
    delta_theta = (r_f - r0) / rL

    xf = x0 + rL * (np.sin(theta0 + delta_theta) - np.sin(theta0))
    yf = y0 - rL * (np.cos(theta0 + delta_theta) - np.cos(theta0))

    pxf = px * np.cos(delta_theta) - py * np.sin(delta_theta)
    pyf = px * np.sin(delta_theta) + py * np.cos(delta_theta)

    path_length = rL * delta_theta

    return xf, yf, pxf, pyf, path_length


def calculate_free_trajectory(x0, y0, z0, px, py, pz, r_f, q, B):
    """
    Calculate the final position and momenta of a charged particle moving in a magnetic field.

    Returns the final position and momenta, or None if the particle doesn't reach the final radius.
    """
    omega = q * B
    # Avoid division by zero
    if np.any(omega == 0):
        raise ValueError("Omega (q * B) must not be zero.")

    r0 = np.sqrt(x0 ** 2 + y0 ** 2)
    pT = np.sqrt(px ** 2 + py ** 2)
    rL = relativistic_larmor_radius(pT, q, B) * 100  # Convert to centimeters

    theta0 = np.arctan2(py, px)
    delta_theta = (r_f - r0) / rL

    # Set delta_theta to np.nan where r_f < r0 or r_f > r0 + 2 * rL
    delta_theta[(r_f < r0) | (r_f > r0 + 2 * rL)] = np.nan

    # Calculate final x, y positions
    xf = x0 + rL * (np.sin(theta0 + delta_theta) - np.sin(theta0))
    yf = y0 - rL * (np.cos(theta0 + delta_theta) - np.cos(theta0))

    # Calculate final px, py momenta
    pxf = px * np.cos(delta_theta) - py * np.sin(delta_theta)
    pyf = px * np.sin(delta_theta) + py * np.cos(delta_theta)

    # Calculate the path length traveled
    path_length = rL * delta_theta

    # Calculate final z position (free motion in z)
    zf = z0 + (pz / pT) * path_length

    return xf, yf, zf, pxf, pyf, pz, path_length


def calc_scattering_angle(n_radiation_lengths, thickness, particle_energy):
    return (13.6 / particle_energy) * np.sqrt(thickness / n_radiation_lengths)


def calc_x0_percent(thickness, radiation_length, density):
    if radiation_length == 0:
        return 0
    return thickness / radiation_length * density


def simulate_highland_scattering(energy, num_rad_lens, thickness, mass=0.511, charge=1, num_particles=1000):
    p = calc_momentum(energy, mass)
    b = calc_beta(energy, mass)

    # Calculate standard deviation of scattering angle using Highland formula
    theta_0 = (13.6 / (b * p)) * abs(charge) * np.sqrt(num_rad_lens) * (1 + 0.038 * np.log(num_rad_lens * e ** 2 / b ** 2))

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


def calc_beta(energy, mass):
    return np.sqrt(1 - (mass / energy) ** 2)


def calc_momentum(energy, mass):
    return energy * calc_beta(energy, mass)


def rotate_vector_array(px_array, py_array, pz_array, theta_array):
    # Stack the momentum components into a 2D array (N, 3)
    p = np.vstack((px_array, py_array, pz_array)).T  # Shape (N, 3)

    # Normalize the momentum vectors
    p_norm = np.linalg.norm(p, axis=1)
    p_hat = p / p_norm[:, np.newaxis]  # Shape (N, 3)

    # Random angles for random axes
    phi = np.random.uniform(0, 2 * np.pi, size=p.shape[0])  # Shape (N,)

    # Calculate random axes for each vector
    random_vec = np.vstack((np.cos(phi), np.sin(phi), np.zeros_like(phi))).T  # Shape (N, 3)
    random_axis = np.cross(p_hat, random_vec)

    # Normalize the random axes
    random_axis_norm = np.linalg.norm(random_axis, axis=1)
    random_axis = random_axis / random_axis_norm[:, np.newaxis]  # Shape (N, 3)

    # Calculate the cosine and sine of the rotation angles
    cos_theta = np.cos(theta_array)  # Shape (N,)
    sin_theta = np.sin(theta_array)  # Shape (N,)

    # Create a rotation matrix for each vector
    identity = np.eye(3)
    rotation_matrices = []

    for axis in random_axis:
        # Skew-symmetric matrix for cross product
        axis_cross = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        # Rodrigues' rotation matrix
        rotation_matrix = (cos_theta[:, np.newaxis] * identity +
                           sin_theta[:, np.newaxis] * axis_cross +
                           (1 - cos_theta[:, np.newaxis]) * np.outer(axis, axis))
        rotation_matrices.append(rotation_matrix)

    # Stack all rotation matrices and rotate the vectors
    p_rotated = np.array([np.dot(rotation_matrices[i], p[i]) for i in range(p.shape[0])])  # Shape (N, 3)

    return p_rotated


def random_displacement_on_cylinder(x, y, z, d):
    # Calculate the radius of the cylinder
    r = np.sqrt(x ** 2 + y ** 2)

    # Handle the case where points are at the axis of the cylinder
    if np.any(r == 0):
        raise ValueError("Some points are at the axis of the cylinder. Cannot displace those points.")

    # Calculate the current angle in the xy-plane
    theta = np.arctan2(y, x)

    # Generate random angles for displacements
    displacement_angles = np.random.uniform(0, 2 * np.pi, size=x.shape)

    # Calculate the new angles
    new_theta = theta + (displacement_angles * d / r)  # d/r gives the arc length to angle

    # Calculate the new x, y coordinates on the surface of the cylinder
    new_x = r * np.cos(new_theta)
    new_y = r * np.sin(new_theta)

    # The z-coordinates remain the same
    new_z = z

    return new_x, new_y, new_z


def relativistic_larmor_radius(p_T, q, B):
    """
    Calculate the relativistic Larmor radius.

    Parameters:
    p_T (float): Transverse momentum in GeV/c.
    q (float): Charge in multiples of elementary charge.
    B (float): Magnetic field strength in Tesla.

    Returns:
    float: The relativistic Larmor radius in meters.
    """
    # Constants
    c = 2.9979e8  # Speed of light in m/s
    e = 1.602e-19  # Elementary charge in Coulombs (not used in the formula since q is in multiples)
    p_T = p_T * 1e9 / c * e  # Convert GeV/c to kg·m/s

    q = q * e  # Convert multiples of elementary charge to Coulombs

    # Calculate the Larmor radius in meters
    r_L = p_T / (q * B)

    return r_L
