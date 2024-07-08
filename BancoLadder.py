#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 07 3:37 PM 2024
Created in PyCharm
Created as saclay_micromegas/BancoLadder.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import quaternion
import matplotlib.pyplot as plt

import uproot
import awkward as ak
import vector


class BancoLadder:
    def __init__(self, name='0', config=None):
        self.name = name
        self.center = np.array([0, 0, 0])
        self.size = np.array([0, 0, 0])
        # self.orientation = np.array([0, 0, 0])  # degrees Rotation along x, y, z axes
        self.rotations = np.array([])  # [degrees, x_bit, y_bit, z_bit] Un-normalized quaterion rotations

        self.pitch_x = 26.88  # microns spacing between pixels in x direction
        self.pitch_y = 29.24  # microns spacing between pixels in y direction
        self.passive_edge_y = 29.12  # microns inactive edge on the y side of each chip
        self.passive_edge_x = 26.88  # microns inactive edge on the x side of each chip
        self.chip_space = 100 + 2 * self.passive_edge_y  # microns space between chips in y direction

        self.n_pix_y = 1024  # Number of pixels in y direction
        self.n_pix_x = 512  # Number of pixels in x direction
        self.n_chips = 5  # Number of chips in y direction

        self.active_size = np.array([self.pitch_x * self.n_pix_x,
                                     self.pitch_y * self.n_pix_y * self.n_chips + (self.n_chips - 1) * self.chip_space,
                                     30]) / 1000  # Active area of detector in mm

        if config is not None:
            self.set_from_config(config)

        self.data = None
        self.noise_pixels = None
        self.data_noise_pixels = None

        self.clusters = None
        self.cluster_triggers = None
        self.cluster_chips = None
        self.all_cluster_centroids_local_coords = None
        self.all_cluster_num_pixels = None
        self.largest_clusters = None
        self.largest_cluster_chips = None
        self.largest_cluster_centroids_local_coords = None
        self.largest_cluster_num_pix = None

        self.cluster_centroids = None

    def set_center(self, x=None, y=None, z=None):
        if x is None:
            x = self.center[0]
        if y is None:
            y = self.center[1]
        if z is None:
            z = self.center[2]
        self.center = np.array([x, y, z])

    def set_size(self, x, y, z):
        self.size = np.array([x, y, z])

    # def set_orientation(self, x=None, y=None, z=None):
    #     if x is None:
    #         x = self.orientation[0]
    #     if y is None:
    #         y = self.orientation[1]
    #     if z is None:
    #         z = self.orientation[2]
    #     self.orientation = np.array([x, y, z])

    def set_orientation(self, x=None, y=None, z=None):
        rotations = []
        if x is not None:
            rotations.append(quaternion_from_angle_axis(x, np.array([1, 0, 0])))
        if y is not None:
            rotations.append(quaternion_from_angle_axis(y, np.array([0, 1, 0])))
        if z is not None:
            rotations.append(quaternion_from_angle_axis(z, np.array([0, 0, 1])))
        self.rotations = np.array(rotations)

    def add_rotation(self, angle, axis):
        if isinstance(axis, str):
            axis = np.array([1 if axis == 'x' else 0, 1 if axis == 'y' else 0, 1 if axis == 'z' else 0])
        else:
            axis = np.array(axis)
        if len(self.rotations) == 0:
            self.rotations = np.array([quaternion_from_angle_axis(angle, axis)])
        else:
            self.rotations = np.vstack([self.rotations, np.array(quaternion_from_angle_axis(angle, axis))])

    def replace_last_rotation(self, angle, axis):
        if isinstance(axis, str):
            axis = np.array([1 if axis == 'x' else 0, 1 if axis == 'y' else 0, 1 if axis == 'z' else 0])
        else:
            axis = np.array(axis)
        if len(self.rotations) == 0:
            self.rotations = np.array([quaternion_from_angle_axis(angle, axis)])
        else:
            self.rotations[-1] = np.array(quaternion_from_angle_axis(angle, axis))

    def set_rotations(self, rotations):
        self.rotations = np.array(rotations)

    def set_from_config(self, det_config):
        self.name = det_config['name']
        self.set_size(det_config['det_size']['x'], det_config['det_size']['y'], det_config['det_size']['z'])
        self.set_center(det_config['det_center_coords']['x'], det_config['det_center_coords']['y'],
                        det_config['det_center_coords']['z'])

        x_rot, y_rot = det_config['det_orientation']['x'], det_config['det_orientation']['y']
        z_rot = det_config['det_orientation']['z']
        x_rot = None if x_rot == 0 else x_rot
        y_rot = None if y_rot == 0 else y_rot
        z_rot = None if z_rot == 0 else z_rot
        self.set_orientation(x_rot, y_rot, z_rot)

    def set_pitch_x(self, pitch_x):
        self.pitch_x = pitch_x

    def set_pitch_y(self, pitch_y):
        self.pitch_y = pitch_y

    def set_chip_space(self, chip_space):
        self.chip_space = chip_space

    def get_cluster_centroid_by_trigger(self, trigger):
        return self.cluster_centroids[self.cluster_triggers.index(trigger)]

    def get_largest_cluster_chip_num_by_trigger(self, trigger):
        return self.cluster_chips[self.cluster_triggers.index(trigger)][0]

    def read_banco_data(self, file_path):
        self.data = read_banco_file(file_path)

    def read_banco_noise(self, file_path, noise_threshold=1):
        self.noise_pixels = get_noise_pixels(read_banco_file(file_path), noise_threshold)

    def get_data_noise_pixels(self, noise_threshold=None):
        if noise_threshold is None:  # Calc prob of pixel firing for 4 hits per event. Multiply for fluctuations
            noise_threshold = 4.0 / (self.n_pix_y * self.n_pix_x * self.n_chips) * 10
        noise_threshold = max(1, int(noise_threshold * len(self.data)))
        self.data_noise_pixels = get_noise_pixels(self.data, noise_threshold)

    def combine_data_noise(self):
        self.noise_pixels = np.concatenate([self.noise_pixels, self.data_noise_pixels], axis=0)

    def cluster_data(self, min_pixels=2, max_pixels=100, chip=None, event_list=None):
        data = self.data
        if event_list is not None:
            data = data[np.isin(data[:, 0], event_list)]
        trigger_col = data[:, 0]  # Get all triggers, repeated if more than one hit per trigger
        unique_trigger_rows = np.unique(trigger_col, return_index=True)[1]  # Get first row indices of unique triggers
        event_split_data = np.split(data, unique_trigger_rows[1:])  # Split the data into events
        event_split_data = {event[0][0]: event[:, 1:] for event in event_split_data}  # Dict of events by trigger

        self.cluster_triggers, self.clusters, self.cluster_chips = [], [], []
        for trigger_id, hit_pixels in event_split_data.items():
            if len(hit_pixels) < min_pixels or len(hit_pixels) > max_pixels:
                continue
            clusters, cluster_chips = [], []
            # Cluster chip by chip. Gap between is too wide for clusters to span chips.
            for chip_i in range(max(hit_pixels[:, 1] // self.n_pix_y) + 1):
                if chip is not None and chip_i != chip:
                    continue
                chip_hit_pixels = hit_pixels[np.where(hit_pixels[:, 1] // self.n_pix_y == chip_i)]
                chip_clusters = find_clusters(chip_hit_pixels, self.noise_pixels, min_pixels=min_pixels)
                clusters.extend(chip_clusters)
                cluster_chips.extend([chip_i] * len(chip_clusters))
            # clusters = find_clusters(hit_pixels, self.noise_pixels, min_pixels=min_pixels)
            # if chip is not None:
            #     clusters = [cluster for cluster in clusters if np.all(cluster[:, 1] // self.n_pix_y == chip)]
            if len(clusters) == 0:
                continue
            self.cluster_triggers.append(trigger_id)
            self.clusters.append(clusters)
            self.cluster_chips.append(cluster_chips)
            # clusters_xy = convert_clusters_to_xy(clusters, self.pitch_x, self.pitch_y, self.chip_space)
            # centroids, num_pixels = get_cluster_centroids(clusters_xy)
            # self.all_cluster_centroids_local_coords.append(centroids)
            # self.all_cluster_num_pixels.append(num_pixels)
        self.get_cluster_centroids()

    def get_cluster_centroids(self):
        self.all_cluster_centroids_local_coords, self.all_cluster_num_pixels = [], []
        for trigger_id, clusters in zip(self.cluster_triggers, self.clusters):
            clusters_xy = convert_clusters_to_xy(clusters, self.pitch_x, self.pitch_y, self.chip_space, self.n_pix_y)
            centroids, num_pixels = get_cluster_centroids(clusters_xy)
            self.all_cluster_centroids_local_coords.append(centroids)
            self.all_cluster_num_pixels.append(num_pixels)

    def get_largest_clusters(self):
        largest_clusters_data = get_largest_cluster(self.clusters, self.all_cluster_centroids_local_coords,
                                                    self.all_cluster_num_pixels, self.cluster_chips)
        largest_clusters, largest_cluster_centroids, largest_clust_pix, largest_clust_chips = largest_clusters_data
        self.largest_clusters = largest_clusters
        self.largest_cluster_centroids_local_coords = np.array(largest_cluster_centroids)
        self.largest_cluster_chips = largest_clust_chips
        # print(f'Largest cluster centroids: {self.largest_cluster_centroids_local_coords}')
        # print(self.largest_cluster_centroids_local_coords.shape)
        self.largest_cluster_num_pix = largest_clust_pix

    def get_clusters_on_chip(self, chip):
        chip_clusters = []
        for clusters in self.clusters:
            chip_clusters.append([cluster for cluster in clusters if np.all(cluster[:, 1] // self.n_pix_y == chip)])
        return chip_clusters

    def convert_cluster_coords(self):
        self.cluster_centroids = self.largest_cluster_centroids_local_coords
        zs = np.full((len(self.cluster_centroids), 1), 0)  # Add z coordinate to centroids
        self.cluster_centroids = np.hstack((self.cluster_centroids, zs))  # Combine x, y, z

        # Center coordinates around center of detector
        self.cluster_centroids = self.cluster_centroids - self.active_size / 2

        # Rotate cluster centroids to global coordinates
        equivalent_quaternion = combine_quaternions(self.rotations)
        self.cluster_centroids = rotate_coordinates(self.cluster_centroids, equivalent_quaternion)

        # Translate cluster centroids to global coordinates
        self.cluster_centroids = self.cluster_centroids + self.center

    def get_cluster_centroids_global_coords(self):
        cluster_centoids_global_coords = []
        for clusters, cluster_centroids in zip(self.clusters, self.all_cluster_centroids_local_coords):
            clusters_xy = convert_clusters_to_xy(clusters, self.pitch_x, self.pitch_y, self.chip_space, self.n_pix_y)
            centroids, num_pixels = get_cluster_centroids(clusters_xy)
            zs = np.full((len(centroids), 1), 0)  # Add z coordinate to centroids
            centroids = np.hstack((centroids, zs))  # Combine x, y, z
            centroids = centroids - self.active_size / 2
            equivalent_quaternion = combine_quaternions(self.rotations)
            centroids = rotate_coordinates(centroids, equivalent_quaternion)
            centroids = centroids + self.center
            cluster_centoids_global_coords.append(centroids)
        return cluster_centoids_global_coords

    def plot_largest_cluster_centroids_local_coords(self):
        fig, ax = plt.subplots()
        ax.scatter(self.largest_cluster_centroids_local_coords[:, 0], self.largest_cluster_centroids_local_coords[:, 1],
                   marker='o', alpha=0.5)
        ax.set_title('Largest Cluster Centroids Local Coordinates')
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        fig.tight_layout()

    def plot_cluster_centroids(self):
        fig, ax = plt.subplots()
        ax.scatter(self.cluster_centroids[:, 0], self.cluster_centroids[:, 1], marker='o', alpha=0.5)
        ax.set_title('Largest Cluster Centroids')
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        fig.tight_layout()


def read_banco_file(file_path):
    field_name = 'fData'
    with uproot.open(file_path) as file:
        tree_name = f"{file.keys()[0].split(';')[0]};{max([int(key.split(';')[-1]) for key in file.keys()])}"
        tree = file[tree_name]
        data = np.array(tree[field_name].array(library='np'))
        trg_nums, chip_nums, col_nums, row_nums = data['trgNum'], data['chipId'], data['col'], data['row']
        col_nums = col_nums + (chip_nums - 4) * 1024
        data = np.array([trg_nums, row_nums, col_nums]).T

    return data


def get_noise_pixels(data, noise_threshold=0.01):
    """
    Get pixels that are noisy in the noise data.
    :param data: Data from noise run, ideally no signal. Shape (triggers, rows, cols)
    :param noise_threshold: Percentage of triggers a pixel must fire in to be considered noisy.
    :return: Array of noisy pixels. Shape (rows, cols)
    """
    triggers, rows_cols = data[:, 0], data[:, 1:]
    num_triggers = np.unique(triggers).size
    noise_pixels, counts = np.unique(rows_cols, return_counts=True, axis=0)
    counts = counts / num_triggers if noise_threshold < 1 else counts
    noise_pixels = noise_pixels[counts > noise_threshold]
    return noise_pixels


def find_clusters(data, noise_pixels=None, min_pixels=1):
    # Group all pixels into clusters
    neighbor_map = {i: [] for i in range(len(data))}
    for i, pixel in enumerate(data):
        for j, neighbor in enumerate(data):
            if i == j:
                continue
            if is_neighbor(pixel, neighbor):
                neighbor_map[i].append(j)

    clusters = []
    while len(neighbor_map) > 0:
        cluster = [list(neighbor_map.keys())[0]]
        while True:
            new_neighbors = []
            for pixel_i in cluster:
                if pixel_i in neighbor_map:
                    new_neighbors.extend(neighbor_map.pop(pixel_i))
            if len(new_neighbors) == 0:
                break
            cluster.extend(new_neighbors)
        clusters.append(list(set(cluster)))  # Remove duplicates

    good_clusters = []
    for cluster in clusters:
        # Remove noise pixels from cluster
        cluster = [np.array(data[pixel]) for pixel in cluster]
        if noise_pixels is not None:
            cluster = [pixel for pixel in cluster if not np.any(np.all(pixel == noise_pixels, axis=1))]
        if len(cluster) == 0:
            continue
        if len(cluster) < min_pixels:
            continue
        good_clusters.append(np.array(cluster))

    return good_clusters


def get_cluster_centroids(clusters):
    # Get x and y centroids of clusters
    cluster_centroids, cluster_num_pixels = [], []
    for cluster in clusters:
        cluster_centroids.append(np.mean(cluster, axis=0))
        cluster_num_pixels.append(len(cluster))
        # if len(cluster) > 1:
        #     print(f'Num_pixels: {len(cluster)}, centroid: {np.mean(cluster, axis=0)}, cluster: {cluster}')

    return cluster_centroids, cluster_num_pixels


def get_largest_cluster(clusters, cluster_centroids, cluster_num_pixels, cluster_chips):
    largest_clusters, largest_cluster_centroids, largest_cluster_num_pixels, largest_cluster_chips = [], [], [], []
    for clusters_i, cluster_centroids, num_pixels, chips in zip(clusters, cluster_centroids, cluster_num_pixels, cluster_chips):
        if len(clusters) == 1:
            largest_clusters.append(clusters_i[0])
            largest_cluster_centroids.append(cluster_centroids[0])
            largest_cluster_num_pixels.append(num_pixels[0])
            largest_cluster_chips.append(chips[0])
        else:
            max_pix_i = np.argmax(num_pixels)
            largest_clusters.append(clusters_i[max_pix_i])
            largest_cluster_centroids.append(cluster_centroids[max_pix_i])
            largest_cluster_num_pixels.append(num_pixels[max_pix_i])
            largest_cluster_chips.append(chips[max_pix_i])
    return largest_clusters, largest_cluster_centroids, largest_cluster_num_pixels, largest_cluster_chips


def convert_clusters_to_xy(cluster_centroids, pitch_x=30., pitch_y=30., chip_space=15., n_pix_y=1024):
    cluster_centroids_xy = []
    for event in cluster_centroids:
        event_xy = []
        for cluster in event:
            x, y = convert_row_col_to_xy(cluster[0], cluster[1], chip=None, n_pix_y=n_pix_y, pix_size_x=pitch_x,
                                         pix_size_y=pitch_y, chip_space=chip_space)
            event_xy.append([x, y])
        cluster_centroids_xy.append(event_xy)
    return cluster_centroids_xy


def is_neighbor(pixel1, pixel2, threshold=1.9):
    return np.sqrt(np.sum((pixel1 - pixel2) ** 2)) <= threshold


def convert_row_col_to_xy(row, col, chip=None, n_pix_y=1024, pix_size_x=30., pix_size_y=30., chip_space=15.):
    """
    Given a row, column, and chip number, return the x and y coordinates of the pixel.
    :param row: Row pixel number, 0-511
    :param col: Column pixel number, 0-1023 or 0-1024 * chip_num if chip is None
    :param chip: Chip number, 0-4
    :param n_pix_y: Number of pixels in the y direction
    :param pix_size_x: Pixel size in the x direction
    :param pix_size_y: Pixel size in the y direction
    :param chip_space: Space between chips
    :return: x, y coordinates of the pixel in mm
    """

    x = (row + 0.5) * pix_size_x
    if chip is None:
        chip = col // n_pix_y
        col = col % n_pix_y
    y = (col + 0.5) * pix_size_y + chip * (n_pix_y * pix_size_y + chip_space)

    x, y = x / 1000, y / 1000  # Convert um to mm

    return x, y


# def rotate_coordinates(x, y, z, orientation):
#     x_rot = x * np.cos(orientation[0]) - y * np.sin(orientation[0])
#     y_rot = x * np.sin(orientation[0]) + y * np.cos(orientation[0])
#     z_rot = z
#
#     x_rot = x_rot * np.cos(orientation[1]) - z_rot * np.sin(orientation[1])
#     z_rot = x_rot * np.sin(orientation[1]) + z_rot * np.cos(orientation[1])
#
#     y_rot = y_rot * np.cos(orientation[2]) - z_rot * np.sin(orientation[2])
#     z_rot = y_rot * np.sin(orientation[2]) + z_rot * np.cos(orientation[2])
#
#     return x_rot, y_rot, z_rot


# def rotate_coordinates(coords, orientation):
#     """
#     Rotate list of 3 dimensional coordinates by the given 3d orientation angles.
#     :param coords:
#     :param orientation:
#     :return:
#     """
#
#     orientation = np.deg2rad(orientation)
#
#     x_rot = coords[:, 0] * np.cos(orientation[0]) - coords[:, 1] * np.sin(orientation[0])
#     y_rot = coords[:, 0] * np.sin(orientation[0]) + coords[:, 1] * np.cos(orientation[0])
#     z_rot = coords[:, 2]
#
#     x_rot = x_rot * np.cos(orientation[1]) - z_rot * np.sin(orientation[1])
#     z_rot = x_rot * np.sin(orientation[1]) + z_rot * np.cos(orientation[1])
#
#     y_rot = y_rot * np.cos(orientation[2]) - z_rot * np.sin(orientation[2])
#     z_rot = y_rot * np.sin(orientation[2]) + z_rot * np.cos(orientation[2])
#
#     return np.array([x_rot, y_rot, z_rot]).T


# def rotate_coordinates(coords, angles):
#     """
#     Rotate 3D coordinates about the x, y, and z axes.
#     :param coords: Array of 3D coordinates (shape: Nx3)
#     :param angles: 3D vector representing the rotation angles for x, y, and z axes (in radians)
#     :return: Array of rotated 3D coordinates (shape: Nx3)
#     """
#     x_angle, y_angle, z_angle = angles
#
#     # Rotation matrix for x-axis
#     rot_x = np.array([[1, 0, 0],
#                       [0, np.cos(x_angle), -np.sin(x_angle)],
#                       [0, np.sin(x_angle), np.cos(x_angle)]])
#
#     # Rotation matrix for y-axis
#     rot_y = np.array([[np.cos(y_angle), 0, np.sin(y_angle)],
#                       [0, 1, 0],
#                       [-np.sin(y_angle), 0, np.cos(y_angle)]])
#
#     # Rotation matrix for z-axis
#     rot_z = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
#                       [np.sin(z_angle), np.cos(z_angle), 0],
#                       [0, 0, 1]])
#
#     # Combined rotation matrix
#     rot_matrix = np.dot(rot_z, np.dot(rot_y, rot_x))
#
#     # Apply rotation matrix to each coordinate
#     rotated_coords = np.dot(coords, rot_matrix.T)
#
#     return rotated_coords


# def rotate_using_quaternions(coords, quaternion_rotation):
#     """
#     Rotate 3D coordinates using a quaternion.
#     :param coords: Array of 3D coordinates (shape: Nx3)
#     :param quaternion_rotation: Quaternion representing the rotation
#     :return: Array of rotated 3D coordinates (shape: Nx3)
#     """
#     # Convert to quaternion array
#     q_rotation = np.quaternion(*quaternion_rotation)
#
#     # Convert each coordinate to a quaternion, apply rotation, and convert back
#     rotated_coords = np.array([q_rotation * np.quaternion(0, *coord) * q_rotation.conjugate() for coord in coords])
#
#     # Extract the vector part (x, y, z)
#     rotated_coords = np.array([[q.x, q.y, q.z] for q in rotated_coords])
#
#     return rotated_coords


def rotate_coordinates(coords, quat):
    """
    Rotate 3D coordinates using a quaternion.
    :param coords: Array of 3D coordinates (shape: Nx3)
    :param quat: Quaternion (w, x, y, z)
    :return: Array of rotated 3D coordinates (shape: Nx3)
    """
    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(quat)

    # Apply rotation matrix to each coordinate
    rotated_coords = np.dot(coords, rotation_matrix.T)

    return rotated_coords


def quaternion_from_angle_axis(angle, axis):
    """
    Create a quaternion from an angle and rotation axis.
    :param angle: Rotation angle in degrees
    :param axis: Rotation axis (x, y, z)
    :return: Quaternion (w, x, y, z)
    """
    half_angle = np.deg2rad(angle) / 2
    w = np.cos(half_angle)
    x, y, z = axis * np.sin(half_angle)
    return w, x, y, z


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    :param q1: First quaternion (w, x, y, z)
    :param q2: Second quaternion (w, x, y, z)
    :return: Resulting quaternion from q1 * q2 (w, x, y, z)
    """
    q1 = np.quaternion(*q1)
    q2 = np.quaternion(*q2)
    q_result = q1 * q2
    return q_result.w, q_result.x, q_result.y, q_result.z


def combine_quaternions(quaternion_list):
    """
    Combine a list of quaternions into a single equivalent quaternion.
    :param quaternion_list: List of quaternions (each quaternion is a list [w, x, y, z])
    :return: Equivalent quaternion from multiplying all quaternions in the list (numpy.quaternion)
    """
    if len(quaternion_list) == 0:
        return np.quaternion(1, 0, 0, 0)  # Identity quaternion if the list is empty

    # Convert list of lists to list of numpy.quaternion objects
    quaternions = [np.quaternion(q[0], q[1], q[2], q[3]) for q in quaternion_list]

    # Start with the first quaternion
    combined_quaternion = quaternions[0]

    # Multiply sequentially with the rest of the quaternions
    for q in quaternions[1:]:
        combined_quaternion *= q

    return combined_quaternion


def quaternion_to_rotation_matrix(quat):
    """
    Convert a quaternion into a 3x3 rotation matrix.
    :param quat: Quaternion (w, x, y, z)
    :return: 3x3 rotation matrix
    """
    if isinstance(quat, np.quaternion):
        quat = (quat.w, quat.x, quat.y, quat.z)
    w, x, y, z = quat
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])


def find_cluster_test():
    # Generate sample data
    data = np.array([
        [0, 1],
        [1, 3],
        [3, 4],
        [3, 5],
        [9, 9],
        [9, 10],
        [9, 11]
    ])

    # Define noise pixels
    noise_pixels = np.array([[1, 3]])

    # Find clusters
    clusters = find_clusters(data, noise_pixels)
    centroids, num_pixels = get_cluster_centroids(clusters)
    print(centroids)
    print(num_pixels)


def convert_row_col_xy_test():
    n_chip = 5
    n_pix_x, n_pix_y = 512, 1024

    fig, ax = plt.subplots()
    for chip in range(n_chip):
        xs, ys = [], []
        for row in np.arange(0, n_pix_x, 16):
            for col in np.arange(0, n_pix_y, 1):
                x, y = convert_row_col_to_xy(row, col, chip)
                xs.append(x)
                ys.append(y)
        ax.scatter(xs, ys, label=f'Chip {chip}', marker='.', alpha=0.5)
    ax.set_title('Chip XY Positions')
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.legend()
    fig.tight_layout()
    plt.show()
