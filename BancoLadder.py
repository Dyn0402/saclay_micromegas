#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 07 3:37 PM 2024
Created in PyCharm
Created as saclay_micromegas/BancoLadder.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak
import vector


class BancoLadder:
    def __init__(self, name='0', config=None):
        self.name = name
        self.center = np.array([0, 0, 0])
        self.size = np.array([0, 0, 0])
        self.orientation = np.array([0, 0, 0])

        self.pitch_x = 29.24
        self.pitch_y = 26.88
        self.chip_space = 15

        if config is not None:
            self.set_from_config(config)

        self.data = None
        self.noise_pixels = None
        self.data_noise_pixels = None

        self.clusters = None
        self.cluster_triggers = None
        self.all_cluster_centroids_local_coords = None
        self.all_cluster_num_pixels = None
        self.largest_cluster_centroids_local_coords = None
        self.largest_cluster_num_pix = None

        self.cluster_centroids = None
        self.cluster_num_pix = None

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

    def set_orientation(self, x=0, y=0, z=0):
        self.orientation = np.array([x, y, z])

    def set_from_config(self, det_config):
        self.name = det_config['name']
        self.set_size(det_config['det_size']['x'], det_config['det_size']['y'], det_config['det_size']['z'])
        self.set_center(det_config['det_center_coords']['x'], det_config['det_center_coords']['y'],
                        det_config['det_center_coords']['z'])
        self.set_orientation(det_config['det_orientation']['x'], det_config['det_orientation']['y'],
                             det_config['det_orientation']['z'])

    def set_pitch_x(self, pitch_x):
        self.pitch_x = pitch_x

    def set_pitch_y(self, pitch_y):
        self.pitch_y = pitch_y

    def read_banco_data(self, file_path):
        self.data = read_banco_file(file_path)

    def read_banco_noise(self, file_path, noise_threshold=0.01):
        self.noise_pixels = get_noise_pixels(read_banco_file(file_path), noise_threshold)

    def get_data_noise_pixels(self, noise_threshold=0.01):
        self.data_noise_pixels = get_noise_pixels(self.data, noise_threshold)

    def combine_data_noise(self):
        self.noise_pixels = np.concatenate([self.noise_pixels, self.data_noise_pixels], axis=0)

    def cluster_data(self, min_pixels=2):
        trigger_col = self.data[:, 0]  # Get all triggers, repeated if more than one hit per trigger
        unique_trigger_rows = np.unique(trigger_col, return_index=True)[1]  # Get first row indices of unique triggers
        event_split_data = np.split(self.data, unique_trigger_rows[1:])  # Split the data into events
        event_split_data = {event[0][0]: event[:, 1:] for event in event_split_data}  # Dict of events by trigger

        self.cluster_triggers, self.all_cluster_centroids_local_coords, self.all_cluster_num_pixels = [], [], []
        self.clusters = []
        for trigger_id, hit_pixels in event_split_data.items():
            if len(hit_pixels) < min_pixels:
                continue
            clusters = find_clusters(hit_pixels, self.noise_pixels, min_pixels=min_pixels)
            if len(clusters) == 0:
                continue
            self.cluster_triggers.append(trigger_id)
            self.clusters.append(clusters)
            clusters_xy = convert_clusters_to_xy(clusters, self.pitch_x, self.pitch_y, self.chip_space)
            centroids, num_pixels = get_cluster_centroids(clusters_xy)
            self.all_cluster_centroids_local_coords.append(centroids)
            self.all_cluster_num_pixels.append(num_pixels)

    def get_largest_clusters(self):
        largest_cluster_centroids, largest_clust_pix = get_largest_cluster(self.all_cluster_centroids_local_coords,
                                                                           self.all_cluster_num_pixels)
        self.largest_cluster_centroids_local_coords = np.array(largest_cluster_centroids)
        # print(f'Largest cluster centroids: {self.largest_cluster_centroids_local_coords}')
        # print(self.largest_cluster_centroids_local_coords.shape)
        self.largest_cluster_num_pix = largest_clust_pix

    def convert_cluster_coords(self):
        self.cluster_centroids = self.largest_cluster_centroids_local_coords
        # print(f'Cluster centroids from clustering: {self.cluster_centroids[:4]}')
        zs = np.full((len(self.cluster_centroids), 1), 0)  # Add z coordinate to centroids
        # print(f'Zs: {zs[:4]}')
        self.cluster_centroids = np.hstack((self.cluster_centroids, zs))  # Combine x, y, z
        # print(f'Cluster centroids after hstack: {self.cluster_centroids[:4]}')

        # Rotate cluster centroids to global coordinates
        self.cluster_centroids = rotate_coordinates(self.cluster_centroids, self.orientation)
        # print(f'Cluster centroids after rotation: {self.cluster_centroids[:4]}')

        # Translate cluster centroids to global coordinates
        # print(f'Translation: {self.center}')
        self.cluster_centroids = self.cluster_centroids + self.center
        # print(f'Cluster centroids after translation: {self.cluster_centroids[:4]}')


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


def cluster_data(data, noise_pixels=None, min_pixels=1):
    data = np.split(data, np.unique(data[:, 0], return_index=True)[1][1:])

    trigger_ids, cluster_centroids, cluster_num_pixels = [], [], []
    for trigger in data:
        trigger_ids.append(trigger[0][0])
        clusters = find_clusters(trigger[:, 1:], noise_pixels, min_pixels=min_pixels)
        clusters_xy = convert_clusters_to_xy(clusters)
        centroids, num_pixels = get_cluster_centroids(clusters_xy)
        cluster_centroids.append(centroids)
        cluster_num_pixels.append(num_pixels)

    return trigger_ids, cluster_centroids, cluster_num_pixels


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


def get_largest_cluster(cluster_centroids, cluster_num_pixels):
    new_cluster_centroids, new_cluster_num_pixels = [], []
    for clusters, num_pixels in zip(cluster_centroids, cluster_num_pixels):
        if len(clusters) == 1:
            new_cluster_centroids.append(clusters[0])
            new_cluster_num_pixels.append(num_pixels[0])
        else:
            max_pix_i = np.argmax(num_pixels)
            new_cluster_centroids.append(clusters[max_pix_i])
            new_cluster_num_pixels.append(num_pixels[max_pix_i])
    return new_cluster_centroids, new_cluster_num_pixels


def convert_clusters_to_xy(cluster_centroids, pitch_x=30., pitch_y=30., chip_space=15.):
    cluster_centroids_xy = []
    for event in cluster_centroids:
        event_xy = []
        for cluster in event:
            x, y = convert_row_col_to_xy(cluster[0], cluster[1], chip=None, n_pix_y=1024, pix_size_x=pitch_x,
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
    :return:
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


def rotate_coordinates(coords, orientation):
    """
    Rotate list of 3 dimensional coordinates by the given 3d orientation angles.
    :param coords:
    :param orientation:
    :return:
    """

    orientation = np.deg2rad(orientation)

    x_rot = coords[:, 0] * np.cos(orientation[0]) - coords[:, 1] * np.sin(orientation[0])
    y_rot = coords[:, 0] * np.sin(orientation[0]) + coords[:, 1] * np.cos(orientation[0])
    z_rot = coords[:, 2]

    x_rot = x_rot * np.cos(orientation[1]) - z_rot * np.sin(orientation[1])
    z_rot = x_rot * np.sin(orientation[1]) + z_rot * np.cos(orientation[1])

    y_rot = y_rot * np.cos(orientation[2]) - z_rot * np.sin(orientation[2])
    z_rot = y_rot * np.sin(orientation[2]) + z_rot * np.cos(orientation[2])

    return np.array([x_rot, y_rot, z_rot]).T


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
