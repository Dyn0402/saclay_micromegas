#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 17 6:42 PM 2024
Created in PyCharm
Created as saclay_micromegas/BancoLadder_new.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
from scipy.optimize import minimize

import uproot

from Detector import Detector


class BancoLadder(Detector):
    def __init__(self, name=None, center=None, size=None, rotations=None, config=None, ladder_config=None):
        super().__init__(name=name, center=center, size=size, rotations=rotations, config=config)

        self.ladder_num = int(self.name[-3:])
        self.data_path = None
        self.noise_path = None

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

        self.ladder_config = ladder_config if ladder_config is not None else {}

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

    def read_banco_data(self, file_path, event_start=None, event_stop=None):
        self.data_path = file_path
        self.data = read_banco_file(file_path, event_start=event_start, event_stop=event_stop)

    def read_banco_noise(self, file_path, noise_threshold=1):
        self.noise_path = file_path
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
        print(f'Number of hits: {len(data)}')
        if event_list is not None:
            data = data[np.isin(data[:, 0], event_list)]
        print(f'Number of hits after filtering: {len(data)}')
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
        if self.cluster_centroids is None or len(self.cluster_centroids) == 0:
            print('No cluster centroids to convert.')
            return
        zs = np.full((len(self.cluster_centroids), 1), 0)  # Add z coordinate to centroids
        self.cluster_centroids = np.hstack((self.cluster_centroids, zs))  # Combine x, y, z

        self.cluster_centroids = self.convert_coords_to_global(self.cluster_centroids)

    def get_cluster_centroids_global_coords(self):
        cluster_centoids_global_coords = []
        for clusters, cluster_centroids in zip(self.clusters, self.all_cluster_centroids_local_coords):
            clusters_xy = convert_clusters_to_xy(clusters, self.pitch_x, self.pitch_y, self.chip_space, self.n_pix_y)
            centroids, num_pixels = get_cluster_centroids(clusters_xy)
            zs = np.full((len(centroids), 1), 0)  # Add z coordinate to centroids
            centroids = np.hstack((centroids, zs))  # Combine x, y, z
            centroids = self.convert_coords_to_global(centroids)

            cluster_centoids_global_coords.append(centroids)
        return cluster_centoids_global_coords

    def get_banco_traversing_triggers(self, ray_data, expansion_factor=1.5):
        z_orig = self.center[2]
        x_bnds = self.center[0] - self.size[0] / 2, self.center[0] + self.size[0] / 2
        y_bnds = self.center[1] - self.size[1] / 2, self.center[1] + self.size[1] / 2
        ray_traversing_triggers = ray_data.get_traversing_triggers(z_orig, x_bnds, y_bnds,
                                                                   expansion_factor=expansion_factor)
        banco_traversing_triggers = ray_traversing_triggers - 1  # Rays start at 1, banco starts at 0
        print(f'Number of traversing triggers: {len(ray_traversing_triggers)}')
        print(f'Bounds: x={x_bnds}, y={y_bnds}, z={z_orig}')
        return banco_traversing_triggers

    def align_ladder(self, ray_data):
        """
        Align the ladder to the ray data.
        :param ray_data:
        :return:
        """
        iterations, zs = list(np.arange(3)), []
        z_rot_align = 0
        for i in iterations:
            print()
            print(f'Iteration {i}: Getting residuals for ladder {self.ladder_num} with '
                  f'center=[{self.center[0]:.2f}, {self.center[1]:.2f}, {self.center[2]:.2f}] mm, rotations='
                  f'z_rot={z_rot_align:.3f}, {self.rotations}')
            zs.append(self.center[2])
            good_triggers = self.get_close_triggers(ray_data)
            x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = self.get_residuals_no_fit_triggers(ray_data, good_triggers, plot=False)
            print(f'Ladder {self.name} X Residuals Mean: {x_mu} Sigma: {x_sd}')
            print(f'Ladder {self.name} Y Residuals Mean: {y_mu} Sigma: {y_sd}')
            aligned_x, aligned_y = self.center[0] + x_mu, self.center[1] + y_mu
            self.set_center(x=aligned_x, y=aligned_y)
            self.convert_cluster_coords()

            z_align = self.res_z_alignment(ray_data, z_range=(20 / (i + 1)), z_points=200, plot=False)
            self.set_center(z=z_align)
            self.convert_cluster_coords()

        good_triggers = self.get_close_triggers(ray_data)
        self.get_residuals_no_fit_triggers(ray_data, good_triggers, plot=True)

        fig, ax = plt.subplots()
        ax.plot(iterations + [iterations[-1] + 1], zs + [self.center[2]], marker='o')
        ax.grid(zorder=0)
        ax.set_title(f'Ladder {self.name} Z Alignment Iterations')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Z Alignment (mm)')
        fig.tight_layout()

        self.align_minimizer(ray_data)

        # x_rot_align, y_rot_align, z_rot_align = self.align_rotation(ray_data, plot=True)
        # self.add_rotation(z_rot_align, 'z')
        # self.convert_cluster_coords()
        # good_triggers = self.get_close_triggers(ray_data)
        # x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = self.get_residuals_no_fit_triggers(ray_data, good_triggers, plot=False)
        # print(f'Ladder {self.name} X Residuals Mean: {x_mu} Sigma: {x_sd}')
        # print(f'Ladder {self.name} Y Residuals Mean: {y_mu} Sigma: {y_sd}')
        # good_triggers = self.get_close_triggers(ray_data)
        # self.get_residuals_no_fit_triggers(ray_data, good_triggers, plot=True)
        # self.align_rotation(ray_data, plot=True)

        x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = self.get_residuals_no_fit_triggers(ray_data, good_triggers, plot=False)
        print(f'Ladder {self.name} X Residuals Mean: {x_mu} Sigma: {x_sd}')
        print(f'Ladder {self.name} Y Residuals Mean: {y_mu} Sigma: {y_sd}')

        self.plot_cluster_centroids()

    def align_ladder_old(self, ray_data):
        """
        Align the ladder to the ray data.
        :param ray_data:
        :return:
        """
        iterations, zs = list(np.arange(10)), []
        z_rot_align = 0
        for i in iterations:
            print()
            print(f'Iteration {i}: Getting residuals for ladder {self.ladder_num} with '
                  f'center=[{self.center[0]:.2f}, {self.center[1]:.2f}, {self.center[2]:.2f}] mm, rotations='
                  f'z_rot={z_rot_align:.3f}, {self.rotations}')
            zs.append(self.center[2])
            # x_res_mean, x_res_sigma, y_res_mean, y_res_sigma = banco_get_residuals(ladder, ray_data, False)
            # x_res_mean, x_res_sigma, y_res_mean, y_res_sigma, r_mu, r_sig = banco_get_residuals_no_fit(ladder, ray_data)
            good_triggers = self.get_close_triggers(ray_data)
            x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = self.get_residuals_no_fit_triggers(ray_data, good_triggers, plot=False)
            print(f'Ladder {self.name} X Residuals Mean: {x_mu} Sigma: {x_sd}')
            print(f'Ladder {self.name} Y Residuals Mean: {y_mu} Sigma: {y_sd}')
            # plt.show()
            aligned_x, aligned_y = self.center[0] + x_mu, self.center[1] + y_mu
            self.set_center(x=aligned_x, y=aligned_y)
            self.convert_cluster_coords()

            z_align = self.res_z_alignment(ray_data, z_range=(20 / (i + 1)), z_points=200, plot=False)
            self.set_center(z=z_align)
            self.convert_cluster_coords()

            # x_rot_align, y_rot_align, z_rot_align = banco_align_rotation(ladder, ray_data, plot=False, n_points=100)
            # ladder.replace_last_rotation(z_rot_align, 'z')
            # ladder.convert_cluster_coords()

        # x_rot_align, y_rot_align, z_rot_align = banco_align_rotation(ladder, ray_data, plot=True)
        # print(f'Final rotation: {ladder.rotations}')
        # plt.show()
        good_triggers = self.get_close_triggers(ray_data)
        self.get_residuals_no_fit_triggers(ray_data, good_triggers, plot=True)
        # plt.show()

        fig, ax = plt.subplots()
        ax.plot(iterations + [iterations[-1] + 1], zs + [self.center[2]], marker='o')
        ax.grid(zorder=0)
        ax.set_title(f'Ladder {self.name} Z Alignment Iterations')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Z Alignment (mm)')
        fig.tight_layout()

        # good_triggers = get_close_triggers(ladder, ray_data)
        # original_center = ladder.center
        # print('Optimizing center...')
        # result = minimize(minimize_translation_residuals, original_center, args=(ladder, ray_data, good_triggers),
        #                   tol=1e-4)
        # print("Optimization Result:")
        # print("Success:", result.success)
        # print("Message:", result.message)
        # print("Number of Iterations:", result.nit)
        # print("Optimal Value of x:", result.x)
        # print("Function Value at Optimal x:", result.fun)
        # ladder.set_center(x=result.x[0], y=result.x[1], z=result.x[2])
        # x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = banco_get_residuals_no_fit_triggers(ladder, ray_data, good_triggers)
        # aligned_x, aligned_y = ladder.center[0] + x_mu, ladder.center[1] + y_mu
        # ladder.set_center(x=aligned_x, y=aligned_y)
        # print(f'Original center: {original_center}')
        # print(f'New center: {ladder.center}')
        # plt.show()

        # banco_res_z_alignment(ladder, ray_data, plot=True)
        # banco_xyz_alignment(ladder, ray_data, plot=True)
        # plt.show()

        # x_res_mean, x_res_sigma, y_res_mean, y_res_sigma = banco_get_residuals(ladder, ray_data, False)
        # plt.show()

        # banco_get_pixel_spacing(ladder, ray_data, True)
        # plt.show()

        x_rot_align, y_rot_align, z_rot_align = self.align_rotation(ray_data, plot=True)
        self.add_rotation(z_rot_align, 'z')
        self.convert_cluster_coords()
        good_triggers = self.get_close_triggers(ray_data)
        x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = self.get_residuals_no_fit_triggers(ray_data, good_triggers, plot=False)
        print(f'Ladder {self.name} X Residuals Mean: {x_mu} Sigma: {x_sd}')
        print(f'Ladder {self.name} Y Residuals Mean: {y_mu} Sigma: {y_sd}')
        good_triggers = self.get_close_triggers(ray_data)
        self.get_residuals_no_fit_triggers(ray_data, good_triggers, plot=True)
        self.align_rotation(ray_data, plot=True)
        # z_rot_angles.append(z_rot_align)
        # n_events.append(len(good_triggers))
        # plt.show()
        # ladder.set_orientation(x_rot_align, y_rot_align, z_rot_align)
        # print(f'{ladder.name} new orientation: {ladder.orientation}')
        # ladder.convert_cluster_coords()

        # Manually align
        # x_align, y_align, z_align = ladder.center
        # x_align -= manual_align[ladder_num]['x']
        # y_align -= manual_align[ladder_num]['y']
        # z_align -= manual_align[ladder_num]['z']
        # ladder.set_center(x=x_align, y=y_align, z=z_align)
        # ladder.convert_cluster_coords()

        self.plot_cluster_centroids()

    def get_close_triggers(self, ray_data):
        """
        Get triggers in ray data that are close to the ladder.
        :param ray_data: Ray data object
        :return: List of event numbers close to the ladder.
        """
        ray_trigger_ids = np.array(self.cluster_triggers) + 1  # Banco starts at 0, rays start at 1
        x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(self.center[2], ray_trigger_ids)
        x_rays, y_rays, event_num_rays = remove_outlying_rays(x_rays, y_rays, event_num_rays, self.size, 1.2)

        cluster_centroids, banco_triggers = np.array(self.cluster_centroids), np.array(self.cluster_triggers) + 1
        cluster_centroids = cluster_centroids[np.isin(banco_triggers, event_num_rays)]

        x_res, y_res = get_ray_ladder_residuals(x_rays, y_rays, cluster_centroids)
        x_res, y_res = np.array(x_res), np.array(y_res)
        r_res = np.sqrt(x_res ** 2 + y_res ** 2)

        # Mask r by n_stds
        r_res_mean = np.mean(r_res)
        mask = r_res < 2 * r_res_mean

        # Mask by n_stds
        # x_res_mean, y_res_mean = np.mean(x_res), np.mean(y_res)
        # x_res_std, y_res_std = np.std(x_res), np.std(y_res)
        # mask = (x_res > x_res_mean - 3 * x_res_std) & (x_res < x_res_mean + 3 * x_res_std) & \
        #        (y_res > y_res_mean - 3 * y_res_std) & (y_res < y_res_mean + 3 * y_res_std)
        # Mask by 10 and 90 percentiles
        # mask = (x_res > np.percentile(x_res, 10)) & (x_res < np.percentile(x_res, 90)) & \
        #        (y_res > np.percentile(y_res, 10)) & (y_res < np.percentile(y_res, 90))
        good_ray_triggers = event_num_rays[mask]

        return good_ray_triggers

    def get_residuals_no_fit_triggers(self, ray_data, ray_triggers, plot=False):
        x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(self.center[2], ray_triggers)

        cluster_centroids, banco_triggers = np.array(self.cluster_centroids), np.array(self.cluster_triggers) + 1
        cluster_centroids = cluster_centroids[np.isin(banco_triggers, event_num_rays)]

        x_res, y_res = get_ray_ladder_residuals(x_rays, y_rays, cluster_centroids, plot=plot)
        x_res, y_res = np.array(x_res), np.array(y_res)
        r_res = np.sqrt(x_res ** 2 + y_res ** 2)

        if plot:
            fig_x, ax_x = plt.subplots()
            ax_x.hist(x_res, bins=np.linspace(min(x_res), max(x_res), 25))
            ax_x.set_title(f'X Residuals')
            ax_x.set_xlabel('X Residual (mm)')
            ax_x.set_ylabel('Entries')
            fig_x.tight_layout()

            fig_y, ax_y = plt.subplots()
            ax_y.hist(y_res, bins=np.linspace(min(y_res), max(y_res), 25))
            ax_y.set_title(f'Y Residuals')
            ax_y.set_xlabel('Y Residual (mm)')
            ax_y.set_ylabel('Entries')
            fig_y.tight_layout()

            fig_r, ax_r = plt.subplots()
            ax_r.hist(r_res, bins=np.linspace(min(r_res), max(r_res), 25))
            ax_r.set_title(f'R Residuals')
            ax_r.set_xlabel('R Residual (mm)')
            ax_r.set_ylabel('Entries')
            fig_r.tight_layout()

        return np.mean(x_res), np.std(x_res), np.mean(y_res), np.std(y_res), np.mean(r_res), np.std(r_res)

    def align_minimizer(self, ray_data):
        """
        Use scipy minimizer to align ladder to ray data.
        :param ray_data:
        :return:
        """
        original_z = self.center[2]
        original_z_rot = 0.0

        z_bounds = (original_z - 20, original_z + 20)
        z_rot_bounds = (-5.0, 5.0)

        self.set_center(z=z_bounds[0])
        z_min_ray_triggers = self.get_close_triggers(ray_data)
        self.set_center(z=z_bounds[1])
        z_max_ray_triggers = self.get_close_triggers(ray_data)
        common_triggers = np.intersect1d(z_min_ray_triggers, z_max_ray_triggers)

        self.add_rotation(original_z_rot, 'z')

        res = minimize(get_residuals_minimizer, np.array([original_z, original_z_rot]),
                       args=(self, ray_data, common_triggers), bounds=[z_bounds, z_rot_bounds])
        print(res)
        z, z_rot = res.x
        print(f'Original Z: {original_z} --> Optimal Z: {z}')
        print(f'Original Z Rotation: {original_z_rot} --> Optimal Z Rotation: {z_rot}')
        self.set_center(z=z)
        self.replace_last_rotation(z_rot, 'z')
        self.convert_cluster_coords()

    def res_z_alignment(self, ray_data, z_range=20., z_points=20, plot=True):
        """
        Align ladder by minimizing residuals between ray and cluster centroids
        :param ladder:
        :param ray_data:
        :param z_range:
        :param z_points:
        :param plot:
        :return:
        """
        original_z = self.center[2]
        zs = np.linspace(original_z - z_range / 2, original_z + z_range / 2, z_points)

        self.set_center(z=min(zs))
        z_min_ray_triggers = self.get_close_triggers(ray_data)
        self.set_center(z=max(zs))
        z_max_ray_triggers = self.get_close_triggers(ray_data)
        common_triggers = np.intersect1d(z_min_ray_triggers, z_max_ray_triggers)

        x_res_widths, y_res_widths, sum_res_widths = [], [], []
        for zi in zs:
            self.set_center(z=zi)
            self.convert_cluster_coords()
            # x_res_mean, x_res_sigma, y_res_mean, y_res_sigma = banco_get_residuals(ladder, ray_data, plot=False)
            x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = self.get_residuals_no_fit_triggers(ray_data, common_triggers,
                                                                                    plot=plot)
            x_res_widths.append(x_sd)
            y_res_widths.append(y_sd)
            sum_res_widths.append(np.sqrt(x_sd ** 2 + y_sd ** 2))

        # Fit both x, y and total residuals to a quadratic function
        # x_res_widths = np.array(x_res_widths)
        # y_res_widths = np.array(y_res_widths)
        # sum_res_widths = np.array(sum_res_widths)
        # p0 = (-1, 4, original_z)
        # popt_x, pcov_x = cf(quadratic_shift, zs, x_res_widths, p0=p0)
        # popt_y, pcov_y = cf(quadratic_shift, zs, y_res_widths, p0=p0)
        # popt_sum, pcov_sum = cf(quadratic_shift, zs, sum_res_widths, p0=p0)

        # Get min of popt_sum
        # z_min_sum = popt_sum[-1]
        z_min_sum = zs[np.argmin(sum_res_widths)]

        if plot:
            fig, ax = plt.subplots()
            ax.grid(zorder=0)
            ax.scatter(zs, x_res_widths, color='blue', marker='o', label='X Residuals')
            ax.scatter(zs, y_res_widths, color='orange', marker='o', label='Y Residuals')
            ax.scatter(zs, sum_res_widths, color='green', marker='o', label='Sum Residuals')
            # zs_plt = np.linspace(min(zs), max(zs), 100)
            # ax.plot(zs_plt, quadratic_shift(zs_plt, *popt_x), color='blue', alpha=0.3)
            # ax.plot(zs_plt, quadratic_shift(zs_plt, *popt_y), color='orange', alpha=0.3)
            # ax.plot(zs_plt, quadratic_shift(zs_plt, *popt_sum), color='green', alpha=0.6)
            ax.axvline(original_z, color='g', linestyle='--', label='Original Z')
            ax.axvline(z_min_sum, color='r', linestyle='--', label='Minimized Z')
            ax.set_title(f'X Residuals vs Z {self.name}')
            ax.set_xlabel('Z (mm)')
            ax.set_ylabel('X Residual Distribution Gaussian Width (mm)')
            ax.legend()
            fig.tight_layout()

        return z_min_sum

    def align_rotation(self, ray_data, plot=True, n_points=200):
        """
        Align ladder by minimizing residuals between ray and cluster centroids
        :param ray_data:
        :param plot:
        :param n_points:
        :return:
        """

        original_rotations = self.rotations
        x_min, x_max = -0.5, 0.5
        y_min, y_max = -0.5, 0.5
        z_min, z_max = -2.5, 2.5
        x_rotations = np.linspace(x_min, x_max, n_points)
        y_rotations = np.linspace(y_min, y_max, n_points)
        z_rotations = np.linspace(z_min, z_max, n_points)
        self.add_rotation(0, [0, 0, 0])

        self.replace_last_rotation(x_min, 'x')
        self.convert_cluster_coords()
        x_min_rot_ray_triggers = self.get_close_triggers(ray_data)
        self.replace_last_rotation(x_max, 'x')
        self.convert_cluster_coords()
        x_max_rot_ray_triggers = self.get_close_triggers(ray_data)
        # Get the intersection of the two sets
        x_rot_ray_triggers = np.intersect1d(x_min_rot_ray_triggers, x_max_rot_ray_triggers)
        x_res_x_rots, y_res_x_rots, r_res_x_rots, sum_res_x_rots = [], [], [], []
        for x_rot_i in x_rotations:
            self.replace_last_rotation(x_rot_i, 'x')
            self.convert_cluster_coords()
            # x_mu, x_sd, y_mu, y_sd = banco_get_residuals(ladder, ray_data, False)
            x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = self.get_residuals_no_fit_triggers(ray_data, x_rot_ray_triggers)
            y_res_x_rots.append(y_sd)
            x_res_x_rots.append(x_sd)
            r_res_x_rots.append(r_mu)
            sum_res_x_rots.append(np.sqrt(x_sd ** 2 + y_sd ** 2))
        # filter out nans
        y_res_x_rots_fit = np.array(y_res_x_rots)
        x_orientations_fit = x_rotations[~np.isnan(y_res_x_rots_fit)]
        y_res_x_rots_fit = y_res_x_rots_fit[~np.isnan(y_res_x_rots_fit)]
        popt_x, pcov_x = cf(quadratic_shift, x_orientations_fit, y_res_x_rots_fit, p0=(-1, 0, 0))
        # x_rot = x_rotations[np.argmin(y_res_x_rots)]
        x_rot_min = popt_x[-1]

        self.replace_last_rotation(y_min, 'y')
        self.convert_cluster_coords()
        y_min_rot_ray_triggers = self.get_close_triggers(ray_data)
        self.replace_last_rotation(y_max, 'y')
        self.convert_cluster_coords()
        y_max_rot_ray_triggers = self.get_close_triggers(ray_data)
        y_rot_ray_triggers = np.intersect1d(y_min_rot_ray_triggers, y_max_rot_ray_triggers)

        x_res_y_rots, y_res_y_rots, r_res_y_rots, sum_res_y_rots = [], [], [], []
        for y_rot_i in y_rotations:
            self.replace_last_rotation(y_rot_i, 'y')
            self.convert_cluster_coords()
            # x_mu, x_sd, y_mu, y_sd = banco_get_residuals(ladder, ray_data, False)
            x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = self.get_residuals_no_fit_triggers(ray_data, y_rot_ray_triggers)
            x_res_y_rots.append(x_sd)
            y_res_y_rots.append(y_sd)
            r_res_y_rots.append(r_sd)
            sum_res_y_rots.append(np.sqrt(x_sd ** 2 + y_sd ** 2))
        # y_rot = y_rotations[np.argmin(x_res_y_rots)]
        y_rot_min = 0

        z0_ray_triggers = self.get_close_triggers(ray_data)
        self.replace_last_rotation(z_min, 'z')
        self.convert_cluster_coords()
        z_min_ray_triggers = self.get_close_triggers(ray_data)
        self.replace_last_rotation(z_max, 'z')
        self.convert_cluster_coords()
        z_max_ray_triggers = self.get_close_triggers(ray_data)
        z_ray_triggers = np.intersect1d(z_min_ray_triggers, z_max_ray_triggers)
        z_ray_triggers = np.intersect1d(z_ray_triggers, z0_ray_triggers)

        x_res_z_rots, y_res_z_rots, r_res_z_rots, sum_res_z_rots = [], [], [], []
        for z_rot_i in z_rotations:
            self.replace_last_rotation(z_rot_i, 'z')
            self.convert_cluster_coords()
            # x_mu, x_sd, y_mu, y_sd = banco_get_residuals(ladder, ray_data, False)
            x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = self.get_residuals_no_fit_triggers(ray_data, z_ray_triggers)
            x_res_z_rots.append(x_sd)
            y_res_z_rots.append(y_sd)
            r_res_z_rots.append(r_sd)
            sum_res_z_rots.append(np.sqrt(x_sd ** 2 + y_sd ** 2))
        z_rot_min = z_rotations[np.argmin(x_res_z_rots)]

        if plot:
            fig_xrot, ax_xrot = plt.subplots()
            x_plot_points = np.linspace(min(x_rotations), max(x_rotations), 1000)
            ax_xrot.plot(x_rotations, y_res_x_rots, color='green', marker='o', label='Y Residuals')
            ax_xrot.plot(x_plot_points, quadratic_shift(x_plot_points, *popt_x), color='r', alpha=0.4)
            # ax_xrot.plot(x_rotations, x_res_x_rots, color='blue', marker='o', label='X Residuals')
            # ax_xrot.plot(x_rotations, r_res_x_rots, color='orange', marker='o', label='R Residuals')
            # ax_xrot.axvline(original_orientation[0], color='g', linestyle='--', label='Original X Rotation')
            ax_xrot.axvline(x_rot_min, color='r', linestyle='--', label='Minimized X Rotation')
            ax_xrot.set_title(f'Y Residuals vs X Rotation {self.name}')
            ax_xrot.set_xlabel('X Rotation (degrees)')
            ax_xrot.set_ylabel('Y Residual Distribution Gaussian Width (mm)')
            ax_xrot.legend()
            fig_xrot.tight_layout()
            fig_xrot.canvas.manager.set_window_title(f'Y Residuals vs X Rotation {self.name}')

            fig_yrot, ax_yrot = plt.subplots()
            ax_yrot.plot(y_rotations, x_res_y_rots, color='blue', marker='o', label='X Residuals')
            # ax_yrot.axvline(original_orientation[1], color='g', linestyle='--', label='Original Y Rotation')
            # ax_yrot.plot(y_rotations, y_res_y_rots, color='green', marker='o', label='Y Residuals')
            # ax_yrot.plot(y_rotations, r_res_y_rots, color='orange', marker='o', label='R Residuals')
            ax_yrot.axvline(y_rot_min, color='r', linestyle='--', label='Minimized Y Rotation')
            ax_yrot.set_title(f'X Residuals vs Y Rotation {self.name}')
            ax_yrot.set_xlabel('Y Rotation (degrees)')
            ax_yrot.set_ylabel('X Residual Distribution Gaussian Width (mm)')
            ax_yrot.legend()
            fig_yrot.tight_layout()
            fig_yrot.canvas.manager.set_window_title(f'X Residuals vs Y Rotation {self.name}')

            fig_zrot, ax_zrot = plt.subplots()
            ax_zrot.plot(z_rotations, x_res_z_rots, color='blue', marker='o', label='X Residuals')
            ax_zrot.plot(z_rotations, y_res_z_rots, color='green', marker='o', label='Y Residuals')
            # ax_zrot.plot(z_rotations, r_res_z_rots, color='orange', marker='o', label='R Residuals')
            ax_zrot.plot(z_rotations, sum_res_z_rots, color='red', marker='o', label='Sum Residuals')
            # ax_zrot.axvline(original_orientation[2], color='g', linestyle='--', label='Original Z Rotation')
            ax_zrot.axvline(z_rot_min, color='r', linestyle='--', label='Minimized Z Rotation')
            ax_zrot.set_title(f'Residuals vs Z Rotation {self.name}')
            ax_zrot.set_xlabel('Z Rotation (degrees)')
            ax_zrot.set_ylabel('Residual Distribution Gaussian Width (mm)')
            ax_zrot.legend()
            fig_zrot.tight_layout()
            fig_zrot.canvas.manager.set_window_title(f'Residuals vs Z Rotation {self.name}')

        self.set_rotations(original_rotations)
        self.convert_cluster_coords()

        return x_rot_min, y_rot_min, z_rot_min

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


def read_banco_file(file_path, event_start=None, event_stop=None):
    field_name = 'fData'
    with uproot.open(file_path) as file:
        tree_name = f"{file.keys()[0].split(';')[0]};{max([int(key.split(';')[-1]) for key in file.keys()])}"
        tree = file[tree_name]
        if event_start is not None and event_stop is not None:
            data = tree[field_name].array(library='np', entry_start=event_start, entry_stop=event_stop)
        elif event_start is not None:
            data = tree[field_name].array(library='np', entry_start=event_start)
        elif event_stop is not None:
            data = tree[field_name].array(library='np', entry_stop=event_stop)
        else:
            data = tree[field_name].array(library='np')
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


def remove_outlying_rays(x_rays, y_rays, event_num_rays, det_size, mult=2.0):
    # Eliminate events in which ray is too far from mean
    x_avg, y_avg = np.mean(x_rays), np.mean(y_rays)
    mask = (x_rays > x_avg - mult * det_size[0]) & (x_rays < x_avg + mult * det_size[0]) & \
           (y_rays > y_avg - mult * det_size[1]) & (y_rays < y_avg + mult * det_size[1])
    x_rays_filter, y_rays_filter = x_rays[mask], y_rays[mask]

    # Iterate once more after outliers are removed
    x_avg, y_avg = np.mean(x_rays_filter), np.mean(y_rays_filter)
    mask = (x_rays > x_avg - mult * det_size[0]) & (x_rays < x_avg + mult * det_size[0]) & \
           (y_rays > y_avg - mult * det_size[1]) & (y_rays < y_avg + mult * det_size[1])
    x_rays_filter, y_rays_filter = x_rays[mask], y_rays[mask]
    event_num_rays = event_num_rays[mask]

    return x_rays_filter, y_rays_filter, event_num_rays


def get_ray_ladder_residuals(x_rays, y_rays, cluster_centroids, plot=False):
    x_rays = np.array(x_rays)
    y_rays = np.array(y_rays)
    cluster_centroids = np.array(cluster_centroids)

    x_residuals = x_rays - cluster_centroids[:, 0]
    y_residuals = y_rays - cluster_centroids[:, 1]

    if plot:
        # Plot a 2D scatter plot of the x, y rays and x, y cluster centroids, with a line connecting the ray to the
        # cluster centroid
        fig, ax = plt.subplots()
        ax.scatter(x_rays, y_rays, color='blue', label='M3 Track', marker='.', alpha=0.5)
        ax.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], color='green', label='Banco Hit', marker='.',
                   alpha=0.5)
        for x_ray, y_ray, x_cent, y_cent in zip(x_rays, y_rays, cluster_centroids[:, 0], cluster_centroids[:, 1]):
            ax.plot([x_ray, x_cent], [y_ray, y_cent], color='red', alpha=0.5)
        ax.set_title('M3 Track vs Banco Centroid Residuals')
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.legend()
        fig.tight_layout()

    return x_residuals, y_residuals


def get_residuals_minimizer(x, ladder, ray_data, ray_triggers):
    """
    Minimize residuals between ray and cluster centroids.
    :param x: Array of z and z rotation
    :param ladder:
    :param ray_data:
    :param ray_triggers:
    :return:
    """
    z, z_rot = x
    ladder.set_center(z=z)
    ladder.replace_last_rotation(z_rot, 'z')
    ladder.convert_cluster_coords()
    x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(z, ray_triggers)

    cluster_centroids, banco_triggers = np.array(ladder.cluster_centroids), np.array(ladder.cluster_triggers) + 1
    cluster_centroids = cluster_centroids[np.isin(banco_triggers, event_num_rays)]

    x_res, y_res = get_ray_ladder_residuals(x_rays, y_rays, cluster_centroids)
    x_res, y_res = np.array(x_res), np.array(y_res)
    r_res = np.sqrt(x_res ** 2 + y_res ** 2)

    return np.mean(r_res)


def quadratic_shift(x, a, c, d):
    return a * (x - d) ** 2 + c
