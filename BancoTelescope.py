#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 17 6:15 PM 2024
Created in PyCharm
Created as saclay_micromegas/BancoTelescope.py

@author: Dylan Neff, Dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf

from DetectorConfigLoader import DetectorConfigLoader
from BancoLadder_new import BancoLadder


class BancoTelescope:
    def __init__(self, det_config_loader=None, sub_run_name=None, data_dir=None, noise_dir=None):
        self.ladders = []
        self.det_config_loader = det_config_loader
        self.sub_run_name = sub_run_name
        self.data_dir = data_dir
        self.noise_dir = noise_dir

        self.four_ladder_triggers = None

        if self.det_config_loader is not None:
            self.load_from_config(self.det_config_loader)

    def get_xy_track_position(self, z_pos, trigger):
        xs, ys, zs = [], [], []
        for ladder in self.ladders:
            x, y, z = ladder.get_cluster_centroid_by_trigger(trigger)
            xs.append(x)
            ys.append(y)
            zs.append(z)

        # Fit xs and yz as a function of z to a line, then plot the points and the lines in two 2D plots and 1 3D plot
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        popt_x, pcov_x = cf(linear, zs, xs)
        popt_y, pcov_y = cf(linear, zs, ys)

        x_fit = linear(z_pos, *popt_x)
        y_fit = linear(z_pos, *popt_y)

        return x_fit, y_fit

    def get_xy_track_positions(self, z_pos, triggers):
        xs, ys = [], []
        for trigger in triggers:
            x, y = self.get_xy_track_position(z_pos, trigger)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def get_all_banco_traversing_triggers(self, ray_data):
        """
        Get the triggers for all events that have a ray traversing any ladder in the Banco telescope.
        :param ray_data:
        :return:
        """
        all_triggers = []
        for ladder in self.ladders:
            all_triggers += list(ladder.get_banco_traversing_triggers(ray_data))
        return np.unique(all_triggers)

    def load_from_config(self, det_config_loader):
        for detector_name in det_config_loader.included_detectors:
            det_config = det_config_loader.get_det_config(detector_name, sub_run_name=self.sub_run_name)
            if det_config['det_type'] == 'banco':
                det = BancoLadder(config=det_config)
                print(f'Loading {det.name}')
                print(f'Center: {det.center}')
                print(f'Size: {det.size}')
                print(f'Rotations: {det.rotations}')
                print(f'Active Size: {det.active_size}')
                self.ladders.append(det)

    def read_data(self, ray_data=None, event_start=None, event_stop=None, filtered=False):
        run_name = get_banco_run_name(self.data_dir)
        for ladder in self.ladders:
            print(f'\nReading data for {ladder.name}')
            data_path = f'{self.data_dir}{run_name}{ladder.ladder_num}.root'
            if filtered:
                data_path = data_path.replace('.root', '_filtered.root')
            noise_path = f'{self.noise_dir}Noise_{ladder.ladder_num}.root'

            banco_traversing_triggers = None
            if ray_data is not None:
                banco_traversing_triggers = ladder.get_banco_traversing_triggers(ray_data)

            print('Reading banco_noise')
            ladder.read_banco_noise(noise_path)
            print('Reading banco_data')
            ladder.read_banco_data(data_path, event_start=event_start, event_stop=event_stop)
            print('Getting data noise pixels')
            ladder.get_data_noise_pixels()
            print('Combining data noise')
            ladder.combine_data_noise()
            print('Clustering data')
            ladder.cluster_data(min_pixels=1, max_pixels=8, chip=None, event_list=banco_traversing_triggers)
            print('Getting largest clusters')
            ladder.get_largest_clusters()
            print('Converting cluster coords')
            ladder.convert_cluster_coords()

    def align_ladders(self, ray_data=None):
        mu = '\u03BC'
        for ladder in self.ladders:
            ladder.align_ladder(ray_data)

        print()
        print(f'Bottom Arm ladder z spacing: {self.ladders[1].center[2] - self.ladders[0].center[2]} mm')
        print(f'Top Arm ladder z spacing: {self.ladders[3].center[2] - self.ladders[2].center[2]} mm')

        # # Combine ladder_cluster_centroids into single dict with trigger_id as key and {ladder: centroid} as value
        # all_trigger_ids = np.unique(np.concatenate([ladder.cluster_triggers for ladder in self.ladders]))
        # all_cluster_centroids = {}
        # for trig_id in all_trigger_ids:
        #     event_ladder_clusters = {}
        #     for ladder in self.ladders:
        #         if trig_id in ladder.cluster_triggers:
        #             event_ladder_clusters[ladder] = ladder.cluster_centroids[
        #                 np.where(ladder.cluster_triggers == trig_id)[0][0]]
        #     all_cluster_centroids[trig_id] = event_ladder_clusters
        #
        # # all_cluster_centroids = self.combine_cluster_centroids()
        #
        # lower_bounds = [ladder.center - ladder.size / 2 for ladder in self.ladders]
        # upper_bounds = [ladder.center + ladder.size / 2 for ladder in self.ladders]
        #
        # residuals, four_ladder_events = {ladder.name: {'x': [], 'y': []} for ladder in self.ladders}, 0
        # self.four_ladder_triggers = []
        # for trig_id, event_clusters in all_cluster_centroids.items():
        #     x, y, z = [], [], []
        #     for ladder, cluster in event_clusters.items():
        #         x.append(cluster[0])
        #         y.append(cluster[1])
        #         z.append(cluster[2])
        #     if len(event_clusters) == 4:
        #         popt_x_inv, pcov_x_inv = cf(linear, z, x)
        #         popt_y_inv, pcov_y_inv = cf(linear, z, y)

        # Precompute a dictionary for each ladder that maps trigger_id to its centroid
        ladder_trigger_centroid_map = {}
        for ladder in self.ladders:
            ladder_trigger_centroid_map[ladder] = {
                trig_id: centroid for trig_id, centroid in zip(ladder.cluster_triggers, ladder.cluster_centroids)
            }

        # Combine ladder_cluster_centroids into a single dict with trigger_id as key and {ladder: centroid} as value
        all_cluster_centroids = {}
        for trig_id in np.unique(np.concatenate([ladder.cluster_triggers for ladder in self.ladders])):
            event_ladder_clusters = {
                ladder: ladder_trigger_centroid_map[ladder][trig_id]
                for ladder in self.ladders if trig_id in ladder_trigger_centroid_map[ladder]
            }
            all_cluster_centroids[trig_id] = event_ladder_clusters

        # Now proceed with calculating residuals, etc.
        residuals, four_ladder_events = {ladder.name: {'x': [], 'y': []} for ladder in self.ladders}, 0
        self.four_ladder_triggers = []
        for trig_id, event_clusters in all_cluster_centroids.items():
            x, y, z = [], [], []
            for ladder, cluster in event_clusters.items():
                x.append(cluster[0])
                y.append(cluster[1])
                z.append(cluster[2])
            if len(event_clusters) == 4:
                popt_x_inv, pcov_x_inv = cf(linear, z, x)
                popt_y_inv, pcov_y_inv = cf(linear, z, y)

                good_event = True
                for ladder, cluster in event_clusters.items():
                    res_x = (cluster[0] - linear(cluster[2], *popt_x_inv)) * 1000
                    res_y = (cluster[1] - linear(cluster[2], *popt_y_inv)) * 1000
                    res_r = np.sqrt(res_x ** 2 + res_y ** 2)
                    if res_r > 100:
                        print(f'Excluding event {trig_id} Ladder {ladder.name} '
                              f'Residuals: X: {res_x:.2f} Y: {res_y:.2f} R: {res_r:.2f}')
                        good_event = False
                if not good_event:
                    continue
                four_ladder_events += 1
                self.four_ladder_triggers.append(trig_id)

                for ladder, cluster in event_clusters.items():
                    residuals[ladder.name]['x'].append((cluster[0] - linear(cluster[2], *popt_x_inv)) * 1000)
                    residuals[ladder.name]['y'].append((cluster[1] - linear(cluster[2], *popt_y_inv)) * 1000)

        for ladder, res in residuals.items():
            print(f'\nLadder {ladder}')
            print(f'X Residuals Mean: {np.mean(res["x"])}')
            print(f'X Residuals Std: {np.std(res["x"])}')
            print(f'Y Residuals Mean: {np.mean(res["y"])}')
            print(f'Y Residuals Std: {np.std(res["y"])}')
            fig_x, ax_x = plt.subplots()
            ax_x.hist(res['x'], bins=np.linspace(min(res['x']), max(res['x']), 25))
            # ax_x.hist(res['x'], bins=np.linspace(np.quantile(res['x'], 0.1), np.quantile(res['x'], 0.9), 25))
            ax_x.set_title(f'X Residuals Ladder {ladder}')
            ax_x.set_xlabel(r'X Residual ($\mu m$)')
            ax_x.set_ylabel('Entries')

            fig_y, ax_y = plt.subplots()
            ax_y.hist(res['y'], bins=np.linspace(min(res['y']), max(res['y']), 25))
            # ax_y.hist(res['y'], bins=np.linspace(np.quantile(res['y'], 0.1), np.quantile(res['y'], 0.9), 25))
            ax_y.set_title(f'Y Residuals Ladder {ladder}')
            ax_y.set_xlabel(r'Y Residual ($\mu m$)')
            ax_y.set_ylabel('Entries')
        print(f'Number of events: {len(all_cluster_centroids)}')
        print(f'Number of events with hits on all 4 ladders {four_ladder_events}')

        iterations, res_widths = np.arange(3), {ladder.name: {'x': [], 'y': [], 'r': []} for ladder in self.ladders}
        for iteration in iterations:
            print(f'Iteration {iteration}')
            residuals = banco_ladder_fit_residuals(self.ladders, self.four_ladder_triggers, False)
            for ladder in self.ladders:
                res_widths[ladder.name]['x'].append(np.std(residuals[ladder.name]['x']))
                res_widths[ladder.name]['y'].append(np.std(residuals[ladder.name]['y']))
                res_widths[ladder.name]['r'].append(np.mean(residuals[ladder.name]['r']))
                x_align = ladder.center[0] - np.mean(residuals[ladder.name]['x']) / 1000
                y_align = ladder.center[1] - np.mean(residuals[ladder.name]['y']) / 1000
                ladder.set_center(x=x_align, y=y_align)
                ladder.convert_cluster_coords()
        for ladder in self.ladders:
            print(f'Ladder {ladder.name} X Residual Width: {res_widths[ladder.name]["x"][-1]:.2f} {mu}m')
            print(f'Ladder {ladder.name} Y Residual Width: {res_widths[ladder.name]["y"][-1]:.2f} {mu}m')
            print(f'Ladder {ladder.name} R Residual Mean: {res_widths[ladder.name]["r"][-1]:.2f} {mu}m')
            fig, ax = plt.subplots()
            ax.plot(iterations, res_widths[ladder.name]['x'], marker='o', label='X')
            ax.plot(iterations, res_widths[ladder.name]['y'], marker='o', label='Y')
            ax.plot(iterations, res_widths[ladder.name]['r'], marker='o', label='R')
            ax.set_title(f'Ladder {ladder.name} Residual Width vs Iteration')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(r'Residual Width ($\mu m$)')
            ax.legend()
            fig.tight_layout()

    def combine_cluster_centroids(self):
        # Combine ladder_cluster_centroids into a single dict with trigger_id as key and {ladder: centroid} as value
        all_trigger_ids = np.unique(np.concatenate([ladder.cluster_triggers for ladder in self.ladders]))
        all_cluster_centroids = {}

        # Sort cluster_triggers for each ladder for faster lookup
        # for ladder in self.ladders:
        #     sorted_idx = np.argsort(ladder.cluster_triggers)
        #     ladder.cluster_triggers = ladder.cluster_triggers[sorted_idx]
        #     ladder.cluster_centroids = ladder.cluster_centroids[sorted_idx]

        # Traverse the sorted cluster_triggers for each ladder using a pointer
        for trig_id in all_trigger_ids:
            event_ladder_clusters = {}

            for ladder in self.ladders:
                cluster_idx = 0
                while cluster_idx < len(ladder.cluster_triggers) and ladder.cluster_triggers[cluster_idx] < trig_id:
                    cluster_idx += 1

                if cluster_idx < len(ladder.cluster_triggers) and ladder.cluster_triggers[cluster_idx] == trig_id:
                    event_ladder_clusters[ladder] = ladder.cluster_centroids[cluster_idx]

            all_cluster_centroids[trig_id] = event_ladder_clusters

        return all_cluster_centroids


def get_banco_run_name(base_dir, start_string='multinoiseScan', end_string='-ladder'):
    """
    Find the name of the banco run from the base directory. Run name starts with start string and ends with end string,
    with the date and time in between. Use the first file containing both strings.
    :param base_dir: Base directory to search for run name.
    :param start_string: String to start run name.
    :param end_string: String to end run name.
    :return: Run name.
    """
    run_name = None
    for file in os.listdir(base_dir):
        if start_string in file and end_string in file:
            # Run name is string up to end string
            run_name = file.split(end_string)[0] + end_string
            break
    return run_name

def banco_ladder_fit_residuals(ladders, triggers, plot=False):
    """
    Fit ladders in each trigger to a line and calculate the residuals on each ladder.
    :param ladders:
    :param triggers:
    :param plot:
    :return:
    """
    residuals = {ladder.name: {'x': [], 'y': [], 'r': []} for ladder in ladders}

    for trigger in triggers:
        x, y, z = [], [], []
        for ladder in ladders:
            cluster = ladder.get_cluster_centroid_by_trigger(trigger)
            x.append(cluster[0])
            y.append(cluster[1])
            z.append(cluster[2])
        popt_x, pcov_x = cf(linear, z, x)
        popt_y, pcov_y = cf(linear, z, y)

        for ladder in ladders:
            cluster = ladder.get_cluster_centroid_by_trigger(trigger)
            x_res = (cluster[0] - linear(cluster[2], *popt_x)) * 1000  # Convert mm to microns
            y_res = (cluster[1] - linear(cluster[2], *popt_y)) * 1000
            r_res = np.sqrt(x_res ** 2 + y_res ** 2)
            residuals[ladder.name]['x'].append(x_res)
            residuals[ladder.name]['y'].append(y_res)
            residuals[ladder.name]['r'].append(r_res)

    if plot:
        for ladder, res in residuals.items():
            print(f'\nLadder {ladder}')
            print(f'X Residuals Mean: {np.mean(res["x"])}')
            print(f'X Residuals Std: {np.std(res["x"])}')
            print(f'Y Residuals Mean: {np.mean(res["y"])}')
            print(f'Y Residuals Std: {np.std(res["y"])}')
            print(f'R Residuals Mean: {np.mean(res["r"])}')
            fig_x, ax_x = plt.subplots()
            fig_y, ax_y = plt.subplots()
            fig_r, ax_r = plt.subplots()

            ax_x.hist(res['x'], bins=np.linspace(min(res['x']), max(res['x']), 25))
            ax_x.hist(res['x'], bins=np.linspace(np.percentile(res['x'], 10), np.percentile(res['x'], 90), 25))
            ax_y.hist(res['y'], bins=np.linspace(min(res['y']), max(res['y']), 25))
            ax_y.hist(res['y'], bins=np.linspace(np.percentile(res['y'], 10), np.percentile(res['y'], 90), 25))
            ax_r.hist(res['r'], bins=np.linspace(0, max(res['r']), 25))
            ax_r.hist(res['r'], bins=np.linspace(0, np.percentile(res['r'], 90), 25))

            ax_x.set_title(f'X Residuals {ladder}')
            ax_x.set_xlabel(r'X Residual ($\mu m$)')
            ax_x.set_ylabel('Entries')
            ax_x.legend()
            ax_y.set_title(f'Y Residuals {ladder}')
            ax_y.set_xlabel(r'Y Residual ($\mu m$)')
            ax_y.set_ylabel('Entries')
            ax_y.legend()
            ax_r.set_title(f'R Residuals {ladder}')
            ax_r.set_xlabel(r'R Residual ($\mu m$)')
            ax_r.set_ylabel('Entries')
            ax_r.legend()

    return residuals

def linear(x, a, b):
    return a * x + b
