#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 03 12:03 PM 2024
Created in PyCharm
Created as saclay_micromegas/banco_read.py

@author: Dylan Neff, Dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit as cf
from scipy.optimize import minimize

import uproot
import awkward as ak
import vector

from cosmic_det_check import get_det_data, get_xy_positions
from BancoLadder import BancoLadder
from M3RefTracking import M3RefTracking


def main():
    # read_decoded_banco()
    # read_raw_banco()
    banco_analysis()
    # get_banco_largest_cluster_npix_dists()
    # banco_noise_analysis()
    # find_cluster_test()
    # convert_row_col_xy_test()
    print('donzo')


def read_decoded_banco():
    vector.register_awkward()
    file_path = 'C:/Users/Dylan/Desktop/banco_test2/fdec.root'
    with uproot.open(file_path) as file:
        print(file.keys())
        tree = file['events;1']
        print(tree.keys())
        data = tree.arrays()
        print(data)
        print(len(data))
        print(data[0])
        print(data[0].fields)
        for field in data[0].fields:
            print(f'Field {field}: {data[0][field]}')
        clust_rows, clust_cols, clust_sizes = [], [], []
        for event in data:
            print(
                f'Event {event["eventId"]} has {len(event["clusters.size"])} clusters of size {event["clusters.size"]}')
            # if len(event['clusters.size']) == 0:
            #     continue
            event_clust_rows = list(event['clusters.rowCentroid'])
            event_clust_cols = list(event['clusters.colCentroid'])
            event_clust_sizes = list(event['clusters.size'])
            bad_i = None
            for i in range(len(event_clust_rows)):
                if event_clust_rows[i] == 102 and event_clust_cols[i] == 3739.5:
                    bad_i = i
            if bad_i is not None:
                event_clust_rows.pop(bad_i)
                event_clust_cols.pop(bad_i)
                event_clust_sizes.pop(bad_i)
            if len(event_clust_rows) == 0:
                continue
            print(f'Event {event["eventId"]} has {len(event_clust_sizes)} clusters of size {event_clust_sizes}')
            clust_rows.append(event_clust_rows[0])
            clust_cols.append(event_clust_cols[0])
            clust_sizes.append(event_clust_sizes[0])
        print(clust_rows)
        print(clust_cols)
        print(clust_sizes)
        print(len(clust_sizes))
        # Histogram of cluster sizes
        fig, ax = plt.subplots()
        plt.hist(clust_sizes)
        plt.title('Cluster Size Histogram')
        plt.xlabel('Cluster Size')
        plt.ylabel('Counts')
        fig.tight_layout()
        # Histogram of cluster rows
        fig, ax = plt.subplots()
        plt.hist(clust_rows)
        plt.title('Cluster Row Histogram')
        plt.xlabel('Cluster Row')
        plt.ylabel('Counts')
        fig.tight_layout()
        # Histogram of cluster cols
        fig, ax = plt.subplots()
        plt.hist(clust_cols)
        plt.title('Cluster Col Histogram')
        plt.xlabel('Cluster Col')
        plt.ylabel('Counts')
        fig.tight_layout()

        plt.show()


def read_raw_banco():
    vector.register_awkward()
    file_path = 'C:/Users/Dylan/Desktop/banco_test2/multinoiseScan_240502_151923-B0-ladder163.root'
    with uproot.open(file_path) as file:
        print(file.keys())
        tree = file['pixTree;2']
        print(tree.keys())
        data = tree.arrays()
        print(data)
        print(len(data))
        print(data[0])
        print(data[0].fields)
        for field in data[0].fields:
            print(f'Field {field}: {data[0][field]}')
            for sub_field in data[0][field].fields:
                print(f'Field {field}.{sub_field}: {data[0][field][sub_field]}')
        print(tree['fData'].array())
        np_array = np.array(tree['fData'].array(library='np'))
        print(np_array)
        print(np_array.shape)
        print(type(np_array))
        print(np_array['trgNum'])
        print(np_array[0])
        print(np_array[0][0])
        print(type(np_array[0]))
        print(type(np_array[0][0]))
    trg_nums, ladder_nums, chip_nums, col_nums, row_nums = [], [], [], [], []
    for event in data:
        trg_nums.append(event['fData']['trgNum'])
        ladder_nums.append(event['fData']['deviceType'])
        chip_nums.append(event['fData']['chipId'])
        col_nums.append(event['fData']['col'])
        row_nums.append(event['fData']['row'])

    print(f'Trigger range: {np.min(trg_nums)} - {np.max(trg_nums)}')
    print(f'Ladder range: {np.min(ladder_nums)} - {np.max(ladder_nums)}')
    print(f'Chip range: {np.min(chip_nums)} - {np.max(chip_nums)}')
    print(f'Column range: {np.min(col_nums)} - {np.max(col_nums)}')
    print(f'Row range: {np.min(row_nums)} - {np.max(row_nums)}')

    hist, bins = np.histogram(trg_nums, bins=np.arange(-0.5, np.max(trg_nums) + 1.5, 1))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots()
    ax.bar(bin_centers, hist, width=1, align='center')
    ax.set_title('Trigger Number Histogram')
    ax.set_xlabel('Trigger Number')
    ax.set_ylabel('Entries')
    fig.tight_layout()

    fig, ax = plt.subplots()
    hist_hist, bins_hist = np.histogram(hist, bins=np.arange(-0.5, np.max(hist) + 1.5, 1))
    bin_centers_hist = (bins_hist[:-1] + bins_hist[1:]) / 2
    ax.bar(bin_centers_hist, hist_hist, width=1, align='center')
    ax.set_title('Trigger Number Histogram Histogram')
    ax.set_xlabel('Entries per Trigger Number')
    ax.set_ylabel('Counts')
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.hist(ladder_nums, bins=np.arange(-0.5, np.max(ladder_nums) + 1.5, 1))
    ax.set_title('Ladder Number Histogram')
    ax.set_xlabel('Ladder Number')
    ax.set_ylabel('Entries')
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.hist(chip_nums, bins=np.arange(-0.5, np.max(chip_nums) + 1.5, 1))
    ax.set_title('Chip Number Histogram')
    ax.set_xlabel('Chip Number')
    ax.set_ylabel('Entries')
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.hist(col_nums, bins=np.arange(-0.5, np.max(col_nums) + 1.5, 1))
    ax.set_title('Column Number Histogram')
    ax.set_xlabel('Column Number')
    ax.set_ylabel('Entries')
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.hist(row_nums, bins=np.arange(-0.5, np.max(row_nums) + 1.5, 1))
    ax.set_title('Row Number Histogram')
    ax.set_xlabel('Row Number')
    ax.set_ylabel('Entries')
    fig.tight_layout()

    plt.show()


def banco_analysis():
    vector.register_awkward()
    # base_dir = 'C:/Users/Dylan/Desktop/banco_test3/'
    base_dir = 'F:/Saclay/banco_data/banco_stats5/'
    det_info_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    # base_dir = '/local/home/dn277127/Bureau/banco_test5/'
    # det_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    run_json_path = f'{base_dir}run_config.json'
    run_name = get_banco_run_name(base_dir)
    run_data = get_det_data(run_json_path)
    # print(run_data)
    banco_names = [det_name for det_name in run_data['included_detectors'] if 'banco' in det_name]
    mu = '\u03BC'

    # ray_data = get_ray_data(base_dir, [0, 1])
    ray_data = M3RefTracking(base_dir, single_track=True)

    ladders, z_rot_angles, n_events = [], [], []
    for banco_name in banco_names:
        det_info = [det for det in run_data['detectors'] if det['name'] == banco_name][0]
        det_type_info = get_det_data(f'{det_info_dir}{det_info["det_type"]}.json')
        det_info.update(det_type_info)
        print()
        print(banco_name)
        print(det_info)
        ladder = BancoLadder(config=det_info)
        ladder_num = int(ladder.name[-3:])
        # if ladder_num == 160:
        #     continue

        z_orig = ladder.center[2]
        x_bnds = ladder.center[0] - ladder.size[0] / 2, ladder.center[0] + ladder.size[0] / 2
        y_bnds = ladder.center[1] - ladder.size[1] / 2, ladder.center[1] + ladder.size[1] / 2
        # for event in ray_data.ray_data:
        #     print(event)
        # print(ray_data.ray_data)
        # print(len(ray_data.ray_data))
        # print(ray_data)
        # print(len(ray_data))
        # input()
        ray_traversing_triggers = ray_data.get_traversing_triggers(z_orig, x_bnds, y_bnds, expansion_factor=0.1)
        banco_traversing_triggers = ray_traversing_triggers - 1  # Rays start at 1, banco starts at 0
        print(f'Number of traversing triggers: {len(ray_traversing_triggers)}')
        print(f'Bounds: x={x_bnds}, y={y_bnds}, z={z_orig}')
        # ray_data.plot_xy(z_orig, ray_traversing_triggers)
        # ray_data.plot_xy(z_orig)
        # plt.show()

        file_path = f'{base_dir}{run_name}{ladder_num}.root'
        noise_path = f'{base_dir}Noise_{ladder_num}.root'
        print('Reading banco_noise')
        ladder.read_banco_noise(noise_path)
        print('Reading banco_data')
        ladder.read_banco_data(file_path)
        print('Getting data noise pixels')
        ladder.get_data_noise_pixels()
        print('Combining data noise')
        ladder.combine_data_noise()
        print('Clustering data')
        ladder.cluster_data(min_pixels=1, max_pixels=8, chip=None, event_list=banco_traversing_triggers)

        # print('\nClusters:')
        # print(ladder.clusters)
        # print('\nAll Cluster Centroids Local Coords:')
        # print(ladder.all_cluster_centroids_local_coords)
        # print('\nTriggers:')
        # print(ladder.cluster_triggers)
        # print(f'\nLen clusters: {len(ladder.clusters)}, len centroids: {len(ladder.all_cluster_centroids_local_coords)},'
        #       f' len triggers: {len(ladder.cluster_triggers)}, len unique triggers: {len(np.unique(ladder.cluster_triggers))}')
        # input()

        print('Getting largest clusters')
        ladder.get_largest_clusters()

        # for trigger, n_pix in zip(ladder.cluster_triggers, ladder.all_cluster_num_pixels):
        #     if len(n_pix) > 1:
        #         plot_event_banco_hits(ladder.data, trigger, ladder.name)
        #         plot_event_banco_hits_coords(ladder, trigger)
        #         plt.show()

        # Manual align
        # if ladder_num in manual_align.keys():
        #     x_align, y_align, z_align = ladder.center
        #     x_align -= manual_align[ladder_num]['x']
        #     y_align -= manual_align[ladder_num]['y']
        #     z_align -= manual_align[ladder_num]['z']
        #     ladder.set_center(x=x_align, y=y_align, z=z_align)

        print('Converting cluster coords')
        ladder.convert_cluster_coords()
        # ladder.plot_largest_cluster_centroids_local_coords()
        # ladder.plot_cluster_centroids()
        # plt.show()

        # ladder.plot_cluster_centroids()

        # ladder.set_orientation(y=0)
        # ladder.convert_cluster_coords()
        # ladder.plot_cluster_centroids()
        # plt.show()

        # ladder.set_orientation(z=45)
        # ladder.convert_cluster_coords()
        # ladder.plot_cluster_centroids()
        #
        # ladder.set_orientation(z=75)
        # ladder.convert_cluster_coords()
        # ladder.plot_cluster_centroids()
        #
        # ladder.set_orientation(z=90)
        # ladder.convert_cluster_coords()
        # ladder.plot_cluster_centroids()
        #
        # ladder.set_orientation(z=120)
        # ladder.convert_cluster_coords()
        # ladder.plot_cluster_centroids()

        # ladder.set_orientation(x=15)
        # ladder.convert_cluster_coords()
        # ladder.plot_cluster_centroids()
        # ladder.set_orientation(y=15)
        # ladder.convert_cluster_coords()
        # ladder.plot_cluster_centroids()
        # plt.show()

        # z_aligned = banco_ref_std_z_alignment(ladder, ray_data, plot=True)
        # plt.show()
        # ladder.set_center(z=z_aligned)
        # ladder.convert_cluster_coords()
        # print(f'Ladder {ladder.name} Center: {ladder.center}')
        # print(ladder.cluster_centroids[:4])
        # input()
        iterations, zs = list(np.arange(10)), []
        # ladder.add_rotation(0, [0, 0, 0])
        z_rot_align = 0
        # good_triggers = get_close_triggers(ladder, ray_data)
        # banco_get_residuals_no_fit_triggers(ladder, ray_data, good_triggers, plot=True)
        for i in iterations:
            print()
            print(f'Iteration {i}: Getting residuals for ladder {ladder_num} with '
                  f'center=[{ladder.center[0]:.2f}, {ladder.center[1]:.2f}, {ladder.center[2]:.2f}] mm, rotations='
                  f'z_rot={z_rot_align:.3f}, {ladder.rotations}')
            zs.append(ladder.center[2])
            # x_res_mean, x_res_sigma, y_res_mean, y_res_sigma = banco_get_residuals(ladder, ray_data, False)
            # x_res_mean, x_res_sigma, y_res_mean, y_res_sigma, r_mu, r_sig = banco_get_residuals_no_fit(ladder, ray_data)
            good_triggers = get_close_triggers(ladder, ray_data)
            x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = banco_get_residuals_no_fit_triggers(ladder, ray_data, good_triggers,
                                                                                     plot=False)
            print(f'Ladder {ladder.name} X Residuals Mean: {x_mu} Sigma: {x_sd}')
            print(f'Ladder {ladder.name} Y Residuals Mean: {y_mu} Sigma: {y_sd}')
            # plt.show()
            aligned_x, aligned_y = ladder.center[0] + x_mu, ladder.center[1] + y_mu
            ladder.set_center(x=aligned_x, y=aligned_y)
            ladder.convert_cluster_coords()

            z_align = banco_res_z_alignment(ladder, ray_data, z_range=(20 / (i + 1)), z_points=200, plot=False)
            ladder.set_center(z=z_align)
            ladder.convert_cluster_coords()

            # x_rot_align, y_rot_align, z_rot_align = banco_align_rotation(ladder, ray_data, plot=False, n_points=100)
            # ladder.replace_last_rotation(z_rot_align, 'z')
            # ladder.convert_cluster_coords()

        # x_rot_align, y_rot_align, z_rot_align = banco_align_rotation(ladder, ray_data, plot=True)
        # print(f'Final rotation: {ladder.rotations}')
        # plt.show()
        good_triggers = get_close_triggers(ladder, ray_data)
        banco_get_residuals_no_fit_triggers(ladder, ray_data, good_triggers, plot=True)
        # plt.show()

        fig, ax = plt.subplots()
        ax.plot(iterations + [iterations[-1] + 1], zs + [ladder.center[2]], marker='o')
        ax.grid(zorder=0)
        ax.set_title(f'Ladder {ladder.name} Z Alignment Iterations')
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

        x_rot_align, y_rot_align, z_rot_align = banco_align_rotation(ladder, ray_data, plot=True)
        ladder.add_rotation(z_rot_align, 'z')
        ladder.convert_cluster_coords()
        good_triggers = get_close_triggers(ladder, ray_data)
        x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = banco_get_residuals_no_fit_triggers(ladder, ray_data, good_triggers,
                                                                                 plot=False)
        print(f'Ladder {ladder.name} X Residuals Mean: {x_mu} Sigma: {x_sd}')
        print(f'Ladder {ladder.name} Y Residuals Mean: {y_mu} Sigma: {y_sd}')
        good_triggers = get_close_triggers(ladder, ray_data)
        banco_get_residuals_no_fit_triggers(ladder, ray_data, good_triggers, plot=True)
        banco_align_rotation(ladder, ray_data, plot=True)
        z_rot_angles.append(z_rot_align)
        n_events.append(len(good_triggers))
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

        ladder.plot_cluster_centroids()
        # plt.show()
        ladders.append(ladder)

    # plt.show()
    print()
    for i, ladder in enumerate(ladders):
        center_string = ', '.join([f'{x:.3f}' for x in ladder.center])
        print(f'Ladder {ladder.name} Center: {center_string}, z_rotation: {z_rot_angles[i]:.4f}, '
              f'n_events: {n_events[i]} Rotations: {ladder.rotations}')

    print(f'Bottom Arm ladder z spacing: {ladders[1].center[2] - ladders[0].center[2]} mm')
    print(f'Top Arm ladder z spacing: {ladders[3].center[2] - ladders[2].center[2]} mm')

    # Combine ladder_cluster_centroids into single dict with trigger_id as key and {ladder: centroid} as value
    all_trigger_ids = np.unique(np.concatenate([ladder.cluster_triggers for ladder in ladders]))
    all_cluster_centroids = {}
    for trig_id in all_trigger_ids:
        event_ladder_clusters = {}
        for ladder in ladders:
            if trig_id in ladder.cluster_triggers:
                event_ladder_clusters[ladder] = ladder.cluster_centroids[
                    np.where(ladder.cluster_triggers == trig_id)[0][0]]
        all_cluster_centroids[trig_id] = event_ladder_clusters

    lower_bounds = [ladder.center - ladder.size / 2 for ladder in ladders]
    upper_bounds = [ladder.center + ladder.size / 2 for ladder in ladders]
    min_x, max_x = min([bound[0] for bound in lower_bounds]), max([bound[0] for bound in upper_bounds])
    min_y, max_y = min([bound[1] for bound in lower_bounds]), max([bound[1] for bound in upper_bounds])

    residuals, four_ladder_events, four_ladder_triggers = {ladder.name: {'x': [], 'y': []} for ladder in ladders}, 0, []
    for trig_id, event_clusters in all_cluster_centroids.items():
        x, y, z = [], [], []
        for ladder, cluster in event_clusters.items():
            x.append(cluster[0])
            y.append(cluster[1])
            z.append(cluster[2])
        if len(event_clusters) == 4:
            # for ladder in ladders:
            #     # plot_event_banco_hits(ladder.data, trig_id, ladder.name)
            #     plot_event_banco_hits_global_coords(ladder, trig_id, x_bounds=(min_x, max_x), y_bounds=(min_y, max_y))

            # popt_x, pcov_x = cf(linear, x, z)
            # popt_y, pcov_y = cf(linear, y, z)

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
            four_ladder_triggers.append(trig_id)

            for ladder, cluster in event_clusters.items():
                residuals[ladder.name]['x'].append((cluster[0] - linear(cluster[2], *popt_x_inv)) * 1000)
                residuals[ladder.name]['y'].append((cluster[1] - linear(cluster[2], *popt_y_inv)) * 1000)

            # fig_x, ax_x = plt.subplots()
            # x_range = np.linspace(min(x), max(x), 100)
            # ax_x.scatter(x, z, color='b')
            # ax_x.plot(x_range, linear(x_range, *popt_x), color='r')
            # ax_x.set_xlim(min_x, max_x)
            # ax_x.set_title(f'X Position vs Ladder for Trigger {trig_id}')
            # ax_x.set_xlabel('X Position (mm)')
            # ax_x.set_ylabel('Ladder Z Position (mm)')
            #
            # fig_y, ax_y = plt.subplots()
            # ax_y.scatter(y, z, color='g')
            # y_range = np.linspace(min(y), max(y), 100)
            # ax_y.plot(y_range, linear(y_range, *popt_y), color='r')
            # ax_y.set_xlim(min_y, max_y)
            # ax_y.set_title(f'Y Position vs Ladder for Trigger {trig_id}')
            # ax_y.set_xlabel('Y Position (mm)')
            # ax_y.set_ylabel('Ladder Z Position (mm)')
            # plt.show()

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

    iterations, res_widths = np.arange(11), {ladder.name: {'x': [], 'y': [], 'r': []} for ladder in ladders}
    for iteration in iterations:
        print(f'Iteration {iteration}')
        residuals = banco_ladder_fit_residuals(ladders, four_ladder_triggers, False)
        for ladder in ladders:
            res_widths[ladder.name]['x'].append(np.std(residuals[ladder.name]['x']))
            res_widths[ladder.name]['y'].append(np.std(residuals[ladder.name]['y']))
            res_widths[ladder.name]['r'].append(np.mean(residuals[ladder.name]['r']))
            x_align = ladder.center[0] - np.mean(residuals[ladder.name]['x']) / 1000
            y_align = ladder.center[1] - np.mean(residuals[ladder.name]['y']) / 1000
            ladder.set_center(x=x_align, y=y_align)
            ladder.convert_cluster_coords()
    for ladder in ladders:
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

    plt.show()


def get_banco_largest_cluster_npix_dists():
    vector.register_awkward()
    base_dir = 'C:/Users/Dylan/Desktop/banco_test3/'
    det_info_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    # base_dir = '/local/home/dn277127/Bureau/banco_test3/'
    # det_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    run_json_path = f'{base_dir}run_config.json'
    run_data = get_det_data(run_json_path)
    print(run_data)
    banco_names = [det_name for det_name in run_data['included_detectors'] if 'banco' in det_name]

    ladders = []
    for banco_name in banco_names:
        det_info = [det for det in run_data['detectors'] if det['name'] == banco_name][0]
        det_type_info = get_det_data(f'{det_info_dir}{det_info["det_type"]}.json')
        det_info.update(det_type_info)
        print(banco_name)
        print(det_info)
        ladder = BancoLadder(config=det_info)
        ladder_num = int(ladder.name[-3:])

        file_path = f'{base_dir}multinoiseScan_240514_231935-B0-ladder{ladder_num}.root'
        noise_path = f'{base_dir}Noise_{ladder_num}.root'
        ladder.read_banco_noise(noise_path)
        ladder.read_banco_data(file_path)
        ladder.get_data_noise_pixels()
        ladder.combine_data_noise()
        # ladder.cluster_data(min_pixels=2)
        ladder.cluster_data(min_pixels=1)
        ladder.get_largest_clusters()
        ladder.convert_cluster_coords()

        for largest_clust_num_pix, trig_id in zip(ladder.largest_cluster_num_pix, ladder.cluster_triggers):
            if largest_clust_num_pix > 10:
                plot_event_banco_hits(ladder.data, trig_id, ladder.name)

        ladders.append(ladder)

    for ladder in ladders:
        fig, ax = plt.subplots()
        max_num_pix = max(ladder.largest_cluster_num_pix)
        counts, bin_edges = np.histogram(ladder.largest_cluster_num_pix, bins=np.arange(-0.5, max_num_pix + 1.5, 1))
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        ax.bar(bin_centers, counts)
        ax.set_title(f'Ladder {ladder.name}')
        fig.tight_layout()

    plt.show()


def banco_noise_analysis():
    vector.register_awkward()
    base_dir = 'C:/Users/Dylan/Desktop/banco_test3/'
    det_info_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    # base_dir = '/local/home/dn277127/Bureau/banco_test3/'
    # det_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    run_json_path = f'{base_dir}run_config.json'
    run_data = get_det_data(run_json_path)
    print(run_data)
    banco_names = [det_name for det_name in run_data['included_detectors'] if 'banco' in det_name]

    for banco_name in banco_names:
        det_info = [det for det in run_data['detectors'] if det['name'] == banco_name][0]
        det_type_info = get_det_data(f'{det_info_dir}{det_info["det_type"]}.json')
        det_info.update(det_type_info)
        print(banco_name)
        print(det_info)
        ladder = BancoLadder(config=det_info)
        ladder_num = int(ladder.name[-3:])

        file_path = f'{base_dir}multinoiseScan_240514_231935-B0-ladder{ladder_num}.root'
        noise_path = f'{base_dir}Noise_{ladder_num}.root'
        ladder.read_banco_noise(noise_path)
        ladder.read_banco_data(file_path)
        ladder.get_data_noise_pixels()
        ladder.combine_data_noise()

        plot_noise_vs_threshold(ladder.data, default_threshold=2, ladder=ladder_num)

    plt.show()


# def get_ray_ladder_residuals_old(x_rays, y_rays, cluster_centroids):
#     x_residuals, y_residuals = [], []
#     for x, y, centroid in zip(x_rays, y_rays, cluster_centroids):
#         x_centroid, y_centroid, z_centroid = centroid
#         x_residuals.append(x - x_centroid)
#         y_residuals.append(y - y_centroid)
#     return x_residuals, y_residuals


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


def compare_rays_to_ladder(x_rays, y_rays, event_num_rays, cluster_centroids, trigger_ids, plot=False, ladder=None,
                           cluster_num_pix=None):
    mask = np.isin(event_num_rays, trigger_ids + 1)

    # Filter x_rays and y_rays based on the mask
    filtered_xs = x_rays[mask]
    filtered_ys = y_rays[mask]
    filtered_event_num_rays = event_num_rays[mask]

    # Create empty lists to store the results
    xs, ys = [], []

    # Iterate over the filtered arrays
    x_residuals, y_residuals = [], []
    for x, y, ray_trigger_id in zip(filtered_xs, filtered_ys, filtered_event_num_rays):
        clust_index = np.where(trigger_ids + 1 == ray_trigger_id)[0][0]
        event_clusters = cluster_centroids[clust_index]
        # print(f'Event clusters: {event_clusters}')
        if len(event_clusters) > 0:
            # print(f'Event# {ray_trigger_id}: ray x={x}, y={y}; banco: {event_clusters}')
            xs.append(x)
            ys.append(y)
            if cluster_num_pix is not None:
                # Get cluster with largest number of pixels
                max_pix_i = np.argmax(cluster_num_pix[clust_index])
                event_max_cluster = event_clusters[max_pix_i]
                x_residuals.append(x - event_max_cluster[0])
                y_residuals.append(y - event_max_cluster[1])

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(xs, ys, alpha=0.5)
        title = 'Ref Track Detector Hits' if ladder is None else f'Ref Track Detector Hits Ladder {ladder}'
        ax.grid(zorder=0)
        ax.set_title(title)
        fig.tight_layout()

        fig, ax = plt.subplots()
        x_res_std = np.std(x_residuals)
        print(f'X Residuals std: {x_res_std}')
        ax.hist(x_residuals, bins=np.linspace(min(x_residuals), max(x_residuals), 25))
        title = 'X Residuals' if ladder is None else f'X Residuals Ladder {ladder}'
        ax.set_title(title)
        ax.set_xlabel('X Residual (mm)')
        ax.set_ylabel('Entries')

        fig, ax = plt.subplots()
        y_res_std = np.std(y_residuals)
        print(f'Y Residuals std: {y_res_std}')
        ax.hist(y_residuals, bins=np.linspace(min(y_residuals), max(y_residuals), 25))
        title = 'Y Residuals' if ladder is None else f'Y Residuals Ladder {ladder}'
        ax.set_title(title)
        ax.set_xlabel('Y Residual (mm)')
        ax.set_ylabel('Entries')

    # Get standard deviation of the middle 80% of the data
    xs = np.array(xs)
    x_min, x_max = np.percentile(xs, 10), np.percentile(xs, 90)
    xs = xs[(xs > x_min) & (xs < x_max)]
    ys = np.array(ys)
    y_min, y_max = np.percentile(ys, 2), np.percentile(ys, 98)
    ys = ys[(ys > y_min) & (ys < y_max)]

    x_res, y_res = None, None
    if len(x_residuals) > 0 and len(y_residuals) > 0:
        x_min, x_max = np.percentile(x_residuals, 10), np.percentile(x_residuals, 90)
        x_reses = np.array(x_residuals)
        x_reses = []

    return np.std(xs), np.std(ys)


def plot_cluster_scatter(cluster_centroids, title=None):
    fig, ax = plt.subplots()
    clusters = []
    for cluster in cluster_centroids:
        clusters.extend(cluster)
    clusters = np.array(clusters)
    ax.scatter(clusters[1], clusters[0], alpha=0.5)
    title = f'Clusters' if title is None else f'Clusters Ladder {title}'
    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    fig.tight_layout()


def plot_cluster_number_histogram(cluster_centroids, ladder=None):
    fig, ax = plt.subplots()
    n_clusters = [len(clusters) for clusters in cluster_centroids]
    ax.hist(n_clusters, bins=np.arange(-0.5, max(n_clusters) + 1.5, 1))
    title = f'Cluster Number Histogram' if ladder is None else f'Cluster Number Histogram Ladder {ladder}'
    ax.set_title(title)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Entries')
    ax.set_yscale('log')
    fig.tight_layout()


def plot_num_pixels_histogram(cluster_num_pixels, ladder):
    fig, ax = plt.subplots()
    # print(cluster_num_pixels)
    # print(type(cluster_num_pixels))
    num_pix = []
    for event in cluster_num_pixels:  # This is what happens when I don't have copilot
        num_pix.extend(event)
    num_pix = np.array(num_pix)
    ax.hist(num_pix, bins=np.arange(-0.5, max(num_pix) + 1.5, 1))
    title = f'Cluster Pixel Size' if ladder is None else f'Cluster Number Histogram Ladder {ladder}'
    ax.set_title(title)
    ax.set_xlabel('Number of Pixels in Cluster')
    ax.set_ylabel('Entries')
    ax.set_yscale('log')
    fig.tight_layout()


def plot_noise_vs_threshold(data, default_threshold=None, ladder=None):
    """
    Get the number of noisy pixels for each threshold.
    :param data: Data from noise run, ideally no signal. Shape (triggers, rows, cols)
    :param default_threshold: Default threshold to use if not specified.
    :param ladder: Ladder number for title.
    :return: Array of noisy pixels for each threshold. Shape (thresholds, rows, cols)
    """
    triggers, rows_cols = data[:, 0], data[:, 1:]
    num_triggers = np.unique(triggers).size
    noise_pixels, counts = np.unique(rows_cols, return_counts=True, axis=0)
    num_noise = []
    thresholds = np.arange(1, 1000, 1)
    for threshold in thresholds:
        num_noise.append(len(noise_pixels[counts >= threshold]))

    rows_cols_denoised = None
    if default_threshold is not None:  # Remove noisy pixels from rows_columns data
        if default_threshold >= 1:
            def_noise_pixels = noise_pixels[counts >= default_threshold]
        elif default_threshold < 1:
            def_noise_pixels = noise_pixels[counts >= default_threshold * num_triggers]
        # Remove 2d noise pixels from data
        rows_cols_denoised = rows_cols
        for noise_pixel in def_noise_pixels:
            rows_cols_denoised = rows_cols_denoised[~np.all(rows_cols_denoised == noise_pixel, axis=1)]
        rows_cols_denoised = np.array(rows_cols_denoised)

    title_suffix = '' if ladder is None else f' Ladder {ladder}'

    fig, ax = plt.subplots()
    bins = np.concatenate([np.arange(-0.5, 11.5, 1), np.linspace(11.5, np.max(counts) + 1.5, 10)])
    ax.hist(counts, bins=bins)
    ax.set_title('Noise Pixel Counts' + title_suffix)
    ax.set_xlabel('Number of Triggers')
    ax.set_ylabel('Number of Pixels')
    fig.tight_layout()

    # rows, row_counts = np.unique(data[:, 1], return_counts=True)
    rows = np.arange(0, np.max(data[:, 1]) + 1, 1)
    row_counts = np.array([np.count_nonzero(data[:, 1] == row) for row in rows])
    fig, ax = plt.subplots()
    ax.plot(rows, row_counts, label='Before Denoising')
    if rows_cols_denoised is not None:
        denoised_row_counts = np.array([np.count_nonzero(rows_cols_denoised[:, 0] == row) for row in rows])
        ax.plot(rows, denoised_row_counts, label='After Denoising')
        ax.legend()
    ax.set_title('Row Noise Pixel Counts' + title_suffix)
    ax.set_xlabel('Row Number')
    ax.set_ylabel('Total Hits in Row')
    ax.set_yscale('log')
    fig.tight_layout()

    # cols, col_counts = np.unique(data[:, 2], return_counts=True)
    cols = np.arange(0, np.max(data[:, 2]) + 1, 1)
    col_counts = np.array([np.count_nonzero(data[:, 2] == col) for col in cols])
    fig, ax = plt.subplots()
    ax.plot(cols, col_counts, label='Before Denoising')
    if rows_cols_denoised is not None:
        denoised_col_counts = np.array([np.count_nonzero(rows_cols_denoised[:, 1] == col) for col in cols])
        ax.plot(cols, denoised_col_counts, label='After Denoising')
        ax.legend()
    ax.set_title('Column Noise Pixel Counts' + title_suffix)
    ax.set_xlabel('Column Number')
    ax.set_ylabel('Total Hits in Column')
    ax.set_yscale('log')
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.plot(thresholds, num_noise)
    if default_threshold is not None and default_threshold >= 1:
        ax.axvline(default_threshold, color='red', linestyle='--')
    ax.set_title('Noise Pixels vs Threshold' + title_suffix)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Noise Pixels')
    ax.grid(zorder=0)
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.plot(thresholds / num_triggers, num_noise)
    if default_threshold is not None and default_threshold < 1:
        ax.axvline(default_threshold, color='red', linestyle='--')
    ax.set_title('Noise Pixels vs Threshold' + title_suffix)
    ax.set_xlabel('Threshold Fraction')
    ax.set_ylabel('Noise Pixels')
    ax.set_xticklabels([f'{x * 100:.1f}%' for x in ax.get_xticks()])
    ax.grid(zorder=0)


def plot_event_banco_hits(data, trigger_id, ladder=None):
    event = data[data[:, 0] == trigger_id]
    fig, ax = plt.subplots()
    ax.scatter(event[:, 2], event[:, 1], alpha=0.5)
    ax.grid(zorder=0)
    title = f'Event {trigger_id} Banco Hits' if ladder is None else f'Event {trigger_id} Banco Hits Ladder {ladder}'
    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_xlim(0, 1024 * 5)
    ax.set_ylabel('Row')
    ax.set_ylim(0, 512)
    fig.tight_layout()


def plot_event_banco_largest_hit_coords(ladder, trigger_id):
    event_cluster_centroid = ladder.cluster_centroids[np.where(ladder.cluster_triggers == trigger_id)[0][0]]
    event_cluster_size = ladder.largest_cluster_num_pix[np.where(ladder.cluster_triggers == trigger_id)[0][0]]
    fig, ax = plt.subplots()
    ax.scatter([event_cluster_centroid[0]], [event_cluster_centroid[1]], s=event_cluster_size * 10, color='g', zorder=1)
    # Write the event_cluster_sizes inside each circle
    # for i, (x, y, z) in enumerate(event_cluster_centroids):
    #     ax.text(x, y, event_cluster_sizes[i], color='w', ha='center', va='center')
    ax.set_xlim(ladder.center[0] - ladder.size[0] / 2, ladder.center[0] + ladder.size[0] / 2)
    ax.set_ylim(ladder.center[1] - ladder.size[1] / 2, ladder.center[1] + ladder.size[1] / 2)
    ax.grid(zorder=0)
    ax.set_title(f'Event {trigger_id} Banco Hits Global Coordinates Ladder {ladder.name}')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    fig.tight_layout()


def plot_event_banco_hits_global_coords(ladder, trigger_id, x_bounds=None, y_bounds=None):
    cluster_centroids = ladder.get_cluster_centroids_global_coords()

    event_cluster_centroids = cluster_centroids[np.where(ladder.cluster_triggers == trigger_id)[0][0]]
    event_cluster_sizes = ladder.all_cluster_num_pixels[np.where(ladder.cluster_triggers == trigger_id)[0][0]]
    event_cluster_centroids = np.array(event_cluster_centroids)
    event_cluster_sizes = np.array(event_cluster_sizes)
    fig, ax = plt.subplots()
    ax.scatter(event_cluster_centroids[:, 0], event_cluster_centroids[:, 1], s=event_cluster_sizes * 150, color='g',
               alpha=0.8, zorder=1)
    # Write the event_cluster_sizes inside each circle
    for i, (x, y, z) in enumerate(event_cluster_centroids):
        ax.text(x, y, event_cluster_sizes[i], color='white', ha='center', va='center')
    ax.grid(zorder=0)
    if x_bounds is not None:
        ax.set_xlim(x_bounds)
        ax.axvline(ladder.center[0] - ladder.size[0] / 2, color='gray', linestyle='-')
        ax.axvline(ladder.center[0] + ladder.size[0] / 2, color='gray', linestyle='-')
    else:
        ax.set_xlim(ladder.center[0] - ladder.size[0] / 2, ladder.center[0] + ladder.size[0] / 2)
    if y_bounds is not None:
        ax.set_ylim(y_bounds)
        ax.axhline(ladder.center[1] - ladder.size[1] / 2, color='gray', linestyle='-')
        ax.axhline(ladder.center[1] + ladder.size[1] / 2, color='gray', linestyle='-')
    else:
        ax.set_ylim(ladder.center[1] - ladder.size[1] / 2, ladder.center[1] + ladder.size[1] / 2)

    ax.set_title(f'Event {trigger_id} Banco Hits Global Coordinates Ladder {ladder.name}')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    fig.tight_layout()


def std_align(z0, ray_data, cluster_centroids, trigger_ids, ladder, plot=True):
    zs = np.linspace(z0 - 20, z0 + 20, 5)
    x_stds, y_stds = [], []
    for zi in zs:
        x_rays, y_rays, event_num_rays = get_xy_positions(ray_data, zi)
        x_std, y_std = compare_rays_to_ladder(x_rays, y_rays, event_num_rays, cluster_centroids, np.array(trigger_ids))
        x_stds.append(x_std)
        y_stds.append(y_std)
    z_fit, popts = [], []
    for std_name, stds in zip(['X Std', 'Y Std'], [x_stds, y_stds]):
        p0 = (-1, 4, z0)
        popt, pcov = cf(quadratic_shift, zs, stds, p0=p0)
        print(f'{std_name} Guess: {p0} Fit: {popt}')
        popts.append(popt)
        # Solve for z of minimum x_std
        # z1 = -popt[1] / (2 * popt[0])
        z1 = popt[-1]
        z_fit.append(z1)
        if plot:
            fig, ax = plt.subplots()
            ax.plot(zs, stds, marker='o')
            plot_z_range = np.linspace(min(zs), max(zs), 100)
            ax.plot(plot_z_range, quadratic_shift(plot_z_range, *popt), color='r', alpha=0.4)
            ax.axvline(z0, color='r', linestyle='--', label='Measured z')
            ax.axvline(z1, color='g', linestyle='--', label='Minimized z')
            ax.grid(zorder=0)
            ax.legend()
            ax.set_title(f'Ladder {ladder} {std_name} vs Z')
            ax.set_xlabel('Z (mm)')
            ax.set_ylabel(f'{std_name} (mm)')
            fig.tight_layout()
    # p0 = (popts[0][0], popts[1][0], 0, popts[0][1], popts[1][1], np.mean(popts[0][2] + popts[1][2]))
    # popt, pcov = cf(quadratic_2d, (x_stds, y_stds), zs, p0=p0)
    # z_fit.append(z_min_xy)

    return z_fit


def banco_ref_std_z_alignment(ladder, ray_data, plot=True, z_align_range=20, z_align_points=10):
    # Get only rays with trigger ids matching ladder triggers
    ray_trigger_ids = np.array(ladder.cluster_triggers) + 1  # Banco starts at 0, rays start at 1

    # First do rough z alignment by minimizing width of ray distribution
    z0 = ladder.center[2]

    # Eliminate events in which ray is too far from mean
    x_rays, y_rays, event_num_rays = get_xy_positions(ray_data, z0, ray_trigger_ids)
    x_rays_filter, y_rays_filter, event_num_rays = remove_outlying_rays(x_rays, y_rays, event_num_rays, ladder.size,
                                                                        1.2)

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(x_rays, y_rays, alpha=0.5)
        ax.scatter(x_rays_filter, y_rays_filter, alpha=0.5)
        ax.scatter([np.mean(x_rays_filter)], [np.mean(y_rays_filter)], marker='x', color='r')
        title = f'Ref Track Detector Hits Ladder {ladder.name}, z={z0}'
        ax.grid(zorder=0)
        ax.set_title(title)
        fig.tight_layout()

        fig, ax = plt.subplots()
        x_banco, y_banco = ladder.cluster_centroids[:, :2].T
        ax.scatter(x_banco, y_banco, alpha=0.5)

    zs = np.linspace(z0 - z_align_range / 2, z0 + z_align_range / 2, z_align_points)
    x_stds, y_stds, x_sigmas, y_sigmas, x_residuals, y_residuals = [], [], [], [], [], []
    for zi in zs:
        x_rays, y_rays, event_num_rays = get_xy_positions(ray_data, zi, event_num_rays)
        # x_counts, x_bins = np.histogram(x_rays, bins=np.linspace(min(x_rays), max(x_rays), 25))
        # x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
        # y_counts, y_bins = np.histogram(y_rays, bins=np.linspace(min(y_rays), max(y_rays), 25))
        # y_bin_centers = (y_bins[:-1] + y_bins[1:]) / 2
        # popt_x, pcov_x = cf(gaus_amp, x_bin_centers, x_counts, p0=(1, np.mean(x_rays), np.std(x_rays)))
        # popt_y, pcov_y = cf(gaus_amp, y_bin_centers, y_counts, p0=(1, np.mean(y_rays), np.std(y_rays)))
        # x_plot = np.linspace(min(x_rays), max(x_rays), 1000)
        # y_plot = np.linspace(min(y_rays), max(y_rays), 1000)
        # if plot:
        #     fig_x, ax_x = plt.subplots()
        #     ax_x.bar(x_bin_centers, x_counts, width=x_bin_centers[1] - x_bin_centers[0])
        #     ax_x.plot(x_plot, gaus_amp(x_plot, *popt_x), color='r')
        #     ax_x.set_title(f'X Ray Distribution z={zi}')
        #     ax_x.set_xlabel('X Position (mm)')
        #     ax_x.set_ylabel('Entries')
        #     fig_x.tight_layout()
        #
        #     fig_y, ax_y = plt.subplots()
        #     ax_y.bar(y_bin_centers, y_counts, width=y_bin_centers[1] - y_bin_centers[0])
        #     ax_y.plot(y_plot, gaus_amp(y_plot, *popt_y), color='r')
        #     ax_y.set_title(f'Y Ray Distribution z={zi}')
        #     ax_y.set_xlabel('Y Position (mm)')
        #     ax_y.set_ylabel('Entries')
        #     fig_y.tight_layout()

        x_std, y_std = np.std(x_rays), np.std(y_rays)
        x_stds.append(x_std)
        y_stds.append(y_std)
        # if plot:
        #     print(f'Z: {zi} X Std: {x_std} Y Std: {y_std}')
        #     print(f'X Fit_sigma: {popt_x[-1]} Y Fit Sigma: {popt_y[-1]}')
        # x_sigmas.append(popt_x[-1])
        # y_sigmas.append(popt_y[-1])

    z_fit, popts = [], []
    for std_name, stds in zip(['X Std', 'Y Std'], [x_stds, y_stds]):
        p0 = (-1, 4, z0)
        popt, pcov = cf(quadratic_shift, zs, stds, p0=p0)
        # print(f'{std_name} Guess: {p0} Fit: {popt}')
        popts.append(popt)
        z1 = popt[-1]
        z_fit.append(z1)
        if plot:
            fig, ax = plt.subplots()
            ax.plot(zs, stds, marker='o', label='Measured Std')
            # ax.plot(zs, sigmas, marker='o', label='Fit Sigma')
            plot_z_range = np.linspace(min(zs), max(zs), 100)
            ax.plot(plot_z_range, quadratic_shift(plot_z_range, *popt), color='r', alpha=0.4)
            ax.axvline(z0, color='r', linestyle='--', label='Measured z')
            ax.axvline(z1, color='g', linestyle='--', label='Minimized z')
            ax.grid(zorder=0)
            ax.legend()
            ax.set_title(f'Ladder {ladder.name} {std_name} vs Z')
            ax.set_xlabel('Z (mm)')
            ax.set_ylabel(f'{std_name} (mm)')
            fig.tight_layout()

    return z_fit[0]


def banco_res_z_alignment(ladder, ray_data, z_range=20., z_points=20, plot=True):
    """
    Align ladder by minimizing residuals between ray and cluster centroids
    :param ladder:
    :param ray_data:
    :param z_range:
    :param z_points:
    :param plot:
    :return:
    """
    original_z = ladder.center[2]
    zs = np.linspace(original_z - z_range / 2, original_z + z_range / 2, z_points)

    ladder.set_center(z=min(zs))
    z_min_ray_triggers = get_close_triggers(ladder, ray_data)
    ladder.set_center(z=max(zs))
    z_max_ray_triggers = get_close_triggers(ladder, ray_data)
    common_triggers = np.intersect1d(z_min_ray_triggers, z_max_ray_triggers)

    x_res_widths, y_res_widths, sum_res_widths = [], [], []
    for zi in zs:
        ladder.set_center(z=zi)
        ladder.convert_cluster_coords()
        # x_res_mean, x_res_sigma, y_res_mean, y_res_sigma = banco_get_residuals(ladder, ray_data, plot=False)
        x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = banco_get_residuals_no_fit_triggers(ladder, ray_data, common_triggers,
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
        ax.set_title(f'X Residuals vs Z {ladder.name}')
        ax.set_xlabel('Z (mm)')
        ax.set_ylabel('X Residual Distribution Gaussian Width (mm)')
        ax.legend()
        fig.tight_layout()

    return z_min_sum


def banco_xyz_alignment(ladder, ray_data, plot=True):
    """
    Align spatial x, y, z simultaneously of ladder by minimizing residuals between ray and cluster centroids
    :param ladder:
    :param ray_data:
    :param plot:
    :return:
    """
    original_center = ladder.center
    x0, y0, z0 = original_center
    xs = np.linspace(x0 - 10, x0 + 10, 20)
    ys = np.linspace(y0 - 10, y0 + 10, 20)
    zs = np.linspace(z0 - 10, z0 + 10, 20)
    x_res_widths, y_res_widths, sum_res_widths = [], [], []
    for xi in xs:
        for yi in ys:
            for zi in zs:
                ladder.set_center(x=xi, y=yi, z=zi)
                x_res_mean, x_res_sigma, y_res_mean, y_res_sigma = banco_get_residuals(ladder, ray_data, plot=False)
                x_res_widths.append(x_res_sigma)
                y_res_widths.append(y_res_sigma)
                sum_res_widths.append(np.sqrt(x_res_sigma ** 2 + y_res_sigma ** 2))

    z_min_sum = zs[np.argmin(sum_res_widths)]

    if plot:
        # Converting the lists into numpy arrays for easier manipulation
        sum_res_widths = np.array(sum_res_widths)

        # Create a meshgrid for plotting
        X, Y, Z = np.meshgrid(xs, ys, zs)
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()

        # Find the index of the minimum sum_res_widths value
        min_index = np.argmin(sum_res_widths)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        sc = ax.scatter(X, Y, Z, c=sum_res_widths, cmap='viridis', marker='o')

        # Marking the smallest value with a red star
        ax.scatter(X[min_index], Y[min_index], Z[min_index], color='red', s=100, label='Min sum_res_widths')

        # Adding color bar for the sum_res_widths
        cb = plt.colorbar(sc)
        cb.set_label('Sum of Residual Widths')

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Scatter Plot of Residual Widths')

        fig.tight_layout()


def banco_get_pixel_spacing(ladder, ray_data, plot=True):
    """
    Vary pixel spacing of detector to minimize residual widths between ray and cluster centroids
    :param ladder:
    :param ray_data:
    :param plot:
    :return:
    """
    # Minimize x residuals varying x spacing
    original_pitch_x = ladder.pitch_x
    pitch_xs = np.linspace(ladder.pitch_x - 4, ladder.pitch_x + 4, 100)
    x_res_widths = []
    for pitch_x in pitch_xs:
        ladder.set_pitch_x(pitch_x)
        ladder.get_cluster_centroids()
        ladder.get_largest_clusters()
        ladder.convert_cluster_coords()
        x_res_widths.append(banco_get_residuals(ladder, ray_data, False)[1])

    original_pitch_y = ladder.pitch_y
    original_chip_space = ladder.chip_space
    pitch_ys = np.linspace(ladder.pitch_y - 0.5, ladder.pitch_y + 0.5, 100)
    chip_spacings = np.linspace(0, 100, 20)

    y_res_widths_chip_spaces = []
    for chip_space in chip_spacings:
        y_res_widths = []
        for pitch_y in pitch_ys:
            ladder.set_pitch_y(pitch_y)
            ladder.set_chip_space(chip_space)
            ladder.get_cluster_centroids()
            ladder.get_largest_clusters()
            ladder.convert_cluster_coords()
            y_res_widths.append(banco_get_residuals(ladder, ray_data, False)[3])
        y_res_widths_chip_spaces.append(y_res_widths)

    ladder.set_pitch_x(original_pitch_x)
    ladder.set_pitch_y(original_pitch_y)
    ladder.set_chip_space(original_chip_space)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(pitch_xs, x_res_widths, marker='o')
        ax.axvline(original_pitch_x, color='g', linestyle='--')
        ax.set_title(f'X Residuals vs X Pitch {ladder.name}')
        ax.set_xlabel(r'X Pitch ($\mu m$)')
        ax.set_ylabel(r'X Residual Distribution Gaussian Width ($\mu m$)')
        fig.tight_layout()

        fig_y, ax_y = plt.subplots()
        for chip_space, y_res_widths in zip(chip_spacings, y_res_widths_chip_spaces):
            ax_y.plot(pitch_ys, y_res_widths, marker='o', label=f'Chip Space {chip_space:.1f}')
        ax_y.axvline(original_pitch_y, color='g', linestyle='--')
        ax_y.set_title(f'Y Residuals vs Y Pitch {ladder.name}')
        ax_y.set_xlabel(r'Y Pitch ($\mu m$)')
        ax_y.set_ylabel(r'Y Residual Distribution Gaussian Width ($\mu m$)')
        ax_y.legend()
        fig_y.tight_layout()

        fig_space, ax_space = plt.subplots()
        transposed_y_res_spaces = np.array(y_res_widths_chip_spaces).transpose()
        for i, pitch_y in enumerate(pitch_ys):
            if i % 10 == 0:
                ax_space.plot(chip_spacings, transposed_y_res_spaces[i], marker='o', label=f'Pitch {pitch_y:.2f}')
        ax_space.axvline(original_chip_space, color='g', linestyle='--')
        ax_space.set_title(f'Y Residuals vs Chip Space {ladder.name}')
        ax_space.set_xlabel(r'Chip Space ($\mu m$)')
        ax_space.set_ylabel(r'Y Residual Distribution Gaussian Width ($\mu m$)')
        ax_space.legend()
        fig_space.tight_layout()


# def banco_align_rotation_old(ladder, ray_data, plot=True):
#     """
#     Align ladder by minimizing residuals between ray and cluster centroids
#     :param ladder:
#     :param ray_data:
#     :param plot:
#     :return:
#     """
#
#     original_orientation = ladder.orientation
#     x_rot, y_rot, z_rot = original_orientation
#     x_orientations = np.linspace(x_rot - 8, x_rot + 8, 1000)
#     y_orientations = np.linspace(y_rot - 8, y_rot + 8, 1000)
#     z_orientations = np.linspace(z_rot - 8, z_rot + 8, 1000)
#
#     y_res_x_rots = []
#     for x_rot_i in x_orientations:
#         ladder.set_orientation(x_rot_i, y_rot, z_rot)
#         # ladder.get_cluster_centroids()
#         # ladder.get_largest_clusters()
#         ladder.convert_cluster_coords()
#         y_res_x_rots.append(banco_get_residuals(ladder, ray_data, False)[3])
#     # filter out nans
#     y_res_x_rots_fit = np.array(y_res_x_rots)
#     x_orientations_fit = x_orientations[~np.isnan(y_res_x_rots_fit)]
#     y_res_x_rots_fit = y_res_x_rots_fit[~np.isnan(y_res_x_rots_fit)]
#     popt_x, pcov_x = cf(quadratic_shift, x_orientations_fit, y_res_x_rots_fit, p0=(1, 1, x_rot))
#     # x_rot = x_orientations[np.argmin(y_res_x_rots)]
#     x_rot_min = popt_x[-1]
#
#     x_res_y_rots = []
#     for y_rot_i in y_orientations:
#         ladder.set_orientation(x_rot, y_rot_i, z_rot)
#         # ladder.get_cluster_centroids()
#         # ladder.get_largest_clusters()
#         ladder.convert_cluster_coords()
#         x_res_y_rots.append(banco_get_residuals(ladder, ray_data, False)[1])
#     # y_rot = y_orientations[np.argmin(x_res_y_rots)]
#     y_rot_min = y_rot
#
#     x_res_z_rots, y_res_z_rots = [], []
#     for z_rot_i in z_orientations:
#         ladder.set_orientation(x_rot, y_rot, z_rot_i)
#         # ladder.get_cluster_centroids()
#         # ladder.get_largest_clusters()
#         ladder.convert_cluster_coords()
#         x_mu, x_sd, y_mu, y_sd = banco_get_residuals(ladder, ray_data, False)
#         x_res_z_rots.append(x_sd)
#         y_res_z_rots.append(y_sd)
#     z_rot_min = z_rot
#
#     if plot:
#         fig_xrot, ax_xrot = plt.subplots()
#         x_plot_points = np.linspace(min(x_orientations), max(x_orientations), 1000)
#         ax_xrot.plot(x_orientations, y_res_x_rots, marker='o')
#         ax_xrot.plot(x_plot_points, quadratic_shift(x_plot_points, *popt_x), color='r', alpha=0.4)
#         ax_xrot.axvline(original_orientation[0], color='g', linestyle='--', label='Original X Rotation')
#         ax_xrot.axvline(x_rot_min, color='r', linestyle='--', label='Minimized X Rotation')
#         ax_xrot.set_title(f'Y Residuals vs X Rotation {ladder.name}')
#         ax_xrot.set_xlabel('X Rotation (degrees)')
#         ax_xrot.set_ylabel('Y Residual Distribution Gaussian Width (mm)')
#         fig_xrot.tight_layout()
#         fig_xrot.canvas.manager.set_window_title(f'Y Residuals vs X Rotation {ladder.name}')
#
#         fig_yrot, ax_yrot = plt.subplots()
#         ax_yrot.plot(y_orientations, x_res_y_rots, marker='o')
#         ax_yrot.axvline(original_orientation[1], color='g', linestyle='--', label='Original Y Rotation')
#         ax_yrot.axvline(y_rot, color='r', linestyle='--', label='Minimized Y Rotation')
#         ax_yrot.set_title(f'X Residuals vs Y Rotation {ladder.name}')
#         ax_yrot.set_xlabel('Y Rotation (degrees)')
#         ax_yrot.set_ylabel('X Residual Distribution Gaussian Width (mm)')
#         fig_yrot.tight_layout()
#         fig_yrot.canvas.manager.set_window_title(f'X Residuals vs Y Rotation {ladder.name}')
#
#         fig_zrot, ax_zrot = plt.subplots()
#         ax_zrot.plot(z_orientations, x_res_z_rots, marker='o', label='X Residuals')
#         ax_zrot.plot(z_orientations, y_res_z_rots, marker='o', label='Y Residuals')
#         ax_zrot.axvline(original_orientation[2], color='g', linestyle='--', label='Original Z Rotation')
#         ax_zrot.axvline(z_rot, color='r', linestyle='--', label='Minimized Z Rotation')
#         ax_zrot.set_title(f'Residuals vs Z Rotation {ladder.name}')
#         ax_zrot.set_xlabel('Z Rotation (degrees)')
#         ax_zrot.set_ylabel('Residual Distribution Gaussian Width (mm)')
#         ax_zrot.legend()
#         fig_zrot.tight_layout()
#         fig_zrot.canvas.manager.set_window_title(f'Residuals vs Z Rotation {ladder.name}')
#
#     ladder.set_orientation(*original_orientation)
#
#     return x_rot_min, y_rot_min, z_rot_min


def banco_align_rotation(ladder, ray_data, plot=True, n_points=1000):
    """
    Align ladder by minimizing residuals between ray and cluster centroids
    :param ladder:
    :param ray_data:
    :param plot:
    :return:
    """

    original_rotations = ladder.rotations
    x_min, x_max = -0.5, 0.5
    y_min, y_max = -0.5, 0.5
    z_min, z_max = -0.5, 0.5
    x_rotations = np.linspace(x_min, x_max, n_points)
    y_rotations = np.linspace(y_min, y_max, n_points)
    z_rotations = np.linspace(z_min, z_max, n_points)
    ladder.add_rotation(0, [0, 0, 0])

    ladder.replace_last_rotation(x_min, 'x')
    ladder.convert_cluster_coords()
    x_min_rot_ray_triggers = get_close_triggers(ladder, ray_data)
    ladder.replace_last_rotation(x_max, 'x')
    ladder.convert_cluster_coords()
    x_max_rot_ray_triggers = get_close_triggers(ladder, ray_data)
    # Get the intersection of the two sets
    x_rot_ray_triggers = np.intersect1d(x_min_rot_ray_triggers, x_max_rot_ray_triggers)
    x_res_x_rots, y_res_x_rots, r_res_x_rots, sum_res_x_rots = [], [], [], []
    for x_rot_i in x_rotations:
        ladder.replace_last_rotation(x_rot_i, 'x')
        ladder.convert_cluster_coords()
        # x_mu, x_sd, y_mu, y_sd = banco_get_residuals(ladder, ray_data, False)
        x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = banco_get_residuals_no_fit_triggers(ladder, ray_data, x_rot_ray_triggers)
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

    ladder.replace_last_rotation(y_min, 'y')
    ladder.convert_cluster_coords()
    y_min_rot_ray_triggers = get_close_triggers(ladder, ray_data)
    ladder.replace_last_rotation(y_max, 'y')
    ladder.convert_cluster_coords()
    y_max_rot_ray_triggers = get_close_triggers(ladder, ray_data)
    y_rot_ray_triggers = np.intersect1d(y_min_rot_ray_triggers, y_max_rot_ray_triggers)

    x_res_y_rots, y_res_y_rots, r_res_y_rots, sum_res_y_rots = [], [], [], []
    for y_rot_i in y_rotations:
        ladder.replace_last_rotation(y_rot_i, 'y')
        ladder.convert_cluster_coords()
        # x_mu, x_sd, y_mu, y_sd = banco_get_residuals(ladder, ray_data, False)
        x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = banco_get_residuals_no_fit_triggers(ladder, ray_data, y_rot_ray_triggers)
        x_res_y_rots.append(x_sd)
        y_res_y_rots.append(y_sd)
        r_res_y_rots.append(r_sd)
        sum_res_y_rots.append(np.sqrt(x_sd ** 2 + y_sd ** 2))
    # y_rot = y_rotations[np.argmin(x_res_y_rots)]
    y_rot_min = 0

    z0_ray_triggers = get_close_triggers(ladder, ray_data)
    ladder.replace_last_rotation(z_min, 'z')
    ladder.convert_cluster_coords()
    z_min_ray_triggers = get_close_triggers(ladder, ray_data)
    ladder.replace_last_rotation(z_max, 'z')
    ladder.convert_cluster_coords()
    z_max_ray_triggers = get_close_triggers(ladder, ray_data)
    z_ray_triggers = np.intersect1d(z_min_ray_triggers, z_max_ray_triggers)
    z_ray_triggers = np.intersect1d(z_ray_triggers, z0_ray_triggers)

    x_res_z_rots, y_res_z_rots, r_res_z_rots, sum_res_z_rots = [], [], [], []
    for z_rot_i in z_rotations:
        ladder.replace_last_rotation(z_rot_i, 'z')
        ladder.convert_cluster_coords()
        # x_mu, x_sd, y_mu, y_sd = banco_get_residuals(ladder, ray_data, False)
        x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = banco_get_residuals_no_fit_triggers(ladder, ray_data, z_ray_triggers)
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
        ax_xrot.set_title(f'Y Residuals vs X Rotation {ladder.name}')
        ax_xrot.set_xlabel('X Rotation (degrees)')
        ax_xrot.set_ylabel('Y Residual Distribution Gaussian Width (mm)')
        ax_xrot.legend()
        fig_xrot.tight_layout()
        fig_xrot.canvas.manager.set_window_title(f'Y Residuals vs X Rotation {ladder.name}')

        fig_yrot, ax_yrot = plt.subplots()
        ax_yrot.plot(y_rotations, x_res_y_rots, color='blue', marker='o', label='X Residuals')
        # ax_yrot.axvline(original_orientation[1], color='g', linestyle='--', label='Original Y Rotation')
        # ax_yrot.plot(y_rotations, y_res_y_rots, color='green', marker='o', label='Y Residuals')
        # ax_yrot.plot(y_rotations, r_res_y_rots, color='orange', marker='o', label='R Residuals')
        ax_yrot.axvline(y_rot_min, color='r', linestyle='--', label='Minimized Y Rotation')
        ax_yrot.set_title(f'X Residuals vs Y Rotation {ladder.name}')
        ax_yrot.set_xlabel('Y Rotation (degrees)')
        ax_yrot.set_ylabel('X Residual Distribution Gaussian Width (mm)')
        ax_yrot.legend()
        fig_yrot.tight_layout()
        fig_yrot.canvas.manager.set_window_title(f'X Residuals vs Y Rotation {ladder.name}')

        fig_zrot, ax_zrot = plt.subplots()
        ax_zrot.plot(z_rotations, x_res_z_rots, color='blue', marker='o', label='X Residuals')
        ax_zrot.plot(z_rotations, y_res_z_rots, color='green', marker='o', label='Y Residuals')
        # ax_zrot.plot(z_rotations, r_res_z_rots, color='orange', marker='o', label='R Residuals')
        ax_zrot.plot(z_rotations, sum_res_z_rots, color='red', marker='o', label='Sum Residuals')
        # ax_zrot.axvline(original_orientation[2], color='g', linestyle='--', label='Original Z Rotation')
        ax_zrot.axvline(z_rot_min, color='r', linestyle='--', label='Minimized Z Rotation')
        ax_zrot.set_title(f'Residuals vs Z Rotation {ladder.name}')
        ax_zrot.set_xlabel('Z Rotation (degrees)')
        ax_zrot.set_ylabel('Residual Distribution Gaussian Width (mm)')
        ax_zrot.legend()
        fig_zrot.tight_layout()
        fig_zrot.canvas.manager.set_window_title(f'Residuals vs Z Rotation {ladder.name}')

    ladder.set_rotations(original_rotations)
    ladder.convert_cluster_coords()

    return x_rot_min, y_rot_min, z_rot_min


def minimize_translation_residuals(ladder_center, ladder, ray_data, good_triggers):
    ladder.set_center(x=ladder_center[0], y=ladder_center[1], z=ladder_center[2])
    ladder.convert_cluster_coords()
    x_mu, x_sd, y_mu, y_sd, r_mu, r_sd = banco_get_residuals_no_fit_triggers(ladder, ray_data, good_triggers)
    return np.sqrt(x_sd ** 2 + y_sd ** 2)


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


def banco_get_residuals(ladder, ray_data, plot=False):
    ray_trigger_ids = np.array(ladder.cluster_triggers) + 1  # Banco starts at 0, rays start at 1
    # x_rays, y_rays, event_num_rays = get_xy_positions(ray_data, ladder.center[2], ray_trigger_ids)
    x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(ladder.center[2], ray_trigger_ids)
    x_rays, y_rays, event_num_rays = remove_outlying_rays(x_rays, y_rays, event_num_rays, ladder.size, 1.2)

    cluster_centroids, banco_triggers = np.array(ladder.cluster_centroids), np.array(ladder.cluster_triggers) + 1
    cluster_centroids = cluster_centroids[np.isin(banco_triggers, event_num_rays)]

    x_res, y_res = get_ray_ladder_residuals(x_rays, y_rays, cluster_centroids)
    x_res, y_res = np.array(x_res), np.array(y_res)

    x_res_mean, y_res_mean = np.mean(x_res), np.mean(y_res)
    x_res_std, y_res_std = np.std(x_res), np.std(y_res)

    # Mask by percentiles instead
    mask = (x_res > x_res_mean - 2 * x_res_std) & (x_res < x_res_mean + 2 * x_res_std) & \
           (y_res > y_res_mean - 2 * y_res_std) & (y_res < y_res_mean + 2 * y_res_std)
    x_res_filter, y_res_filter = x_res[mask], y_res[mask]

    # p0_x = np.array([np.mean(x_res_filter), np.std(x_res_filter) / 2])
    # fit_x = minimize(neg_log_likelihood, p0_x, args=(x_res_filter,), bounds=[(-np.inf, np.inf), (1e-5, np.inf)])
    # fitted_mu_x, fitted_sigma_x = fit_x.x
    #
    # p0_y = np.array([np.mean(y_res_filter), np.std(y_res_filter) / 2])
    # fit_y = minimize(neg_log_likelihood, p0_y, args=(y_res_filter,), bounds=[(-np.inf, np.inf), (1e-5, np.inf)])
    # fitted_mu_y, fitted_sigma_y = fit_y.x

    try:
        # Determine binning in a smarter/more robust way
        counts_x, bins_x = np.histogram(x_res_filter, bins=np.linspace(min(x_res_filter), max(x_res_filter), 50))
        bin_centers_x = (bins_x[:-1] + bins_x[1:]) / 2
        bin_width_x = bins_x[1] - bins_x[0]
        p0_x = [len(x_res_filter), np.mean(x_res_filter), np.std(x_res_filter)]
        popt_x, pcov_x = cf(gaus_amp, bin_centers_x, counts_x, p0=p0_x)

        # Filter out outliers for stable binning and refit
        x_res_filter2 = x_res_filter[np.abs(x_res_filter - popt_x[1]) < 3 * popt_x[2]]
        counts_x, bins_x = np.histogram(x_res_filter2, bins=np.linspace(min(x_res_filter2), max(x_res_filter2), 50))
        bin_centers_x = (bins_x[:-1] + bins_x[1:]) / 2
        bin_width_x = bins_x[1] - bins_x[0]
        p0_x = [len(x_res_filter2), np.mean(x_res_filter2), np.std(x_res_filter2)]
        popt_x, pcov_x = cf(gaus_amp, bin_centers_x, counts_x, p0=p0_x)

        # Similar for y
        counts_y, bins_y = np.histogram(y_res_filter, bins=np.linspace(min(y_res_filter), max(y_res_filter), 50))
        bin_centers_y = (bins_y[:-1] + bins_y[1:]) / 2
        bin_width_y = bins_y[1] - bins_y[0]
        p0_y = [len(y_res_filter), np.mean(y_res_filter), np.std(y_res_filter)]
        popt_y, pcov_y = cf(gaus_amp, bin_centers_y, counts_y, p0=p0_y)

        # Filter out outliers for stable binning and refit
        y_res_filter2 = y_res_filter[np.abs(y_res_filter - popt_y[1]) < 3 * popt_y[2]]
        counts_y, bins_y = np.histogram(y_res_filter2, bins=np.linspace(min(y_res_filter2), max(y_res_filter2), 50))
        bin_centers_y = (bins_y[:-1] + bins_y[1:]) / 2
        bin_width_y = bins_y[1] - bins_y[0]
        p0_y = [len(y_res_filter2), np.mean(y_res_filter2), np.std(y_res_filter2)]
        popt_y, pcov_y = cf(gaus_amp, bin_centers_y, counts_y, p0=p0_y)
    except RuntimeError:
        print(f'Error fitting residuals for ladder {ladder.name}')
        return float('nan'), float('nan'), float('nan'), float('nan')

    if plot:
        # print(f'X_residuals unbinned gaussian fit: mu={fitted_mu_x:.2f}mm, sigma={fitted_sigma_x:.2f}mm')
        print(f'X_residuals binned gaussian fit: mu={popt_x[1]:.2f}mm, sigma={popt_x[2]:.2f}mm')
        # print(f'Y_residuals unbinned gaussian fit: mu={fitted_mu_y:.2f}mm, sigma={fitted_sigma_y:.2f}mm')
        print(f'Y_residuals binned gaussian fit: mu={popt_y[1]:.2f}mm, sigma={popt_y[2]:.2f}mm')

        # x_plot_points = np.linspace(min(x_res_filter2), max(x_res_filter2), 1000)
        # fig_x_bar, ax_x_bar = plt.subplots()
        # ax_x_bar.bar(bin_centers_x, counts_x, width=bin_width_x, align='center')
        # # ax_x_bar.plot(x_plot_points, len(x_res) * gaussian(x_plot_points, *fit_x.x), color='r')
        # ax_x_bar.plot(x_plot_points, gaus_amp(x_plot_points, *popt_x), color='g')
        # ax_x_bar.set_title(f'X Residuals {ladder.name}')
        # ax_x_bar.set_xlabel('X Residual (mm)')
        # ax_x_bar.set_ylabel('Entries')
        # fig_x_bar.tight_layout()
        # fig_x_bar.canvas.manager.set_window_title(f'X Residuals {ladder.name}')

        y_plot_points = np.linspace(min(y_res_filter2), max(y_res_filter2), 1000)
        fig_y_bar, ax_y_bar = plt.subplots()
        ax_y_bar.bar(bin_centers_y, counts_y, width=bin_width_y, align='center')
        # ax_y_bar.plot(y_plot_points, len(y_res) * gaussian(y_plot_points, *fit_y.x), color='r')
        ax_y_bar.plot(y_plot_points, gaus_amp(y_plot_points, *popt_y), color='g')
        ax_y_bar.set_title(f'Y Residuals {ladder.name}')
        ax_y_bar.set_xlabel('Y Residual (mm)')
        ax_y_bar.set_ylabel('Entries')
        fig_y_bar.tight_layout()
        fig_y_bar.canvas.manager.set_window_title(f'Y Residuals {ladder.name}')

    # return fitted_mu_x, fitted_sigma_x, fitted_mu_y, fitted_sigma_y
    return popt_x[1], popt_x[2], popt_y[1], popt_y[2]


def banco_align_ladders(ladders, triggers=None):
    """
    Align ladders between themselves by minimizing residuals between clusters and a linear fit between ladders
    :param ladders:
    :param triggers:
    :return:
    """
    if triggers is None:
        triggers = np.unique(np.concatenate([ladder.cluster_triggers for ladder in ladders]))


def banco_ladder_fit_residuals_by_chip(ladders, triggers, plot=False):
    """
    Fit ladders in each trigger to a line and calculate the residuals on each ladder.
    :param ladders:
    :param triggers:
    :param plot:
    :return:
    """
    residuals = {ladder.name: {chip_num: {'x': [], 'y': [], 'r': []} for chip_num in range(ladder.n_chips)}
                 for ladder in ladders}

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
            chip = ladder.get_largest_cluster_chip_num_by_trigger(trigger)
            x_res = (cluster[0] - linear(cluster[2], *popt_x)) * 1000  # Convert mm to microns
            y_res = (cluster[1] - linear(cluster[2], *popt_y)) * 1000
            r_res = np.sqrt(x_res ** 2 + y_res ** 2)
            residuals[ladder.name][chip]['x'].append(x_res)
            residuals[ladder.name][chip]['y'].append(y_res)
            residuals[ladder.name][chip]['r'].append(r_res)

    if plot:
        for ladder, res_data in residuals.items():
            # all_res = np.concatenate([np.concatenate(res_data[chip]['r']) for chip in range(ladder.n_chips)])
            # print(f'\nLadder {ladder}')
            # print(f'X Residuals Mean: {np.mean(all_res["x"])}')
            # print(f'X Residuals Std: {np.std(res["x"])}')
            # print(f'Y Residuals Mean: {np.mean(res["y"])}')
            # print(f'Y Residuals Std: {np.std(res["y"])}')
            fig_x, ax_x = plt.subplots()
            fig_y, ax_y = plt.subplots()
            fig_r, ax_r = plt.subplots()

            for chip_i, res_xyr in res_data.items():
                if len(res_xyr['x']) < 2:
                    continue
                ax_x.hist(res_xyr['x'], bins=np.linspace(min(res_xyr['x']), max(res_xyr['x']), 25), alpha=0.5,
                          label=f'Chip {chip_i}')
                ax_y.hist(res_xyr['y'], bins=np.linspace(min(res_xyr['y']), max(res_xyr['y']), 25), alpha=0.5,
                          label=f'Chip {chip_i}')
                ax_r.hist(res_xyr['r'], bins=np.linspace(min(res_xyr['r']), max(res_xyr['r']), 25), alpha=0.5,
                          label=f'Chip {chip_i}')
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


def banco_get_residuals_no_fit(ladder, ray_data):
    ray_trigger_ids = np.array(ladder.cluster_triggers) + 1  # Banco starts at 0, rays start at 1
    # x_rays, y_rays, event_num_rays = get_xy_positions(ray_data, ladder.center[2], ray_trigger_ids)
    x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(ladder.center[2], ray_trigger_ids)
    x_rays, y_rays, event_num_rays = remove_outlying_rays(x_rays, y_rays, event_num_rays, ladder.size, 1.2)

    cluster_centroids, banco_triggers = np.array(ladder.cluster_centroids), np.array(ladder.cluster_triggers) + 1
    cluster_centroids = cluster_centroids[np.isin(banco_triggers, event_num_rays)]

    x_res, y_res = get_ray_ladder_residuals(x_rays, y_rays, cluster_centroids)
    x_res, y_res = np.array(x_res), np.array(y_res)

    # Mask by n_stds
    x_res_mean, y_res_mean = np.mean(x_res), np.mean(y_res)
    x_res_std, y_res_std = np.std(x_res), np.std(y_res)
    mask = (x_res > x_res_mean - 3 * x_res_std) & (x_res < x_res_mean + 3 * x_res_std) & \
           (y_res > y_res_mean - 3 * y_res_std) & (y_res < y_res_mean + 3 * y_res_std)
    # Mask by 10 and 90 percentiles
    # mask = (x_res > np.percentile(x_res, 10)) & (x_res < np.percentile(x_res, 90)) & \
    #        (y_res > np.percentile(y_res, 10)) & (y_res < np.percentile(y_res, 90))
    x_res_filter, y_res_filter = x_res[mask], y_res[mask]
    r_res_filter = np.sqrt(x_res_filter ** 2 + y_res_filter ** 2)

    return (np.mean(x_res_filter), np.std(x_res_filter), np.mean(y_res_filter), np.std(y_res_filter),
            np.mean(r_res_filter), np.std(r_res_filter))


def banco_get_residuals_no_fit_triggers(ladder, ray_data, ray_triggers, plot=False):
    x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(ladder.center[2], ray_triggers)

    cluster_centroids, banco_triggers = np.array(ladder.cluster_centroids), np.array(ladder.cluster_triggers) + 1
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


def get_close_triggers(ladder, ray_data):
    ray_trigger_ids = np.array(ladder.cluster_triggers) + 1  # Banco starts at 0, rays start at 1
    x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(ladder.center[2], ray_trigger_ids)
    x_rays, y_rays, event_num_rays = remove_outlying_rays(x_rays, y_rays, event_num_rays, ladder.size, 1.2)

    cluster_centroids, banco_triggers = np.array(ladder.cluster_centroids), np.array(ladder.cluster_triggers) + 1
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


# def plot_banco_m3_tracks(ladder, ray_data, triggers):
#     """
#     Make a 3D plot of the m3 detectors and each ladder, along with the hits and tracks
#     :param ladder:
#     :param ray_data:
#     :param triggers:
#     :return:
#     """


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


def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def quadratic_shift(x, a, c, d):
    return a * (x - d) ** 2 + c


def quadratic_2d(xy, ax, ay, cross, bx, by, c):
    x, y = xy
    return ax * x ** 2 + ay * y ** 2 + cross * x * y + bx * x + by * y + c


def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaus_amp(x, amp, mu, sigma):
    return amp * gaussian(x, mu, sigma)


def neg_log_likelihood(params, data):
    mu, sigma = params
    likelihoods = gaussian(data, mu, sigma)
    log_likelihoods = np.log(likelihoods)
    return -np.sum(log_likelihoods)


if __name__ == '__main__':
    main()
