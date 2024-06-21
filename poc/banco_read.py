#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 03 12:03 PM 2024
Created in PyCharm
Created as saclay_micromegas/banco_read.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit as cf
from scipy.optimize import minimize

import uproot
import awkward as ak
import vector

from cosmic_det_check import get_ray_data, get_det_data, get_xy_positions
from BancoLadder import BancoLadder


def main():
    # read_decoded_banco()
    # read_raw_banco()
    banco_analysis()
    # get_banco_largest_cluster_npix_dists()
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


def banco_analysis_old():
    vector.register_awkward()
    # base_dir = 'C:/Users/Dylan/Desktop/banco_test3/'
    base_dir = '/local/home/dn277127/Bureau/banco_test3/'
    det_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    run_json_path = f'{base_dir}run_config.json'
    run_data = get_det_data(run_json_path)
    print(run_data)
    banco_names = [det_name for det_name in run_data['included_detectors'] if 'banco' in det_name]
    det_info_path = f'{det_info_dir}{det["det_type"]}.json'
    det_data = get_det_data(f'{det_info_dir}{banco_names[0]}.json')
    banco_data = [det for det in run_data['detectors'] if 'banco' in det['name']][0]
    print(banco_data)
    banco_bot_z = banco_data['det_center_coords']['z']
    bot_to_cent = run_data['bench_geometry']['banco_arm_bottom_to_center']
    arm_sep = run_data['bench_geometry']['banco_arm_separation_z']
    ladder_z = {160: banco_bot_z + bot_to_cent / 2, 163: banco_bot_z + 3 * bot_to_cent / 2,
                157: banco_bot_z + bot_to_cent / 2 + arm_sep, 162: banco_bot_z + 3 * bot_to_cent / 2 + arm_sep}
    print(ladder_z)
    ladder_x_center = banco_data['det_center_coords']['x']
    ladder_y_center = banco_data['det_center_coords']['y'] - 50
    ladder_x_len = 15
    ladder_y_len = 150
    ladder_z_width = 2  # Not useful
    ladder_x_pix = 512
    ladder_y_pix = 1024 * 5

    ray_data = get_ray_data(base_dir, [0, 1])
    # print(ray_data)

    # banco_file = f'{base_dir}multinoiseScan_240514_231935-B0-ladder157.root'
    # banco_data = read_banco_file(banco_file)
    # print(banco_data)

    ladder_nums = [157, 160, 162, 163]
    # ladder_nums = [157]
    noise_noise_threshold, data_noise_threshold = 2, 3
    ladders_trigger_ids, ladders_cluster_centroids, ladder_align_z = {}, {}, {}
    ladders = {}
    for ladder_num in ladder_nums:
        print(f'\nLadder {ladder_num}')
        ladder = BancoLadder(ladder_num)

        ladder.set_center(ladder_x_center, ladder_y_center, ladder_z[ladder_num])
        ladder.set_size(ladder_x_len, ladder_y_len, ladder_z_width)

        file_path = f'{base_dir}multinoiseScan_240514_231935-B0-ladder{ladder_num}.root'
        noise_path = f'{base_dir}Noise_{ladder_num}.root'
        ladder.read_banco_noise(noise_path)
        ladder.read_banco_data(file_path)
        ladder.get_data_noise_pixels()
        ladder.combine_data_noise()
        ladder.cluster_data(min_pixels=2)
        ladder.get_largest_clusters()
        ladder.convert_cluster_coords()

        align_banco_ladder_to_ref(ladder, ray_data)
        ladder.set_orientation(y=180)
        ladder.convert_cluster_coords()
        align_banco_ladder_to_ref(ladder, ray_data)
        plt.show()

        z_min_std = std_align(ladder_z[ladder_num], ray_data, cluster_centroids, trigger_ids, ladder, plot=False)
        # z_min_x, z_min_y, z_min_xy = z_min_std[0], z_min_std[1], z_min_std[2]
        z_min_x, z_min_y = z_min_std[0], z_min_std[1]

        print(f'Z_min stds: {z_min_std}')

        ladder_align_z[ladder_num] = z_min_x
        x_rays, y_rays, event_num_rays = get_xy_positions(ray_data, z_min_x)

        compare_rays_to_ladder(x_rays, y_rays, event_num_rays, cluster_centroids, np.array(trigger_ids),
                               plot=True, ladder=ladder_num, cluster_num_pix=cluster_num_pixels)

        # plot_cluster_number_histogram(cluster_centroids, ladder)
        # plot_num_pixels_histogram(cluster_num_pixels, ladder)

        # Get only trigger_id/centroid pairs with a single cluster
        # trigger_ids = [trigger_ids[i] for i in range(len(cluster_centroids)) if len(cluster_centroids[i]) == 1]
        # cluster_centroids = [cluster_centroids[i] for i in range(len(cluster_centroids)) if len(cluster_centroids[i]) == 1]
        # Get the cluster with the largest number of pixels for each trigger if there are multiple clusters
        trigger_ids, cluster_centroids, cluster_num_pixels = get_largest_cluster(trigger_ids, cluster_centroids,
                                                                                 cluster_num_pixels)
        trigger_ids, cluster_centroids = np.array(trigger_ids), np.array(cluster_centroids)
        ladders_trigger_ids[ladder_num], ladders_cluster_centroids[ladder_num] = trigger_ids, cluster_centroids
        # ladders_trigger_ids[ladder], ladders_cluster_centroids = trigger_ids, cluster_centroids
        # for trigger_id, clusters in zip(trigger_ids, cluster_centroids):
        #     print(f'Trigger {trigger_id} has {len(clusters)} clusters')
        #     for cluster in clusters:
        #         print(f'Cluster at {cluster}')

        # plot_cluster_scatter(cluster_centroids, ladder)

    # Combine ladder_cluster_centroids into single dict with trigger_id as key and {ladder: centroid} as value
    all_trigger_ids = np.unique(np.concatenate([ladders_trigger_ids[ladder] for ladder in ladder_nums]))
    all_cluster_centroids = {}
    for trig_id in all_trigger_ids:
        event_ladder_clusters = {}
        for ladder in ladder_nums:
            if trig_id in ladders_trigger_ids[ladder]:
                event_ladder_clusters[ladder] = ladders_cluster_centroids[ladder][
                    np.where(ladders_trigger_ids[ladder] == trig_id)[0][0]]
        all_cluster_centroids[trig_id] = event_ladder_clusters

    for trig_id, event_clusters in all_cluster_centroids.items():
        print(f'\nTrigger {trig_id}')
        z, x, y = [], [], []
        for ladder, cluster in event_clusters.items():
            print(f'Ladder {ladder}: {cluster}')
            z.append(ladder_align_z[ladder])
            x.append(cluster[0])
            y.append(cluster[1])
        if len(event_clusters) == 4:
            for ladder in ladder_nums:
                plot_event_banco_hits(ladder_data[ladder], trig_id, ladder)
            popt_x, pcov_x = cf(linear, x, z)
            popt_y, pcov_y = cf(linear, y, z)

            fig_x, ax_x = plt.subplots()
            x_range = np.linspace(min(x), max(x), 100)
            ax_x.scatter(x, z, color='b')
            ax_x.plot(x_range, linear(x_range, *popt_x), color='r')
            ax_x.set_title(f'X Position vs Ladder for Trigger {trig_id}')
            ax_x.set_xlabel('X Position (mm)')
            ax_x.set_ylabel('Ladder Z Position (mm)')

            fig_y, ax_y = plt.subplots()
            ax_y.scatter(y, z, color='g')
            y_range = np.linspace(min(y), max(y), 100)
            ax_y.plot(y_range, linear(y_range, *popt_y), color='r')
            ax_y.set_title(f'Y Position vs Ladder for Trigger {trig_id}')
            ax_y.set_xlabel('Y Position (mm)')
            ax_y.set_ylabel('Ladder Z Position (mm)')

            plt.show()

    plt.show()


def banco_analysis():
    vector.register_awkward()
    # base_dir = 'C:/Users/Dylan/Desktop/banco_test3/'
    base_dir = '/local/home/dn277127/Bureau/banco_test3/'
    det_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    run_json_path = f'{base_dir}run_config.json'
    run_data = get_det_data(run_json_path)
    print(run_data)
    banco_names = [det_name for det_name in run_data['included_detectors'] if 'banco' in det_name]

    ray_data = get_ray_data(base_dir, [0, 1])

    ladders = []
    for banco_name in banco_names:
        det_info = [det for det in run_data['detectors'] if det['name'] == banco_name][0]
        det_type_info = get_det_data(f'{det_info_dir}{det_info["det_type"]}.json')
        det_info.update(det_type_info)
        print(banco_name)
        print(det_info)
        ladder = BancoLadder(config=det_info)
        ladder_num = int(ladder.name[-3:])
        # if ladder_num != 157:
        #     continue

        file_path = f'{base_dir}multinoiseScan_240514_231935-B0-ladder{ladder_num}.root'
        noise_path = f'{base_dir}Noise_{ladder_num}.root'
        ladder.read_banco_noise(noise_path)
        ladder.read_banco_data(file_path)
        ladder.get_data_noise_pixels()
        ladder.combine_data_noise()
        ladder.cluster_data(min_pixels=2)
        ladder.get_largest_clusters()
        ladder.convert_cluster_coords()

        z_aligned = banco_ref_std_z_alignment(ladder, ray_data)
        ladder.set_center(z=z_aligned)
        ladder.convert_cluster_coords()
        x_res_mean, x_res_sigma, y_res_mean, y_res_sigma = banco_get_residuals(ladder, ray_data, False)
        aligned_x, aligned_y = ladder.center[0] + x_res_mean, ladder.center[1] + y_res_mean
        ladder.set_center(x=aligned_x, y=aligned_y)
        ladder.convert_cluster_coords()
        x_res_mean, x_res_sigma, y_res_mean, y_res_sigma = banco_get_residuals(ladder, ray_data, True)
        ladders.append(ladder)

    # Combine ladder_cluster_centroids into single dict with trigger_id as key and {ladder: centroid} as value
    all_trigger_ids = np.unique(np.concatenate([ladder.cluster_triggers for ladder in ladders]))
    all_cluster_centroids = {}
    for trig_id in all_trigger_ids:
        event_ladder_clusters = {}
        for ladder in ladders:
            if trig_id in ladder.cluster_triggers:
                event_ladder_clusters[ladder] = ladder.cluster_centroids[np.where(ladder.cluster_triggers == trig_id)[0][0]]
        all_cluster_centroids[trig_id] = event_ladder_clusters

    for trig_id, event_clusters in all_cluster_centroids.items():
        print(f'\nTrigger {trig_id}')
        z, x, y = [], [], []
        for ladder, cluster in event_clusters.items():
            print(f'Ladder {ladder.name}: {cluster}')
            z.append(ladder.center[2])
            x.append(cluster[0])
            y.append(cluster[1])
        if len(event_clusters) == 4:
            for ladder in ladders:
                plot_event_banco_hits(ladder.data, trig_id, ladder)
            popt_x, pcov_x = cf(linear, x, z)
            popt_y, pcov_y = cf(linear, y, z)

            fig_x, ax_x = plt.subplots()
            x_range = np.linspace(min(x), max(x), 100)
            ax_x.scatter(x, z, color='b')
            ax_x.plot(x_range, linear(x_range, *popt_x), color='r')
            ax_x.set_title(f'X Position vs Ladder for Trigger {trig_id}')
            ax_x.set_xlabel('X Position (mm)')
            ax_x.set_ylabel('Ladder Z Position (mm)')

            fig_y, ax_y = plt.subplots()
            ax_y.scatter(y, z, color='g')
            y_range = np.linspace(min(y), max(y), 100)
            ax_y.plot(y_range, linear(y_range, *popt_y), color='r')
            ax_y.set_title(f'Y Position vs Ladder for Trigger {trig_id}')
            ax_y.set_xlabel('Y Position (mm)')
            ax_y.set_ylabel('Ladder Z Position (mm)')

    plt.show()


def get_banco_largest_cluster_npix_dists():
    vector.register_awkward()
    # base_dir = 'C:/Users/Dylan/Desktop/banco_test3/'
    base_dir = '/local/home/dn277127/Bureau/banco_test3/'
    det_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
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
        ladder.cluster_data(min_pixels=2)
        ladder.get_largest_clusters()
        ladder.convert_cluster_coords()

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


def get_ray_ladder_residuals(x_rays, y_rays, cluster_centroids):
    x_residuals, y_residuals = [], []
    for x, y, centroid in zip(x_rays, y_rays, cluster_centroids):
        x_centroid, y_centroid, z_centroid = centroid
        x_residuals.append(x - x_centroid)
        y_residuals.append(y - y_centroid)
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
        ax.grid()
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
    ax.plot(rows, row_counts)
    ax.set_title('Row Noise Pixel Counts' + title_suffix)
    ax.set_xlabel('Row Number')
    ax.set_ylabel('Total Hits in Row')
    ax.set_yscale('log')
    fig.tight_layout()

    # cols, col_counts = np.unique(data[:, 2], return_counts=True)
    cols = np.arange(0, np.max(data[:, 2]) + 1, 1)
    col_counts = np.array([np.count_nonzero(data[:, 2] == col) for col in cols])
    fig, ax = plt.subplots()
    ax.plot(cols, col_counts)
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
    ax.grid()
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.plot(thresholds / num_triggers, num_noise)
    if default_threshold is not None and default_threshold < 1:
        ax.axvline(default_threshold, color='red', linestyle='--')
    ax.set_title('Noise Pixels vs Threshold' + title_suffix)
    ax.set_xlabel('Threshold Fraction')
    ax.set_ylabel('Noise Pixels')
    ax.set_xticklabels([f'{x * 100:.1f}%' for x in ax.get_xticks()])
    ax.grid()


def plot_event_banco_hits(data, trigger_id, ladder=None):
    event = data[data[:, 0] == trigger_id]
    fig, ax = plt.subplots()
    ax.scatter(event[:, 2], event[:, 1], alpha=0.5)
    title = f'Event {trigger_id} Banco Hits' if ladder is None else f'Event {trigger_id} Banco Hits Ladder {ladder}'
    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_xlim(0, 1024 * 5)
    ax.set_ylabel('Row')
    ax.set_ylim(0, 512)
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
            ax.grid()
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
    x_rays_filter, y_rays_filter, event_num_rays = remove_outlying_rays(x_rays, y_rays, event_num_rays, ladder.size)

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(x_rays, y_rays, alpha=0.5)
        ax.scatter(x_rays_filter, y_rays_filter, alpha=0.5)
        ax.scatter([np.mean(x_rays_filter)], [np.mean(y_rays_filter)], marker='x', color='r')
        title = f'Ref Track Detector Hits Ladder {ladder.name}, z={z0}'
        ax.grid()
        ax.set_title(title)
        fig.tight_layout()

        fig, ax = plt.subplots()
        x_banco, y_banco = ladder.cluster_centroids[:, :2].T
        ax.scatter(x_banco, y_banco, alpha=0.5)

    zs = np.linspace(z0 - z_align_range / 2, z0 + z_align_range / 2, z_align_points)
    x_stds, y_stds, x_residuals, y_residuals = [], [], [], []
    for zi in zs:
        x_rays, y_rays, event_num_rays = get_xy_positions(ray_data, zi, event_num_rays)
        x_std, y_std = np.std(x_rays), np.std(y_rays)
        x_stds.append(x_std)
        y_stds.append(y_std)

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
            ax.plot(zs, stds, marker='o')
            plot_z_range = np.linspace(min(zs), max(zs), 100)
            ax.plot(plot_z_range, quadratic_shift(plot_z_range, *popt), color='r', alpha=0.4)
            ax.axvline(z0, color='r', linestyle='--', label='Measured z')
            ax.axvline(z1, color='g', linestyle='--', label='Minimized z')
            ax.grid()
            ax.legend()
            ax.set_title(f'Ladder {ladder.name} {std_name} vs Z')
            ax.set_xlabel('Z (mm)')
            ax.set_ylabel(f'{std_name} (mm)')
            fig.tight_layout()

    return z_fit[0]


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
    x_rays, y_rays, event_num_rays = get_xy_positions(ray_data, ladder.center[2], ray_trigger_ids)
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

    p0_x = np.array([np.mean(x_res_filter), np.std(x_res_filter) / 2])
    fit_x = minimize(neg_log_likelihood, p0_x, args=(x_res_filter,), bounds=[(-np.inf, np.inf), (1e-5, np.inf)])
    fitted_mu_x, fitted_sigma_x = fit_x.x

    p0_y = np.array([np.mean(y_res_filter), np.std(y_res_filter) / 2])
    fit_y = minimize(neg_log_likelihood, p0_y, args=(y_res_filter,), bounds=[(-np.inf, np.inf), (1e-5, np.inf)])
    fitted_mu_y, fitted_sigma_y = fit_y.x

    # Determine binning in a smarter/more robust way
    counts_x, bins_x = np.histogram(x_res_filter, bins=np.linspace(min(x_res_filter), max(x_res_filter), 50))
    bin_centers_x = (bins_x[:-1] + bins_x[1:]) / 2
    bin_width_x = bins_x[1] - bins_x[0]
    p0_x = [len(x_res_filter), np.mean(x_res_filter), np.std(x_res_filter)]
    popt_x, pcov_x = cf(gaus_amp, bin_centers_x, counts_x, p0=p0_x)

    counts_y, bins_y = np.histogram(y_res_filter, bins=np.linspace(min(y_res_filter), max(y_res_filter), 50))
    bin_centers_y = (bins_y[:-1] + bins_y[1:]) / 2
    bin_width_y = bins_y[1] - bins_y[0]
    p0_y = [len(y_res_filter), np.mean(y_res_filter), np.std(y_res_filter)]
    popt_y, pcov_y = cf(gaus_amp, bin_centers_y, counts_y, p0=p0_y)

    if plot:
        print(f'X_residuals unbinned gaussian fit: mu={fitted_mu_x:.2f}mm, sigma={fitted_sigma_x:.2f}mm')
        print(f'X_residuals binned gaussian fit: mu={popt_x[1]:.2f}mm, sigma={popt_x[2]:.2f}mm')
        print(f'Y_residuals unbinned gaussian fit: mu={fitted_mu_y:.2f}mm, sigma={fitted_sigma_y:.2f}mm')
        print(f'Y_residuals binned gaussian fit: mu={popt_y[1]:.2f}mm, sigma={popt_y[2]:.2f}mm')

        x_plot_points = np.linspace(min(x_res), max(x_res), 100)

        fix_x_bar, ax_x_bar = plt.subplots()
        ax_x_bar.bar(bin_centers_x, counts_x, width=bin_width_x, align='center')
        ax_x_bar.plot(x_plot_points, len(x_res) * gaussian(x_plot_points, *fit_x.x), color='r')
        ax_x_bar.plot(x_plot_points, gaus_amp(x_plot_points, *popt_x), color='g')
        ax_x_bar.set_title('X Residuals')
        ax_x_bar.set_xlabel('X Residual (mm)')
        ax_x_bar.set_ylabel('Entries')

        y_plot_points = np.linspace(min(y_res), max(y_res), 100)

        fix_y_bar, ax_y_bar = plt.subplots()
        ax_y_bar.bar(bin_centers_y, counts_y, width=bin_width_y, align='center')
        ax_y_bar.plot(y_plot_points, len(y_res) * gaussian(y_plot_points, *fit_y.x), color='r')
        ax_y_bar.plot(y_plot_points, gaus_amp(y_plot_points, *popt_y), color='g')
        ax_y_bar.set_title('Y Residuals')
        ax_y_bar.set_xlabel('Y Residual (mm)')
        ax_y_bar.set_ylabel('Entries')

    return fitted_mu_x, fitted_sigma_x, fitted_mu_y, fitted_sigma_y


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
