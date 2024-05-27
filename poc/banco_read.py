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

import uproot
import awkward as ak
import vector


def main():
    # read_decoded_banco()
    # read_raw_banco()
    banco_analysis()
    # find_cluster_test()
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
            print(f'Event {event["eventId"]} has {len(event["clusters.size"])} clusters of size {event["clusters.size"]}')
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
    ladders = [157, 160, 162, 163]
    noise_noise_threshold, data_noise_threshold = 2, 3
    ladders_trigger_ids, ladders_cluster_centroids = {}, {}
    for ladder in ladders:
        file_path = f'C:/Users/Dylan/Desktop/banco_test2/multinoiseScan_240502_151923-B0-ladder{ladder}.root'
        noise_path = f'C:/Users/Dylan/Desktop/banco_test2/Noise_{ladder}.root'
        noise_data = read_banco_file(noise_path)
        # plot_noise_vs_threshold(noise_data, noise_noise_threshold, ladder)
        noise_pixels = get_noise_pixels(noise_data, noise_noise_threshold)
        data = read_banco_file(file_path)
        # plot_noise_vs_threshold(data, data_noise_threshold)
        data_noise_pixels = get_noise_pixels(data, data_noise_threshold)
        noise_pixels = np.unique(np.concatenate([noise_pixels, data_noise_pixels]), axis=0)
        trigger_ids, cluster_centroids = cluster_data(data, noise_pixels)
        # Get only trigger_id/centroid pairs with a single cluster
        trigger_ids = [trigger_ids[i] for i in range(len(cluster_centroids)) if len(cluster_centroids[i]) == 1]
        cluster_centroids = [cluster_centroids[i] for i in range(len(cluster_centroids)) if len(cluster_centroids[i]) == 1]
        trigger_ids, cluster_centroids = np.array(trigger_ids), np.array(cluster_centroids)
        ladders_trigger_ids[ladder], ladders_cluster_centroids[ladder] = trigger_ids, cluster_centroids
        # ladders_trigger_ids[ladder], ladders_cluster_centroids = trigger_ids, cluster_centroids
        # for trigger_id, clusters in zip(trigger_ids, cluster_centroids):
        #     print(f'Trigger {trigger_id} has {len(clusters)} clusters')
        #     for cluster in clusters:
        #         print(f'Cluster at {cluster}')
        plot_cluster_number_histogram(cluster_centroids, ladder)
        plot_cluster_scatter(cluster_centroids, ladder)
    fig, ax = plt.subplots()
    # plot correlation in x position of centroids between ladder pairs
    for i in range(len(ladders) - 1):
        ladder1, ladder2 = ladders[i], ladders[i + 1]
        centroids1, centroids2 = ladders_cluster_centroids[ladder1], ladders_cluster_centroids[ladder2]
        centroids1 = np.reshape(centroids1, (centroids1.shape[0], centroids1.shape[-1]))
        centroids2 = np.reshape(centroids2, (centroids2.shape[0], centroids2.shape[-1]))
        trigger_ids1, trigger_ids2 = ladders_trigger_ids[ladder1], ladders_trigger_ids[ladder2]
        common_triggers = np.intersect1d(trigger_ids1, trigger_ids2)
        common_i1 = np.where(np.isin(trigger_ids1, common_triggers))
        common_i2 = np.where(np.isin(trigger_ids2, common_triggers))
        centroids1, centroids2 = centroids1[common_i1], centroids2[common_i2]
        ax.scatter(centroids1[:, 1], centroids2[:, 1], alpha=0.5, label=f'Ladders {ladder1} - {ladder2}')
    ax.set_title('Cluster Centroid Correlation')
    ax.set_xlabel('Column Ladder 1')
    ax.set_ylabel('Column Ladder 2')
    ax.legend()
    plt.show()


def plot_cluster_scatter(cluster_centroids, title=None):
    fig, ax = plt.subplots()
    clusters = []
    for cluster in cluster_centroids:
        clusters.extend(cluster)
    clusters = np.array(clusters)
    ax.scatter(clusters[:, 1], clusters[:, 0], alpha=0.5)
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
    find_clusters(data, noise_pixels)


def cluster_data(data, noise_pixels=None):
    data = np.split(data, np.unique(data[:, 0], return_index=True)[1][1:])

    trigger_ids, cluster_centroids = [], []
    for trigger in data:
        trigger_ids.append(trigger[0][0])
        cluster_centroids.append(find_clusters(trigger[:, 1:], noise_pixels))

    return trigger_ids, cluster_centroids


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


def find_clusters(data, noise_pixels=None):
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
        clusters.append(cluster)

    # Get x and y centroids of clusters
    cluster_centroids = []
    for cluster in clusters:
        # Remove noise pixels from cluster
        cluster = [np.array(data[pixel]) for pixel in cluster]
        if noise_pixels is not None:
            cluster = [pixel for pixel in cluster if not np.any(np.all(pixel == noise_pixels, axis=1))]
        if len(cluster) == 0:
            continue
        cluster_centroids.append(np.mean(cluster, axis=0))

    return cluster_centroids


def is_neighbor(pixel1, pixel2, threshold=1.9):
    return np.sqrt(np.sum((pixel1 - pixel2)**2)) <= threshold


if __name__ == '__main__':
    main()
