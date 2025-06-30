#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 16 11:04 2024
Created in PyCharm
Created as saclay_micromegas/DreamSubDetector

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt


class DreamSubDetector:
    def __init__(self, sub_index=None):
        self.sub_index = sub_index
        self.x_pos, self.y_pos = None, None
        self.x_chans, self.y_chans = None, None
        self.x_amps, self.y_amps = None, None
        self.x_hits, self.y_hits = None, None
        self.x_times, self.y_times = None, None
        self.x_cluster_triggers, self.y_cluster_triggers = None, None
        self.x_clusters, self.y_clusters = None, None
        self.x_cluster_sizes, self.y_cluster_sizes = None, None
        self.x_cluster_centroids, self.y_cluster_centroids = None, None
        self.x_largest_clusters, self.y_largest_clusters = None, None
        self.x_largest_cluster_sizes, self.y_largest_cluster_sizes = None, None
        self.x_largest_cluster_centroids, self.y_largest_cluster_centroids = None, None

        self.x_pitch, self.y_pitch = None, None
        self.x_interpitch, self.y_interpitch = None, None
        self.x_connector, self.y_connector = None, None
        self.resist = None

        self.description = None

    def set_x(self, pos, amps, hits, times, pitch, interpitch, connector,
              cluster_triggers=None, clusters=None, cluster_sizes=None, cluster_centroids=None,
              largest_clusters=None, largest_cluster_sizes=None, largest_cluster_centroids=None,
              chans=None):
        self.x_pos = pos
        self.x_amps = amps
        self.x_hits = hits
        self.x_times = times
        self.x_pitch = float(pitch)
        self.x_interpitch = float(interpitch)
        self.x_connector = connector
        self.x_cluster_triggers = cluster_triggers
        self.x_clusters = clusters
        self.x_cluster_sizes = cluster_sizes
        self.x_cluster_centroids = cluster_centroids
        self.x_largest_clusters = largest_clusters
        self.x_largest_cluster_sizes = largest_cluster_sizes
        self.x_largest_cluster_centroids = largest_cluster_centroids
        self.x_chans = chans

        self.set_description()

    def set_y(self, pos, amps, hits, times, pitch, interpitch, connector,
              cluster_triggers=None, clusters=None, cluster_sizes=None, cluster_centroids=None,
              largest_clusters=None, largest_cluster_sizes=None, largest_cluster_centroids=None,
              chans=None):
        self.y_pos = pos
        self.y_amps = amps
        self.y_hits = hits
        self.y_times = times
        self.y_pitch = float(pitch)
        self.y_interpitch = float(interpitch)
        self.y_connector = connector
        self.y_cluster_triggers = cluster_triggers
        self.y_clusters = clusters
        self.y_cluster_sizes = cluster_sizes
        self.y_cluster_centroids = cluster_centroids
        self.y_largest_clusters = largest_clusters
        self.y_largest_cluster_sizes = largest_cluster_sizes
        self.y_largest_cluster_centroids = largest_cluster_centroids
        self.y_chans = chans

        self.set_description()

    def set_description(self):
        """
        Write string describing sub-detector. Specifically, pitch and interpitch for x and y. Keep concise.
        :return:
        """
        self.description = f'pitch: ({self.x_pitch}, {self.y_pitch}) mm, ' \
                            f'interpitch: ({self.x_interpitch}, {self.y_interpitch}) mm'

    def get_event_centroids(self):
        """
        Get x and y centroids for events in which there are clusters in both x and y.
        Have AI optimize this, currently slow.
        :return:
        """
        # Get list of x triggers in which the trigger appears exactly once in self.x_cluster_triggers
        non_degenerate_x_triggers = {trigger: index for index, trigger in enumerate(self.x_cluster_triggers)
                                     if np.count_nonzero(self.x_cluster_triggers == trigger) == 1}
        non_degenerate_y_triggers = {trigger: index for index, trigger in enumerate(self.y_cluster_triggers)
                                     if np.count_nonzero(self.y_cluster_triggers == trigger) == 1}

        triggers, centroids = [], []

        for x_trigger, x_event_index in non_degenerate_x_triggers.items():
            if x_trigger not in non_degenerate_y_triggers:
                continue
            y_event_index = non_degenerate_y_triggers[x_trigger]
            x_centroid = self.x_largest_cluster_centroids[x_event_index]
            y_centroid = self.y_largest_cluster_centroids[y_event_index]

            triggers.append(x_trigger)
            centroids.append(np.array([x_centroid, y_centroid]))
        return np.array(triggers), np.array(centroids)

    def get_cluster_sizes(self, largest=True, event_nums=None, return_ray_mask=False):
        """
        Get cluster sizes for x and y. If largest is True, return only the largest cluster sizes.
        If event_nums is provided, filter clusters by these event numbers.
        :param largest: bool, whether to return only the largest clusters
        :param event_nums: list or array of event numbers to filter by
        :param return_ray_mask: bool, whether to return mask for the events
        :return: tuple of x and y cluster sizes
        """
        if largest:
            x_sizes = np.array(self.x_largest_cluster_sizes)
            y_sizes = np.array(self.y_largest_cluster_sizes)
        else:
            x_sizes = np.array([cluster for event_clusters in self.x_cluster_sizes for cluster in event_clusters])
            y_sizes = np.array([cluster for event_clusters in self.y_cluster_sizes for cluster in event_clusters])

        if event_nums is not None:
            event_mask_x = np.isin(self.x_cluster_triggers, event_nums) & np.isin(self.x_cluster_triggers, self.y_cluster_triggers)
            event_mask_y = np.isin(self.y_cluster_triggers, event_nums) & np.isin(self.y_cluster_triggers, self.x_cluster_triggers)
            x_sizes = x_sizes[event_mask_x]
            y_sizes = y_sizes[event_mask_y]

        if return_ray_mask:
            ray_mask = np.isin(event_nums, self.x_cluster_triggers) & np.isin(event_nums, self.y_cluster_triggers)
            return x_sizes, y_sizes, ray_mask

        return x_sizes, y_sizes

    def plot_cluster_sizes(self, largest=True, event_nums=None):
        """
        Plot a histogram of cluster sizes for x and y.
        :return:
        """
        x_sizes, y_sizes = self.get_cluster_sizes(largest=largest, event_nums=event_nums)

        fig, ax = plt.subplots()
        ax.hist(x_sizes, bins=np.arange(0.5, max(x_sizes) + 1.5, 1), histtype='step', label='X Clusters')
        ax.hist(y_sizes, bins=np.arange(0.5, max(y_sizes) + 1.5, 1), histtype='step', label='Y Clusters')
        x_mean, x_err = np.mean(x_sizes), np.std(x_sizes) / np.sqrt(len(x_sizes))
        y_mean, y_err = np.mean(y_sizes), np.std(y_sizes) / np.sqrt(len(y_sizes))
        ax.annotate(f'X Average Size: {x_mean:.2f}\nY Average Size: {y_mean:.2f}',
                    xy=(0.5, 0.95), xycoords='axes fraction', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
                    ha='center', va='top')
        ax.set_xlabel('Cluster Size')
        ax.set_ylabel('Count')
        ax.set_title(f'Cluster Size Distribution {self.description}')
        ax.legend()
        fig.tight_layout()

        return x_mean, y_mean, x_err, y_err
