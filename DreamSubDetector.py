#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 16 11:04 2024
Created in PyCharm
Created as saclay_micromegas/DreamSubDetector

@author: Dylan Neff, dn277127
"""

import numpy as np


class DreamSubDetector:
    def __init__(self):
        self.x_pos, self.y_pos = None, None
        self.x_chans, self.y_chans = None, None
        self.x_amps, self.y_amps = None, None
        self.x_hits, self.y_hits = None, None
        self.x_cluster_triggers, self.y_cluster_triggers = None, None
        self.x_clusters, self.y_clusters = None, None
        self.x_cluster_centroids, self.y_cluster_centroids = None, None
        self.x_largest_clusters, self.y_largest_clusters = None, None
        self.x_largest_cluster_centroids, self.y_largest_cluster_centroids = None, None

        self.x_pitch, self.y_pitch = None, None
        self.x_interpitch, self.y_interpitch = None, None
        self.x_connector, self.y_connector = None, None
        self.resist = None

    def set_x(self, pos, amps, hits, pitch, interpitch, connector, cluster_triggers=None, clusters=None,
              cluster_centroids=None, largest_clusters=None, largest_cluster_centroids=None, chans=None):
        self.x_pos = pos
        self.x_amps = amps
        self.x_hits = hits
        self.x_pitch = pitch
        self.x_interpitch = interpitch
        self.x_connector = connector
        self.x_cluster_triggers = cluster_triggers
        self.x_clusters = clusters
        self.x_cluster_centroids = cluster_centroids
        self.x_largest_clusters = largest_clusters
        self.x_largest_cluster_centroids = largest_cluster_centroids
        self.x_chans = chans

    def set_y(self, pos, amps, hits, pitch, interpitch, connector, cluster_triggers=None, clusters=None,
              cluster_centroids=None, largest_clusters=None, largest_cluster_centroids=None, chans=None):
        self.y_pos = pos
        self.y_amps = amps
        self.y_hits = hits
        self.y_pitch = pitch
        self.y_interpitch = interpitch
        self.y_connector = connector
        self.y_cluster_triggers = cluster_triggers
        self.y_clusters = clusters
        self.y_cluster_centroids = cluster_centroids
        self.y_largest_clusters = largest_clusters
        self.y_largest_cluster_centroids = largest_cluster_centroids
        self.y_chans = chans

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