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
        triggers, centroids = [], []
        for x_event_index, x_trigger in enumerate(self.x_cluster_triggers):
            # Check if x_trigger in y_cluster_triggers and if so get index
            y_event_index = np.where(np.array(self.y_cluster_triggers) == x_trigger)[0]
            if len(y_event_index) == 0:
                continue
            elif len(y_event_index) > 1:
                print(f'Error: Multiple y triggers for x trigger {x_trigger}')
                continue
            y_event_index = y_event_index[0]
            x_centroid = self.x_largest_cluster_centroids[x_event_index]
            y_centroid = self.y_largest_cluster_centroids[y_event_index]

            triggers.append(x_trigger)
            centroids.append(np.array([x_centroid, y_centroid]))
        return np.array(triggers), np.array(centroids)

