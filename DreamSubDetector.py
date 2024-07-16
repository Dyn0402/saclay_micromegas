#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 16 11:04 2024
Created in PyCharm
Created as saclay_micromegas/DreamSubDetector

@author: Dylan Neff, dn277127
"""


class DreamSubDetector:
    def __init__(self):
        self.x_pos, self.y_pos = None, None
        self.x_amps, self.y_amps = None, None
        self.x_hits, self.y_hits = None, None
        self.x_clusters, self.y_clusters = None, None

        self.x_pitch, self.y_pitch = None, None
        self.x_interpitch, self.y_interpitch = None, None
        self.x_connector, self.y_connector = None, None
        self.resist = None

    def set_x(self, pos, amps, hits, pitch, interpitch, connector):
        self.x_pos = pos
        self.x_amps = amps
        self.x_hits = hits
        self.x_pitch = pitch
        self.x_interpitch = interpitch
        self.x_connector = connector

    def set_y(self, pos, amps, hits, pitch, interpitch, connector):
        self.y_pos = pos
        self.y_amps = amps
        self.y_hits = hits
        self.y_pitch = pitch
        self.y_interpitch = interpitch
        self.y_connector = connector

    def get_clusters(self):
        print(f'x_hits: {self.x_hits.shape}')
