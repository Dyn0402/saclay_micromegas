#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 03 6:51 PM 2024
Created in PyCharm
Created as saclay_micromegas/august_cosmic_event_num_wrap.py

@author: Dylan Neff, Dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak


def main():
    plot_event_num()
    print('donzo')


def plot_event_num():
    rays_path = 'F:/Saclay/cosmic_data/sg1_stats_7-26-24/max_hv_long_1/m3_tracking_root/'
    for file in os.listdir(rays_path):
        print(file)
        with uproot.open(rays_path + file) as file:
            print(file.keys()[-1])
            tree = file[file.keys()[-1]]
            event_nums = tree['evn'].array()
            print(event_nums)


if __name__ == '__main__':
    main()
