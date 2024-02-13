#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 13 5:30 PM 2024
Created in PyCharm
Created as saclay_micromegas/cosmic_test.py

@author: Dylan Neff, Dylan
"""

import os
import subprocess
import shutil
import re
from datetime import datetime, timedelta
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from scipy.optimize import curve_fit as cf

import uproot
import awkward as ak

from Measure import Measure
# from fe_analysis import *


def main():
    signal_file_path = 'C:/Users/Dylan/Desktop/test/test_signal.root'
    ray_file_path = 'C:/Users/Dylan/Desktop/test/rays_CosTb_380V_stats_datrun_240212_11H42_000.root'
    # Open the ROOT file with uproot
    signal_root_file = uproot.open(signal_file_path)
    ray_root_file = uproot.open(ray_file_path)

    # Access the tree in the file
    signal_tree = signal_root_file['T;32']
    ray_tree = ray_root_file['T;1']

    # Get the variable data from the tree
    signal_evttime = ak.flatten(signal_tree['evttime'].array(), axis=None)
    ray_evttime = ak.flatten(ray_tree['evttime'].array(), axis=None)
    print(signal_evttime)
    print(ray_evttime)

    print(f'Min ray time: {min(ray_evttime)}')
    print(f'Min signal time: {min(signal_evttime)}')

    print(f'Signal time shift: {(signal_evttime - min(signal_evttime)) * 1000}')

    n_plot = int(len(ray_evttime) / 8)
    fig, ax = plt.subplots()
    ax.scatter(signal_evttime[:n_plot], ray_evttime[:n_plot], s=1)

    fig, ax = plt.subplots()
    ax.plot(signal_evttime[:n_plot], marker='o', linestyle='None', label='Signal')

    fig, ax = plt.subplots()
    ax.plot(np.diff(signal_evttime[:n_plot]), marker='o', linestyle='None', label='Signal')

    fig, ax = plt.subplots()
    ax.hist(np.diff(signal_evttime), bins=100, histtype='step', label='Signal')
    ax.hist(np.diff(ray_evttime), bins=100, histtype='step', label='Ray')
    ax.set_yscale('log')
    ax.legend()

    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()
