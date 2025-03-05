#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 05 13:34 2025
Created in PyCharm
Created as saclay_micromegas/understand_timestamp

@author: Dylan Neff, dn277127
"""

import platform
import os
import numpy as np
import matplotlib.pyplot as plt

import uproot


def main():
    if platform.system() == 'Windows':
        base_path = 'F:/Saclay/cosmic_data/'
    elif platform.system() == 'Linux':
        base_path = '/local/home/dn277127/Bureau/cosmic_data/'
    else:
        print('Unknown platform')
        return
    test_root_path = f'{base_path}ip1_test_2-27-25/long_run_1/filtered_root/'

    event_nums, timestamps, delta_timestamps = [], [], []
    for root_file in os.listdir(test_root_path):
        if root_file.endswith('.root') and '_06_' in root_file and '_array' in root_file and '_002_' in root_file:
            print(root_file)
            root_path = f'{test_root_path}{root_file}'
            root = uproot.open(root_path)
            tree = root['nt']
            timestamps_i = tree['timestamp'].array()
            event_nums_i = tree['eventId'].array()
            delta_timestamps_i = tree['delta_timestamp'].array()
            timestamps.extend(timestamps_i)
            event_nums.extend(event_nums_i)
            delta_timestamps.extend(delta_timestamps_i)

    # Sort timestamps and event_nums by event_nums
    timestamps = np.array(timestamps)
    event_nums = np.array(event_nums)
    delta_timestamps = np.array(delta_timestamps)
    sort_inds = np.argsort(event_nums)
    timestamps = timestamps[sort_inds]
    event_nums = event_nums[sort_inds]
    delta_timestamps = delta_timestamps[sort_inds]

    # Make array of 64 bit binary timestamps
    timestamps_bin = np.array([np.binary_repr(ts, 65) for ts in timestamps])

    min_print_event, max_print_event = 54800, 55100
    for event_i, timestamp_bin_i, timestamp_i in zip(event_nums, timestamps_bin, timestamps):
        if min_print_event <= event_i <= max_print_event:
            print(f'Event {event_i:5d}:   {timestamp_bin_i},  {timestamp_i:e}')
        if (event_i > max_print_event):
            break

    # Plot timestamps vs event_nums
    fig, ax = plt.subplots()
    ax.plot(event_nums, timestamps, '.')
    ax.set_xlabel('Event Number')
    ax.set_ylabel('Timestamp')
    ax.set_title(f'{root_file}')

    # Plot delta_timestamps vs event_nums
    fig, ax = plt.subplots()
    ax.plot(event_nums, delta_timestamps, '.')
    ax.set_xlabel('Event Number')
    ax.set_ylabel('Delta Timestamp')
    ax.set_title(f'{root_file}')

    # Plot delta_timestamps vs timestamps
    fig, ax = plt.subplots()
    ax.plot(timestamps, delta_timestamps, '.')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Delta Timestamp')
    ax.set_title(f'{root_file}')

    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
