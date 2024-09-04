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
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

import uproot
import awkward as ak

# matplotlib.use('TkAgg')  # or another available backend

def main():
    plot_event_num()
    print('donzo')


def plot_event_num():
    # rays_path = 'F:/Saclay/cosmic_data/sg1_stats_7-26-24/max_hv_long_1/m3_tracking_root/'
    rays_path = '/local/home/dn277127/Bureau/cosmic_data/sg1_stats_7-26-24/max_hv_long_1/m3_tracking_root/'
    file_dates_path = '/local/home/dn277127/Bureau/cosmic_data/sg1_stats_7-26-24/max_hv_long_1/file_dates.txt'

    file_dates, file_nums = [], []
    with open(file_dates_path, 'r') as file:
        for line in file:
            file_num, date, time = line.split()
            time = time.split('.')[0]
            file_dates.append(datetime.strptime(f'{date} {time}', '%Y-%m-%d %H:%M:%S'))
            file_nums.append(int(file_num))
    file_dates, file_nums = zip(*sorted(zip(file_dates, file_nums)))
    times_per_file = np.diff(file_dates)
    times_per_file = [time.total_seconds() for time in times_per_file] + [0]

    file_num_dates_map = dict(zip(file_nums, file_dates))
    file_num_times_map = dict(zip(file_nums, times_per_file))

    dates, start_event_nums, file_nums, events_per_file, file_times = [], [], [], [], []
    for file_name in os.listdir(rays_path):
        with uproot.open(rays_path + file_name) as file:
            file_num = int(file_name.split('_')[-2])
            # if file_num < 500 or file_num > 700:
            #     continue
            tree = file[file.keys()[-1]]
            event_nums = tree['evn'].array()
            # if event_nums[0] < 0.85e6:
            # if event_nums[0] < 1.6e6:
            #     continue
            # if event_nums[-1] > 0.95e6:
            #     continue
            mod_date = datetime.fromtimestamp(os.path.getctime(rays_path + file_name))
            # print(file_name)
            # print(mod_date)
            # print(event_nums[0], event_nums[-1])
            # dates.append(mod_date)
            dates.append(file_num_dates_map[file_num])
            start_event_nums.append(event_nums[0])
            file_nums.append(file_num)
            events_per_file.append(len(event_nums))
            file_times.append(file_num_times_map[file_num])

            print(file_num, event_nums[0], event_nums[-1], mod_date)

    fig, ax = plt.subplots()
    ax.plot(file_nums, events_per_file, 'o')
    ax.set_xlabel('File Number')
    ax.set_ylabel('Events per File')

    fig2, ax2 = plt.subplots()
    ax2.plot(file_nums, start_event_nums, 'o')
    ax2.set_xlabel('File Number')
    ax2.set_ylabel('Event Number')

    fig3, ax3 = plt.subplots()
    ax3.plot(dates, file_times, 'o')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Times per file (s)')

    fig4, ax4 = plt.subplots()
    ax4.plot(file_num, file_times, 'o')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Events per File')

    plt.show()


def write_file_dates():
    rays_path = '/mnt/cosmic_data/Run/sg1_stats_7-26-24/max_hv_long_1/m3_tracking_root/'
    dates, file_nums = [], []
    for file_name in os.listdir(rays_path):
        file_num = int(file_name.split('_')[-2])
        mod_date = datetime.fromtimestamp(os.path.getctime(rays_path + file_name))
        dates.append(mod_date)
        file_nums.append(file_num)

        print(file_num, mod_date)

    with open('file_dates.txt', 'w') as file:
        for i in range(len(file_nums)):
            file.write(f'{file_nums[i]}\t{dates[i]}\n')


if __name__ == '__main__':
    main()
