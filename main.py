#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 05 11:29 2023
Created in PyCharm
Created as saclay_micromegas/main

@author: Dylan Neff, dn277127
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from fe_analysis import *


def main():
    # base_path = '/local/home/dn277127/Documents/TestBeamData/2023_July_Saclay/dec6/'
    base_path = 'F:/Saclay/'
    data_base = f'{base_path}TestBeamData/2023_July_Saclay/dec6/'
    # base_path = '/media/ucla/Saclay/TestBeamData/2023_July_Saclay/dec6/'
    fdf_dir = base_path
    raw_root_dir = f'{data_base}raw_root/'
    ped_flag = '_pedthr_'
    connected_channels = load_connected_channels()  # Hard coded into function

    # process_fdfs(fdf_dir, raw_root_dir)
    run_full_analysis(base_path, raw_root_dir, ped_flag, connected_channels)
    # single_file_analysis(raw_root_dir, ped_flag, connected_channels)
    # get_run_periods(fdf_dir, ped_flag)

    print('donzo')


def run_full_analysis(base_path, raw_root_dir, ped_flag, connected_channels):
    num_threads = 15
    free_memory = 2.0  # GB of memory to allocate (in theory, in reality needs a lot of wiggle room)
    chunk_size = 25000
    print(f'{num_threads} threads, {chunk_size} chunk size')
    # run_files = ['P22_P2_2_ME_400_P2_2_DR_1000']  # If 'all' run all files found
    run_files = 'all'  # If 'all' run all files found
    ped_time = '_231206_14H51_'
    out_directory = f'{base_path}Analysis/'
    out_file_path = f'{out_directory}analysis_data.txt'

    num_detectors = 2
    noise_sigmas = 8

    ped_files = [file for file in os.listdir(raw_root_dir) if file.endswith('.root') and ped_flag in file]
    ped_file = ped_files[0] if len(ped_files) == 0 else [file for file in ped_files if ped_time in file][0]
    if len(ped_files) > 1:
        print(f'Warning: Multiple ped files found: {ped_files}.\nUsing {ped_file}')

    # Get pedestal data
    ped_root_path = os.path.join(raw_root_dir, ped_file)
    pedestals, noise_thresholds = run_pedestal(ped_root_path, num_detectors, noise_sigmas, connected_channels)

    data_files = [os.path.join(raw_root_dir, file) for file in os.listdir(raw_root_dir)
                  if file.endswith('.root') and ped_flag not in file and
                  (run_files == 'all' or any(run_file in file for run_file in run_files))]

    process_data = [(file, pedestals, noise_thresholds, num_detectors, connected_channels, chunk_size, out_directory)
                    for file in data_files]
    file_data = []
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        with tqdm(total=len(process_data), desc='Processing Trees') as pbar:
            for file_res in executor.map(analyze_file, *zip(*process_data)):
                if file_res is not None:
                    file_data.append(file_res)
                pbar.update(1)

    write_to_file(file_data, out_file_path)

    run_periods = get_run_periods(raw_root_dir, ped_flag, plot=True)

    peak_analysis(file_data, run_periods)

    plt.show()


def single_file_analysis(raw_root_dir, ped_flag, connected_channels):
    chunk_size = 3500
    # file_name = 'P22_P2_2_ME_400_P2_2_DR_1000'
    # file_name = 'P22_P2_2_ME_400_P2_2_DR_1000_231213_15H46'  # Easier one
    file_name = 'P22_P2_2_ME_400_P2_2_DR_500_231213_11H17'  # Harder one
    ped_time = '_231206_14H51_'

    num_detectors = 2
    noise_sigmas = 8
    plot_pedestals = False

    ped_files = [file for file in os.listdir(raw_root_dir) if file.endswith('.root') and ped_flag in file]
    ped_file = ped_files[0] if len(ped_files) == 0 else [file for file in ped_files if ped_time in file][0]
    if len(ped_files) > 1:
        print(f'Warning: Multiple ped files found: {ped_files}.\nUsing {ped_file}')

    # Get pedestal data
    ped_root_path = os.path.join(raw_root_dir, ped_file)
    pedestals, noise_thresholds = run_pedestal(ped_root_path, num_detectors, noise_sigmas, connected_channels,
                                               plot_pedestals)

    data_files = [os.path.join(raw_root_dir, file) for file in os.listdir(raw_root_dir)
                  if file.endswith('.root') and file_name in file]
    data_file = data_files[0]
    if len(data_files) > 1:
        print(f'Warning: Multiple data files found: {data_files}.\nUsing {data_file}')

    analyze_file_qa(data_file, pedestals, noise_thresholds, num_detectors, connected_channels, chunk_size)

    plt.show()


def process_fdfs(fdf_dir, raw_root_dir):
    num_threads = 15
    overwrite = False

    fdf_files = [file for file in os.listdir(fdf_dir) if file.endswith('.fdf')]
    fdf_data_list = [(file, fdf_dir, raw_root_dir, overwrite, file_i) for file_i, file in enumerate(fdf_files)]
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        with tqdm(total=len(fdf_files), desc='Processing fdfs') as pbar:
            for root_name in executor.map(process_fdf, *zip(*fdf_data_list)):
                pbar.update(1)


if __name__ == '__main__':
    main()
