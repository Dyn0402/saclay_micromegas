#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 03 1:01 PM 2024
Created in PyCharm
Created as saclay_micromegas/m3_ref_hv_cal.py

@author: Dylan Neff, Dylan
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak
import vector

from dream_functions import *
from cosmic_det_check import get_det_data


def main():
    vector.register_awkward()
    hvs = [440, 445, 450, 455, 460]
    # hvs = [440]
    # root_dir = 'F:/Saclay/cosmic_data/hv_scan_7-4-24/'
    root_dir = 'F:/Saclay/cosmic_data/hv_scan_7-6-24/'
    run_json_path = f'{root_dir}run_config.json'
    ped_dir_name = 'decoded_root'
    dat_dir_name = 'filtered_root'
    plot_max_dists = False

    run_data = get_det_data(run_json_path)
    sub_runs = run_data['sub_runs']
    print(sub_runs)
    sub_runs_dict = {sub_run['sub_run_name']: sub_run for sub_run in sub_runs}
    sub_run_names = [sub_run['sub_run_name'] for sub_run in sub_runs]
    print(sub_run_names)
    detectors = run_data['detectors']
    print(detectors)
    detector_dict = {det['name']: det for det in detectors}
    included_detectors = run_data['included_detectors']
    print(included_detectors)
    test_detector_names = [det for det in included_detectors if 'banco_ladder' not in det and 'm3_' not in det]
    print(test_detector_names)
    for det in test_detector_names:
        print(f'{det}: {detector_dict[det]["dream_feus"]}')
        feu_channels = [chan for orientation, chan in detector_dict[det]['dream_feus'].items()]
        print(feu_channels)
        feu_nums = set([chan[0] for chan in feu_channels])
        if len(feu_nums) != 1:
            print(f'Error: {det} has multiple FEUs: {feu_nums}')
        else:
            detector_dict[det]['dream_feu_num'] = list(feu_nums)[0]
        chan_nums = sorted([chan[1] for chan in feu_channels])
        print(chan_nums)
        detector_dict[det]['dream_chan_nums'] = chan_nums
    noise_sigmas = 4

    gain_data = {det_name: {'hv': [], 'gain_hv': [], 'gain_x': [], 'gain_y': [], 'num_x': [], 'num_y': [], 'num_og': []}
                 for det_name in test_detector_names}
    for sub_run_name in sub_run_names:
        sub_run = sub_runs_dict[sub_run_name]
        sub_run_hvs = sub_run['hvs']
        sub_run_dir = f'{root_dir}{sub_run_name}/'
        ped_dir = f'{sub_run_dir}{ped_dir_name}/'
        dat_dir = f'{sub_run_dir}{dat_dir_name}/'

        ped_dir_files = os.listdir(ped_dir)
        dat_dir_files = os.listdir(dat_dir)

        for det_name in test_detector_names:
            print(f'\n{det_name} in {sub_run_name}')
            det_info = detector_dict[det_name]
            chan_nums, feu_num = det_info['dream_chan_nums'], det_info['dream_feu_num']
            hv_chans = det_info['hv_channels']
            drift_hv_chan = hv_chans['drift']
            mesh_resist_hv_chan = hv_chans['mesh_1'] if 'mesh_1' in hv_chans else hv_chans['resist_1']
            mesh_resist_hv_chan = hv_chans['resist_2'] if 'resist_2' in hv_chans else mesh_resist_hv_chan
            print(f' HV Chans: {drift_hv_chan}, {mesh_resist_hv_chan}')
            drift_hv = sub_run_hvs[str(drift_hv_chan[0])][str(drift_hv_chan[1])]
            mesh_resist_hv = sub_run_hvs[str(mesh_resist_hv_chan[0])][str(mesh_resist_hv_chan[1])]
            print(f' HVs: {drift_hv}, {mesh_resist_hv}')

            ped_file = [file for file in ped_dir_files if '_array' in file and '_pedthr_' in file and
                        get_num_from_fdf_file_name(file, -1) == feu_num]
            if len(ped_file) != 1:
                print(f'Error: {det_name} single ped file not found in {ped_dir}: {ped_file}')
                continue
            else:
                ped_file = ped_file[0]

            dat_files = [file for file in dat_dir_files if '_array' in file and '_datrun_' in file and
                         get_num_from_fdf_file_name(file, -1) == feu_num]

            peds = read_det_data(f'{ped_dir}{ped_file}', num_detectors=chan_nums, tree_name='nt')
            peds = np.reshape(peds, (peds.shape[0], len(chan_nums), peds.shape[1] // len(chan_nums), peds.shape[2]))
            pedestals = get_pedestals(peds)
            ped_common_noise = get_common_noise(peds, pedestals)
            ped_fits = get_pedestal_fits(peds, common_noise=ped_common_noise)
            pedestals, ped_rms = ped_fits['mean'], ped_fits['sigma']
            ped_thresholds = get_noise_thresholds(ped_rms, noise_sigmas=noise_sigmas)

            data = []
            for dat_file in dat_files:
                data_i = read_det_data(f'{dat_dir}{dat_file}', num_detectors=chan_nums, tree_name='nt')
                data_i = np.reshape(data_i, (data_i.shape[0], len(chan_nums), data_i.shape[1] // len(chan_nums), data_i.shape[2]))
                data.append(data_i)
            data = np.concatenate(data, axis=0)
            data = subtract_pedestal(data, pedestals)

            # Split data on second axis, such that 2 channel numbers go to x and 2 go to y
            data_x, data_y = np.split(data, 2, axis=1)
            ped_thresholds_x, ped_thresholds_y = np.split(ped_thresholds, 2)

            data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1] * data_x.shape[2], data_x.shape[3]))
            data_y = np.reshape(data_y, (data_y.shape[0], data_y.shape[1] * data_y.shape[2], data_y.shape[3]))
            ped_thresholds_x = np.reshape(ped_thresholds_x, (ped_thresholds_x.shape[0] * ped_thresholds_x.shape[1]))
            ped_thresholds_y = np.reshape(ped_thresholds_y, (ped_thresholds_y.shape[0] * ped_thresholds_y.shape[1]))

            # data_x = filter_noise_events(data_x, ped_thresholds_x)
            # data_y = filter_noise_events(data_y, ped_thresholds_y)
            x_mask = filter_noise_events(data_x, ped_thresholds_x, return_type='mask')
            y_mask = filter_noise_events(data_y, ped_thresholds_y, return_type='mask')
            xy_mask = np.logical_and(x_mask, y_mask)
            data_x = data_x[xy_mask]
            data_y = data_y[xy_mask]

            sample_maxes_x, sample_maxes_y = get_sample_max(data_x), get_sample_max(data_y)

            event_maxes_x, event_maxes_y = get_sample_max(sample_maxes_x), get_sample_max(sample_maxes_y)
            x_median, y_median = np.median(event_maxes_x), np.median(event_maxes_y)
            x_mean, y_mean = np.mean(event_maxes_x), np.mean(event_maxes_y)
            gain_data[det_name]['hv'].append(mesh_resist_hv)
            if '_m3_' not in sub_run_name:
                gain_data[det_name]['gain_hv'].append(mesh_resist_hv)
                gain_data[det_name]['gain_x'].append(x_mean)
                gain_data[det_name]['gain_y'].append(y_mean)
            gain_data[det_name]['num_x'].append(data_x.shape[0])
            gain_data[det_name]['num_y'].append(data_y.shape[0])
            gain_data[det_name]['num_og'].append(data.shape[0])
            print(f' X Events: {data_x.shape[0]}, Y Events: {data_y.shape[0]}')
            print(f' X Median: {x_median:.2f}, Y Median: {y_median:.2f}')
            print(f' X Mean: {x_mean:.2f}, Y Mean: {y_mean:.2f}')

            if plot_max_dists:
                fig, ax = plt.subplots()
                ax.hist(event_maxes_x, bins=np.arange(-0.5, 4200.5, 50), color='blue', histtype='step', label='X')
                ax.hist(event_maxes_y, bins=np.arange(-0.5, 4200.5, 50), color='green', histtype='step', label='Y')
                ax.axvline(x_median, color='blue', ls='--', label=f'X median: {x_median:.2f}')
                ax.axvline(y_median, color='green', ls='--', label=f'Y median: {y_median:.2f}')
                ax.set_yscale('log')
                ax.set_xlabel('ADC')
                ax.set_ylabel('Events')
                ax.set_title(f'{det_name} Event Maxes, Drift = {drift_hv}V, Mesh/Resist = {mesh_resist_hv}V')
                ax.legend()
                fig.tight_layout()

    # Plot gain curves
    fig, ax = plt.subplots()
    fig_nevents, ax_nevents = plt.subplots()
    colors = ['blue', 'green', 'maroon', 'purple', 'orange']
    for i, (det_name, data) in enumerate(gain_data.items()):
        ax.plot(data['gain_hv'], data['gain_x'], ls='-', color=colors[i], marker='o', alpha=0.7, label=f'{det_name} X')
        ax.plot(data['gain_hv'], data['gain_y'], ls='-', color=colors[i], marker='s', alpha=0.7, label=f'{det_name} Y')

        ax_nevents.plot(data['hv'], data['num_x'], color=colors[i], alpha=0.5, marker='o', label=f'{det_name} X')
        ax_nevents.plot(data['hv'], data['num_y'], color=colors[i], alpha=0.5, marker='s', label=f'{det_name} Y')
        ax_nevents.plot(data['hv'], data['num_og'], color=colors[i], alpha=0.5, marker='x', label=f'{det_name} OG')

    ax.set_xlabel('Mesh/Resist Voltage (V)')
    ax.set_ylabel('ADC')
    ax.set_title('Gain Curves')
    ax.legend()
    fig.tight_layout()

    ax_nevents.set_xlabel('Mesh/Resist Voltage (V)')
    ax_nevents.set_ylabel('Events')
    ax_nevents.set_title('Number of Events')
    ax_nevents.legend()
    fig_nevents.tight_layout()

    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
