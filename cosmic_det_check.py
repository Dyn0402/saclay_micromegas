#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 15 6:02 AM 2024
Created in PyCharm
Created as saclay_micromegas/cosmic_det_check.py

@author: Dylan Neff, Dylan
"""


import os

import uproot
import awkward as ak
import vector

import json

from Measure import Measure

from fe_analysis import *


def main():
    vector.register_awkward()
    main_dir = 'C:/Users/Dylan/Desktop/hv_test/'
    decoded_dir = f'{main_dir}decoded_root/'
    m3_dir = f'{main_dir}m3_tracking_root/'
    run_json_path = f'{main_dir}run_config.json'
    file_nums = [0, 1]
    run_data = get_det_data(run_json_path)
    print(run_data)
    ray_data = get_ray_data(m3_dir, file_nums)
    dream_data, dream_event_ids = get_dream_data(decoded_dir, file_nums)
    hit_events_dict = get_hit_events(dream_data, dream_event_ids, run_data)
    plot_hits(hit_events_dict, ray_data)
    plt.show()
    print('donzo')


def plot_hits(hit_events_dict, ray_data):
    ray_event_ids = ray_data['evn']
    n_ray_events = len(ray_event_ids)

    for det, det_data in hit_events_dict.items():
        n_det_hits = len(det_data['hit_event_ids'])
        mask = np.isin(ray_event_ids, det_data['hit_event_ids'])
        one_track_mask = np.array([x.size == 1 for x in ray_data['X_Up']])
        mask = mask & one_track_mask

        z_up, z_down = ray_data['Z_Up'][mask], ray_data['Z_Down'][mask]
        x_up, x_down = ray_data['X_Up'][mask], ray_data['X_Down'][mask]
        y_up, y_down = ray_data['Y_Up'][mask], ray_data['Y_Down'][mask]

        x_up, x_down = np.array([x[0] for x in x_up]), np.array([x[0] for x in x_down])
        y_up, y_down = np.array([y[0] for y in y_up]), np.array([y[0] for y in y_down])

        det_z = det_data['z']

        # Calculate the interpolation factors
        t = (det_z - z_up) / (z_down - z_up)
        # print(t)
        t = np.mean(t)

        # Interpolate the x and y positions
        x_positions = x_up + t * (x_down - x_up)
        y_positions = y_up + t * (y_down - y_up)

        fig, ax = plt.subplots()
        ax.set_title(f'{det} hits')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        # Draw a thin square at -100, -100, 100, 100
        ax.plot([-100, 100, 100, -100, -100], [-100, -100, 100, 100, -100], color='black', linewidth=0.5)
        ax.scatter(x_positions, y_positions, marker='o', color='blue', alpha=0.5)
        ax.scatter(0, 0, marker='x', color='red')
        text = f'{n_det_hits} of {n_ray_events} hits\n{n_det_hits / n_ray_events * 100:.2f}%'
        ax.annotate(text, (0.2, 0.92), xycoords='axes fraction', fontsize=14, bbox=dict(facecolor='wheat', alpha=0.5),
                    ha='center', va='center')
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)

        fig.tight_layout()


def get_hit_events(dream_data, dream_event_ids, run_data):
    dets = [det for det in run_data['included_detectors'] if 'm3_' not in det and 'banco' not in det]
    print(dets)
    det_zs, det_feus = [], []
    for det in dets:
        for det_info in run_data['detectors']:
            if det_info['name'] == det:
                det_zs.append(det_info['det_center_coords']['z'])
                det_feus.append(det_info['dream_feus']['x_1'])

    print(det_zs)
    print(det_feus)
    print(dream_data.keys())
    print(dream_data)

    det_data = {det: {'z': z, 'feu': feu} for det, z, feu in zip(dets, det_zs, det_feus)}
    for det, z, feu in zip(dets, det_zs, det_feus):
        data = dream_data[feu[0]][feu[1]]
        print(f'\n{det}, z={z}, {feu}, data shape: {data.shape}')
        data_sample_maxes = get_sample_max(data)
        data_event_maxes = get_sample_max(data_sample_maxes)
        hit_event_ids = dream_event_ids[data_event_maxes > 500]
        det_data[det].update({'hit_event_ids': hit_event_ids})
    print(det_data)

    return det_data


def get_ray_data(ray_dir, file_nums):
    variables = ['evn', 'evttime', 'rayN', 'Z_Up', 'X_Up', 'Y_Up', 'Z_Down', 'X_Down', 'Y_Down', 'Chi2X', 'Chi2Y']
    data = None
    for file_name in os.listdir(ray_dir):
        if not file_name.endswith('.root'):
            continue

        file_num = int(file_name.split('_')[-2])
        if file_num not in file_nums:
            continue

        with uproot.open(f'{ray_dir}{file_name}') as file:
            print(file.keys())
            tree = file['T;1']
            print(tree.keys())
            new_data = tree.arrays(variables, library='np')
            if data is None:
                data = new_data
            else:
                for var in variables:
                    data[var] = np.concatenate((data[var], new_data[var]))

    return data


def get_dream_data(dream_dir, file_nums):
    variables = ['eventId', 'timestamp', 'delta_timestamp', 'ftst', 'amplitude']
    dets, event_ids = {}, {}
    for file_name in os.listdir(dream_dir):
        if not file_name.endswith('.root') or '_array' not in file_name or '_datrun_' not in file_name:
            continue

        file_num = int(file_name.split('_')[-4])
        if file_num not in file_nums:
            continue

        feu_num = int(file_name.split('_')[-3])

        print(f'\n{file_name}')
        with uproot.open(f'{dream_dir}{file_name}') as file:
            print(file.keys())
            tree = file[file.keys()[0]]
            print(tree.keys())
            data = tree.arrays(variables, library='np')
            print(data['eventId'])
            print(data['amplitude'].shape)
            new_event_ids = data['eventId']
            if file_num not in event_ids:
                event_ids.update({file_num: new_event_ids})
            det1, det2 = np.split(data['amplitude'], 2, axis=1)
            if feu_num not in dets:
                dets.update({feu_num: {1: det1, 5: det2}})
            else:
                dets[feu_num][1] = np.concatenate((dets[feu_num][1], det1))
                dets[feu_num][5] = np.concatenate((dets[feu_num][5], det2))

    event_ids = np.concatenate([event_ids[file_num] for file_num in file_nums])
    return dets, event_ids


def get_det_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


if __name__ == '__main__':
    main()
