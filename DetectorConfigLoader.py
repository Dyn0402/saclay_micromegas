#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 09 3:06 PM 2024
Created in PyCharm
Created as saclay_micromegas/DetectorConfigLoader.py

@author: Dylan Neff, Dylan
"""


import json


class DetectorConfigLoader:
    def __init__(self, run_config_json_path, det_type_info_dir=None):
        self.config = load_json_file(run_config_json_path)
        self.included_detectors = self.config['included_detectors']
        self.sub_run_names = [sub_run['sub_run_name'] for sub_run in self.config['sub_runs']]
        self.det_type_info_dir = det_type_info_dir
        self.det_maps = None

    def get_det_config(self, det_name, sub_run_name=None):
        if det_name not in self.config['included_detectors']:
            print(f'Error: Detector {det_name} not in included detectors.')
            return None

        det_config = next((det for det in self.config['detectors'] if det['name'] == det_name), None)
        if det_config is None:
            print(f'Error: Detector {det_name} not found in run config.')
            return None

        if sub_run_name is None:
            sub_run_name = self.sub_run_names[0]
            if len(self.sub_run_names) > 1:
                print(f'Warning: Sub run not specified, using {sub_run_name}.')
        sub_run = next((sub_run for sub_run in self.config['sub_runs'] if sub_run['sub_run_name'] == sub_run_name), None)
        if sub_run is None:
            print(f'Error: Sub run {sub_run_name} not found for detector {det_name}.')
        else:
            det_config.update({'run_time': sub_run['run_time']})
            if isinstance(det_config['hv_channels'], dict):
                hvs = get_hvs(det_config['hv_channels'], sub_run['hvs'])
                det_config.update({'hvs': hvs})
        if self.det_type_info_dir is not None:
            det_type_info = load_json_file(self.det_type_info_dir + det_config['det_type'] + '.json')
            det_config.update({key: val for key, val in det_type_info.items()})

        return det_config


def get_hvs(det_hv_channels, sub_run_hvs):
    hvs = {}
    for hv_name, hv_channels in det_hv_channels.items():
        hv_slot = sub_run_hvs.get(str(hv_channels[0]), None)
        if hv_slot is None:
            print(f'Warning {hv_name} HV slot {hv_channels[0]} not found in sub run HVs, setting to 0.')
            hvs.update({hv_name: 0})
        else:
            hv_chan = hv_slot.get(str(hv_channels[1]), None)
            if hv_chan is None:
                print(f'Warning {hv_name} HV channel {hv_channels[1]} not found in sub run HVs, setting to 0.')
                hvs.update({hv_name: 0})
            else:
                hvs.update({hv_name: hv_chan})

    return hvs


def load_json_file(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    return json_data


def load_det_maps(det_map_dir):
    for file in os.listdir(det_map_dir):
        is

