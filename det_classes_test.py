#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 09 4:03 PM 2024
Created in PyCharm
Created as saclay_micromegas/det_classes_test.py

@author: Dylan Neff, Dylan
"""

from DetectorConfigLoader import DetectorConfigLoader
from Detector import Detector
from DreamDetector import DreamDetector
from DreamData import DreamData


def main():
    run_dir = 'F:/Saclay/cosmic_data/hv_scan_7-6-24/'
    run_json_path = f'{run_dir}run_config.json'
    det_type_info_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    data_dir = 'F:/Saclay/cosmic_data/hv_scan_7-6-24/hv1/filtered_root/'
    ped_dir = 'F:/Saclay/cosmic_data/hv_scan_7-6-24/hv1/decoded_root/'

    det_config_loader = DetectorConfigLoader(run_json_path, det_type_info_dir)
    for detctor_name in det_config_loader.included_detectors:
        if detctor_name != 'urw_inter':
            continue

        print(detctor_name)
        det_config = det_config_loader.get_det_config(detctor_name)
        # print(det_config)
        if det_config['det_type'] == 'm3':
            continue
        if det_config['det_type'] == 'banco':
            det = Detector(config=det_config)
        else:  # Dream
            det = DreamDetector(config=det_config)
            print(f'FEU Num: {det.feu_num}')
            print(f'FEU Channels: {det.feu_channels}')
            print(f'HV: {det.hv}')

        print(f'Name: {det.name}')
        print(f'Center: {det.center}')
        print(f'Size: {det.size}')
        print(f'Rotations: {det.rotations}')

        dream_data = DreamData(data_dir, det.feu_num, det.feu_channels, ped_dir)
        dream_data.read_ped_data()
        dream_data.read_data()

        print('\n')

    print('donzo')


if __name__ == '__main__':
    main()
