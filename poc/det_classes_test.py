#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 09 4:03 PM 2024
Created in PyCharm
Created as saclay_micromegas/det_classes_test.py

@author: Dylan Neff, Dylan
"""

import matplotlib.pyplot as plt

from DetectorConfigLoader import DetectorConfigLoader
from Detector import Detector
from DreamDetector import DreamDetector
from DreamData import DreamData


def main():
    # run_dir = 'F:/Saclay/cosmic_data/hv_scan_7-6-24/'
    # det_type_info_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    run_dir = '/local/home/dn277127/Bureau/cosmic_data/hv_scan_7-6-24/'
    det_type_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    run_json_path = f'{run_dir}run_config.json'
    data_dir = f'{run_dir}/hv1/filtered_root/'
    ped_dir = f'{run_dir}/hv1/decoded_root/'

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
            print(f'FEU Channels: {det.feu_connectors}')
            print(f'HV: {det.hv}')

        print(f'Name: {det.name}')
        print(f'Center: {det.center}')
        print(f'Size: {det.size}')
        print(f'Rotations: {det.rotations}')

        dream_data = DreamData(data_dir, det.feu_num, det.feu_connectors, ped_dir)
        dream_data.read_ped_data()
        # dream_data.plot_pedestal_fit(50)
        # dream_data.plot_pedestals()
        # plt.show()
        dream_data.read_data()
        # dream_data.subtract_pedestals_from_data()
        # dream_data.get_event_amplitudes()
        dream_data.plot_event_amplitudes()
        dream_data.plot_event_mean_times()
        dream_data.plot_event_fit_success()
        dream_data.plot_fit_param('time_shift')
        dream_data.plot_fit_param('q')
        param_ranges = {'time_shift': [2, 8], 'q': [0.55, 0.7], 'amplitude': [500, 1000], 'mean': [5, 10]}
        dream_data.plot_fits(param_ranges)
        param_ranges = {'time_shift': [2, 8], 'q': [0.55, 0.7], 'amplitude': [10, 50], 'mean': [5, 10]}
        dream_data.plot_fits(param_ranges)
        plt.show()
        print(dream_data.data_amps)

        print('\n')

    print('donzo')


if __name__ == '__main__':
    main()
