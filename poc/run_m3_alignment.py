#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 03 21:11 2024
Created in PyCharm
Created as saclay_micromegas/run_m3_alignment

@author: Dylan Neff, dn277127
"""

import os
import platform

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit as cf
from scipy.stats import skewnorm, alpha

from M3RefTracking import M3RefTracking
from DetectorConfigLoader import DetectorConfigLoader
from Detector import Detector
from DreamDetector import DreamDetector
from DreamData import DreamData

from det_classes_test import plot_ray_hits_2d, align_dream, get_residuals


def main():
    # Check if platform is Windows or Linux
    if platform.system() == 'Windows':
        base_dir = 'F:/Saclay/cosmic_data/'
        det_type_info_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
        chunk_size = 100  # Number of files to process at once
    elif platform.system() == 'Linux':
        base_dir = '/local/home/dn277127/Bureau/cosmic_data/'
        det_type_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
        chunk_size = 7  # Number of files to process at once
    else:
        raise OSError('Unknown platform')
    run_name = 'urw_stats_10-31-24'
    run_dir = f'{base_dir}{run_name}/'
    sub_run_name = 'long_run'

    # det_single = 'urw_inter'
    det_single = 'urw_strip'

    # file_nums = 'all'
    file_nums = list(range(0, 25))

    realign_dream = True  # If False, read alignment from file, if True, realign Dream detector

    run_json_path = f'{run_dir}run_config.json'
    data_dir = f'{run_dir}{sub_run_name}/filtered_root/'
    ped_dir = f'{run_dir}{sub_run_name}/decoded_root/'
    m3_dir = f'{run_dir}{sub_run_name}/m3_tracking_root/'

    alignment_dir = f'{run_dir}alignments/'
    try:
        os.mkdir(alignment_dir)
    except FileExistsError:
        pass

    z_align_range = [5, 5]  # mm range to search for optimal z position

    det_config_loader = DetectorConfigLoader(run_json_path, det_type_info_dir)

    print(f'Getting ray data...')
    ray_data = M3RefTracking(m3_dir, single_track=True, file_nums=file_nums)

    for detector_name in det_config_loader.included_detectors:
        if det_single is not None and detector_name != det_single:
            continue

        print(detector_name)
        det_config = det_config_loader.get_det_config(detector_name, sub_run_name=sub_run_name)
        if det_config['det_type'] == 'm3':
            continue
        if det_config['det_type'] == 'banco':
            continue
        else:  # Dream
            det = DreamDetector(config=det_config)
            print(f'FEU Num: {det.feu_num}')
            print(f'FEU Channels: {det.feu_connectors}')
            print(f'HV: {det.hv}')
            det.load_dream_data(data_dir, ped_dir, 8, file_nums, chunk_size,
                                waveform_fit_func='parabola_vectorized', save_waveforms=False)
            print(f'Hits shape: {det.dream_data.hits.shape}')
            print(f'feu_connectors: {det.dream_data.feu_connectors}')
            det.dream_data.correct_for_fine_timestamps()

            det.make_sub_detectors()

            if realign_dream:
                det.add_rotation(90, 'z')

            plot_ray_hits_2d(det, ray_data)
            det.plot_hits_1d()

            # Print first few event numbers for ray data then Dream data
            print(f'Ray event numbers: {ray_data.get_xy_positions(det.center[2])[-1][:5]}')
            print(f'Dream event numbers: {det.dream_data.event_nums[:5]}')
            # input('Press enter to continue')

            x_res_i_mean, y_res_i_mean, x_res_i_std, y_res_i_std = get_residuals(det, ray_data)
            det.set_center(x=det.center[0] - x_res_i_mean, y=det.center[1] - y_res_i_mean)
            get_residuals(det, ray_data, plot=True)

            z_orig = det.center[2]
            x_bnds = det.center[0] - det.size[0] / 2, det.center[0] + det.size[0] / 2
            y_bnds = det.center[1] - det.size[1] / 2, det.center[1] + det.size[1] / 2
            # ray_traversing_triggers = ray_data.get_traversing_triggers(z_orig, x_bnds, y_bnds, expansion_factor=0.1)

            alignment_file = f'{alignment_dir}{det.name}_alignment.txt'
            if realign_dream:
                align_dream(det, ray_data, z_align_range)
                det.write_det_alignment_to_file(alignment_file)
            else:
                det.read_det_alignment_from_file(alignment_file)
            plot_ray_hits_2d(det, ray_data)

            plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
