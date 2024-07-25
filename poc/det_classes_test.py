#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 09 4:03 PM 2024
Created in PyCharm
Created as saclay_micromegas/det_classes_test.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt

from M3RefTracking import M3RefTracking
from DetectorConfigLoader import DetectorConfigLoader
from Detector import Detector
from DreamDetector import DreamDetector
from DreamData import DreamData


def main():
    # base_dir = 'F:/Saclay/cosmic_data/'
    # det_type_info_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    base_dir = '/local/home/dn277127/Bureau/cosmic_data/'
    det_type_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    # run_name = 'new_strip_check_7-12-24'
    run_name = 'ig1_test1'
    # run_name = 'banco_flipped_7-8-24'
    # run_name = 'ig1_sg1_stats4'
    run_dir = f'{base_dir}{run_name}/'
    # sub_run_name = 'hv1'
    # sub_run_name = 'new_detector_short'
    # sub_run_name = 'drift_600_resist_460'
    sub_run_name = 'quick_test'
    # sub_run_name = 'max_hv_long'

    # det_single = 'asacusa_strip_1'
    det_single = 'asacusa_strip_2'
    # det_single = 'strip_grid_1'
    # det_single = 'inter_grid_1'
    # det_single = 'urw_inter'
    # det_single = None

    run_json_path = f'{run_dir}run_config.json'
    data_dir = f'{run_dir}{sub_run_name}/filtered_root/'
    ped_dir = f'{run_dir}{sub_run_name}/decoded_root/'
    m3_dir = f'{run_dir}{sub_run_name}/m3_tracking_root/'

    ray_data = M3RefTracking(m3_dir, single_track=True)

    det_config_loader = DetectorConfigLoader(run_json_path, det_type_info_dir)
    for detctor_name in det_config_loader.included_detectors:
        if det_single is not None and detctor_name != det_single:
            continue

        print(detctor_name)
        det_config = det_config_loader.get_det_config(detctor_name, sub_run_name=sub_run_name)
        # print(det_config)
        if det_config['det_type'] == 'm3':
            continue
        if det_config['det_type'] == 'banco':
            det = Detector(config=det_config)
        else:  # Dream
            det = DreamDetector(config=det_config)
            # ray_data.plot_xy(det.center[2])
            # plt.show()
            print(f'FEU Num: {det.feu_num}')
            print(f'FEU Channels: {det.feu_connectors}')
            print(f'HV: {det.hv}')
            det.load_dream_data(data_dir, ped_dir)
            print(f'Hits shape: {det.dream_data.hits.shape}')
            # det.dream_data.plot_noise_metric()
            det.dream_data.plot_pedestals()
            det.dream_data.plot_hits_vs_strip()
            det.dream_data.plot_amplitudes_vs_strip()
            # plt.show()
            det.make_sub_detectors()
            event_nums = det.plot_xy_amp_sum_vs_event_num(500, 15)
            det.plot_amplitude_sum_vs_event_num()
            det.plot_num_hit_xy_hist()
            print(f'Det data: {len(det.dream_data.data)}')
            print(f'Ray data: {len(ray_data.ray_data)}')
            det.plot_centroids_2d()
            plt.show()

            x_res_i_mean, y_res_i_mean, x_res_i_std, y_res_i_std = get_residuals(det, ray_data)
            det.set_center(x=det.center[0] - x_res_i_mean, y=det.center[1] - y_res_i_mean)

            # zs = np.linspace(det.center[2], det.center[2] + 200, 100)
            zs = [det.center[2]]
            x_residuals, y_residuals = [], []
            for z in zs:
                det.set_center(z=z)
                # x_res_i, y_res_i = [], []
                # for sub_det_i, sub_det in enumerate(det.sub_detectors):
                #     event_nums_i, centroids_i = sub_det.get_event_centroids()
                #     event_nums_i = list(np.array(event_nums_i) - 0)
                #     x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(det.center[2], event_nums_i)
                #     centroids_i_matched = []
                #     for event_num in event_nums_i:
                #         if event_num in event_num_rays:
                #             centroids_i_matched.append(centroids_i[list(event_nums_i).index(event_num)])
                #     centroids_i_matched = np.array(centroids_i_matched)
                #     x_res_i.extend(centroids_i_matched[:, 0] - x_rays)
                #     y_res_i.extend(centroids_i_matched[:, 1] - y_rays)
                #     plot_xy_residuals_2d(x_rays, y_rays, centroids_i_matched[:, 0], centroids_i_matched[:, 1])
                # x_res_i_std, y_res_i_std = np.std(x_res_i), np.std(y_res_i)
                x_res_i_mean, y_res_i_mean, x_res_i_std, y_res_i_std = get_residuals(det, ray_data, plot=True)
                x_residuals.append(x_res_i_std)
                y_residuals.append(y_res_i_std)
            fig, ax = plt.subplots()
            ax.plot(zs, x_residuals, label='X Residuals')
            ax.plot(zs, y_residuals, label='Y Residuals')
            ax.set_xlabel('z (mm)')
            ax.set_ylabel('Residual (mm)')
            ax.legend()
            ax.grid()
            fig.tight_layout()
            plt.show()

            for event_num in event_nums:
                det.plot_event_1d(event_num)
                det.plot_event_2d(event_num)
                plt.show()
            for event_i, event in enumerate(det.dream_data.data_amps):
                print(f'Event #{event_i}: {np.sum(event)}')

            input('Press Enter to continue...')

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
        dream_data.plot_event_time_maxes()
        dream_data.plot_event_fit_success()
        dream_data.plot_fit_param('time_shift')
        dream_data.plot_fit_param('q')
        param_ranges = {'time_shift': [2, 8], 'time_max': [0, 10], 'amplitude': [0, 5000]}
        dream_data.plot_fit_param('amplitude', param_ranges)
        param_ranges = {'amplitude': [100, 5000]}
        dream_data.plot_fit_param('time_max', param_ranges)
        param_ranges = {'time_shift': [2, 8], 'q': [0.55, 0.7], 'amplitude': [500, 1000], 'time_max': [5, 10]}
        dream_data.plot_fits(param_ranges)
        param_ranges = {'time_shift': [2, 8], 'q': [0.55, 0.7], 'amplitude': [10, 50], 'time_max': [5, 10]}
        dream_data.plot_fits(param_ranges)
        param_ranges = {'time_shift': [2, 8], 'amplitude': [100, 1000], 'time_max': [1, 15]}
        dream_data.plot_fits(param_ranges)
        # param_ranges = {'time_max': [40, 60]}
        # dream_data.plot_fits(param_ranges, n_max=10)
        plt.show()
        print(dream_data.data_amps)

        print('\n')

    print('donzo')


def get_residuals(det, ray_data, plot=False):
    x_res_i, y_res_i = [], []
    for sub_det_i, sub_det in enumerate(det.sub_detectors):
        event_nums_i, centroids_i = sub_det.get_event_centroids()
        event_nums_i = list(np.array(event_nums_i) - 0)
        x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(det.center[2], event_nums_i)
        centroids_i_matched = []
        for event_num in event_nums_i:
            if event_num in event_num_rays:
                centroids_i_matched.append(centroids_i[list(event_nums_i).index(event_num)])
        centroids_i_matched = np.array(centroids_i_matched)
        x_res_i.extend(centroids_i_matched[:, 0] - x_rays)
        y_res_i.extend(centroids_i_matched[:, 1] - y_rays)
        if plot:
            plot_xy_residuals_2d(x_rays, y_rays, centroids_i_matched[:, 0], centroids_i_matched[:, 1])
    x_res_i_mean, y_res_i_mean = np.mean(x_res_i), np.mean(y_res_i)
    x_res_i_std, y_res_i_std = np.std(x_res_i), np.std(y_res_i)

    return x_res_i_mean, y_res_i_mean, x_res_i_std, y_res_i_std


def plot_xy_residuals_2d(xs_ref, ys_ref, xs_meas, ys_meas):
    """
    Plot residuals of measured x and y positions vs reference x and y positions.
    :param xs_ref: Reference x positions.
    :param ys_ref: Reference y positions.
    :param xs_meas: Measured x positions.
    :param ys_meas: Measured y positions.
    :return:
    """
    fig, ax = plt.subplots()
    ax.scatter(xs_ref, ys_ref, color='blue', label='Reference', marker='.', alpha=0.5)
    ax.scatter(xs_meas, ys_meas, color='green', label='Measured', marker='.', alpha=0.5)
    for x_ref, y_ref, x_meas, y_meas in zip(xs_ref, ys_ref, xs_meas, ys_meas):
        ax.plot([x_ref, x_meas], [y_ref, y_meas], color='red', linewidth=0.5)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.legend()
    fig.tight_layout()

    x_res = xs_meas - xs_ref
    y_res = ys_meas - ys_ref

    fig, ax = plt.subplots()
    ax.scatter(xs_ref, x_res, color='blue', label='X Residuals')
    ax.scatter(ys_ref, y_res, color='red', label='Y Residuals')
    ax.set_xlabel('Reference Position (mm)')
    ax.set_ylabel('Residual (mm)')
    ax.legend()
    ax.grid()
    fig.tight_layout()


if __name__ == '__main__':
    main()
