#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 04 8:17 PM 2024
Created in PyCharm
Created as saclay_micromegas/detector_characterization.py

@author: Dylan Neff, Dylan
"""

import os

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

from det_classes_test import plot_ray_hits_2d


def main():
    base_dir = 'F:/Saclay/cosmic_data/'
    det_type_info_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    out_dir = 'F:/Saclay/Analysis/Cosmic Bench/9-24-24/'
    # base_dir = '/local/home/dn277127/Bureau/cosmic_data/'
    # det_type_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    # out_dir = '/local/home/dn277127/Bureau/cosmic_data/Analysis/10-16-24/'
    # run_name = 'sg1_stats_7-26-24'
    # run_name = 'urw_inter_sp1_test_10-14-24'
    # run_name = 'urw_inter_sp1_test2_10-16-24'
    run_name = 'urw_stats_10-31-24'
    run_dir = f'{base_dir}{run_name}/'
    # sub_run_name = 'max_hv_long_1'
    sub_run_name = 'long_run'
    # sub_run_name = 'test_1'

    # det_single = 'asacusa_strip_1'
    # det_single = 'asacusa_strip_2'
    # det_single = 'strip_grid_1'
    # det_single = 'inter_grid_1'
    # det_single = 'urw_inter'
    det_single = 'urw_strip'
    # det_single = 'strip_plein_1'
    # det_single = None

    # file_nums = 'all'
    # file_nums = list(range(0, 645))
    file_nums = list(range(0, 10))

    chunk_size = 100  # Number of files to process at once

    realign_dream = True  # If False, read alignment from file, if True, realign Dream detector

    run_json_path = f'{run_dir}run_config.json'
    data_dir = f'{run_dir}{sub_run_name}/filtered_root/'
    ped_dir = f'{run_dir}{sub_run_name}/decoded_root/'
    banco_data_dir = f'{run_dir}{sub_run_name}/banco_data/'
    banco_noise_dir = f'{base_dir}/banco_noise/'
    m3_dir = f'{run_dir}{sub_run_name}/m3_tracking_root/'
    # out_dir = f'{out_dir}{det_single}/'
    # try:
    #     os.mkdir(out_dir)
    # except FileExistsError:
    #     pass

    z_align_range = [5, 5]  # mm range to search for optimal z position

    det_config_loader = DetectorConfigLoader(run_json_path, det_type_info_dir)

    print(f'Getting ray data...')
    ray_data = M3RefTracking(m3_dir, single_track=True, file_nums=file_nums)

    for detector_name in det_config_loader.included_detectors:
        if det_single is not None and detector_name != det_single:
            continue

        print(detector_name)
        det_config = det_config_loader.get_det_config(detector_name, sub_run_name=sub_run_name)
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
            det.load_dream_data(data_dir, ped_dir, 10, file_nums, chunk_size, save_waveforms=True)
            print(f'Hits shape: {det.dream_data.hits.shape}')
            # det.dream_data.plot_noise_metric()
            det.dream_data.plot_fine_timestamp_hist()
            det.dream_data.plot_event_amplitudes()
            det.dream_data.plot_fits({'time_max': [-1, 40]}, n_max=5)
            # plt.show()
            # det.dream_data.plot_fit_param('sigma', {'sigma': [0, 10], 'time_max': [-1, 40]})
            # det.dream_data.plot_fit_param('time_max', {'sigma': [0, 10], 'time_max': [-1, 40]})
            det.dream_data.plot_fit_param('time_max', {'time_max': [-1, 40]})
            # plt.show()
            det.dream_data.plot_event_time_maxes(max_channel=True, channels=np.arange(int(256 / 2), 256), min_amp=None)
            det.dream_data.correct_for_fine_timestamps()
            det.dream_data.plot_event_time_maxes(max_channel=True, channels=np.arange(int(256 / 2), 256), min_amp=None)

            # Iterate over fine timestamp correction values and plot the sigma
            fine_time_stamp_constants = np.linspace(0, 0.5, 50)
            sigmas = []
            det.dream_data.uncorrect_for_fine_timestamps()
            for fine_time_stamp_constant in fine_time_stamp_constants:
                det.dream_data.fine_timestamp_constant = fine_time_stamp_constant
                det.dream_data.correct_for_fine_timestamps()
                sigma = det.dream_data.plot_event_time_maxes(max_channel=True, channels=np.arange(int(256 / 2), 256), min_amp=None)
                sigmas.append(sigma)
                det.dream_data.uncorrect_for_fine_timestamps()

            fig, ax = plt.subplots()
            ax.plot(fine_time_stamp_constants, sigmas)
            ax.set_xlabel('Fine Timestamp Constant')
            ax.set_ylabel('Sigma')
            ax.set_title('Sigma vs Fine Timestamp Constant')
            plt.show()

            # det.dream_data.plot_fits({'sigma': [0, 10], 'time_max': [-1, 40]}, n_max=20)
            plt.show()

            # hit_thresh = [1, 75]
            # hit_thresh = [75, 1000]
            # Get indices of events with hits above threshold
            # hits = np.sum(det.dream_data.hits, axis=1)
            # hit_indices = np.where((hits > hit_thresh[0]) & (hits < hit_thresh[1]))[0]
            # det.dream_data.filter_data(hit_indices)
            # det.dream_data.plot_noise_metric()

            in_range = False  # If True, filter hits in range, if False, filter hits out of range
            # x_ray_bounds, y_ray_bounds = [-30, 30], [-30, 30]
            x_ray_bounds, y_ray_bounds = [-62, 42], [-40, 45]
            ray_events = filter_ray_xy(ray_data, det.center[2], x_ray_bounds, y_ray_bounds)
            if in_range:
                dream_data_indices = np.where(np.isin(det.dream_data.event_nums, ray_events))[0]
            else:
                dream_data_indices = np.where(~np.isin(det.dream_data.event_nums, ray_events))[0]

            # max_amp_thresh = [3500, 4000]
            max_amp_thresh = [0, 4000]
            max_amps = np.max(det.dream_data.data_amps, axis=1)
            max_amp_indices = np.where((max_amps > max_amp_thresh[0]) & (max_amps < max_amp_thresh[1]))[0]
            dream_data_indices = np.intersect1d(dream_data_indices, max_amp_indices)

            det.dream_data.filter_data(dream_data_indices)

            det.dream_data.plot_noise_metric()

            det.dream_data.plot_pedestals()
            # det.dream_data.plot_common_noise(0)
            det.dream_data.plot_hits_vs_strip(print_dead_strips=True)
            det.dream_data.plot_amplitudes_vs_strip()
            # plt.show()
            # param_ranges = {'amplitude': [10, 5000]}
            # det.dream_data.plot_fit_param('amplitude', param_ranges)
            # det.dream_data.plot_fit_param('time_max')
            # # plt.show()
            det.make_sub_detectors()
            event_nums = det.plot_xy_amp_sum_vs_event_num(True, 500, False, 15)

            # det.plot_amplitude_sum_vs_event_num()
            # det.plot_num_hit_xy_hist()
            # print(f'Det data: {len(det.dream_data.data_amps)}')
            # print(f'Ray data: {len(ray_data.ray_data)}')
            # det.plot_centroids_2d()
            # plot_ray_hits_2d(det, ray_data)
            if realign_dream:
                det.add_rotation(90, 'z')
            # det.plot_centroids_2d()
            # det.plot_centroids_2d_heatmap()
            # det.plot_centroids_2d_scatter_heat()
            plot_ray_hits_2d(det, ray_data)
            det.plot_hits_1d()

            for event_num in range(len(det.dream_data.hits)):
                det.plot_event_1d(event_num)
                det.plot_event_2d(event_num)
                det.dream_data.plot_waveforms(event_num)
                plt.show()

            plt.show()

            z_orig = det.center[2]
            x_bnds = det.center[0] - det.size[0] / 2, det.center[0] + det.size[0] / 2
            y_bnds = det.center[1] - det.size[1] / 2, det.center[1] + det.size[1] / 2
            ray_traversing_triggers = ray_data.get_traversing_triggers(z_orig, x_bnds, y_bnds, expansion_factor=0.1)

            alignment_file = f'{alignment_dir}{det.name}_alignment.txt'
            if realign_dream:
                align_dream(det, ray_data, z_align_range)
                det.write_det_alignment_to_file(alignment_file)
            else:
                det.read_det_alignment_from_file(alignment_file)
            plot_ray_hits_2d(det, ray_data)

            # sub_centroids, sub_triggers = det.get_sub_centroids_coords()
            # weird_trigger = None
            # for sub_det, sub_centroids_i, sub_triggers_i in zip(det.sub_detectors, sub_centroids, sub_triggers):
            #     print('Here')
            #     for centroid, trigger in zip(sub_centroids_i, sub_triggers_i):
            #         if centroid[1] < -50:
            #             print(f'Weird centroid: {centroid}, trigger: {trigger}')
            #             weird_trigger = int(trigger)
            #     sub_det.plot_cluster_sizes()
            #     break
            #
            # # Get event number index corresponding to weird trigger from det.dream_data
            # weird_trigger_index = np.where(det.dream_data.event_nums == weird_trigger)[0][0]
            # print(f'Weird trigger index: {weird_trigger_index}')
            # det.plot_event_1d(weird_trigger_index)
            # det.plot_event_2d(weird_trigger_index)
            #
            # for sub_det in det.sub_detectors:
            #     triggers, centroids = sub_det.get_event_centroids()
            #     if triggers.shape[0] != centroids.shape[0]:
            #         print(f'Error: Triggers and centroids have different shapes: {triggers.shape}, {centroids.shape}')
            #     if len(centroids) == 0:
            #         continue
            #     zs = np.full((len(centroids), 1), 0)  # Add z coordinate to centroids
            #     centroids = np.hstack((centroids, zs))  # Combine x, y, z
            #     centroids_rot = det.convert_coords_to_global(centroids)
            #     # Get centroid and rotated centroid corresponding to weird trigger
            #     weird_centroid = centroids[np.where(triggers == weird_trigger)[0][0]]
            #     weird_centroid_rot = centroids_rot[np.where(triggers == weird_trigger)[0][0]]
            #     print(f'Weird centroid: {weird_centroid}, rotated: {weird_centroid_rot}')
            #     break

            # get_residuals(det, ray_data, plot=True, in_det=True, tolerance=1.0)
            get_banco_telescope_residuals(det, banco_telescope, banco_triggers, plot=True)
            # plt.show()

            # x_subs_mean, y_subs_mean, x_subs_std, y_subs_std = get_residuals(det, ray_data, plot=False, sub_reses=True)
            # pitches_x, pitches_y, inter_pitches_x, inter_pitches_y  = [], [], [], []
            # resolutions, res_xs, res_ys = [], [], []
            # for i, (x_mean, y_mean, x_std, y_std) in enumerate(zip(x_subs_mean, y_subs_mean, x_subs_std, y_subs_std)):
            #     x_mean, y_mean = int(x_mean * 1000), int(y_mean * 1000)
            #     x_std, y_std = int(x_std * 1000), int(y_std * 1000)
            #     pitch_x = det.sub_detectors[i].x_pitch
            #     pitch_y = det.sub_detectors[i].y_pitch
            #     inter_pitch_x = det.sub_detectors[i].x_interpitch
            #     inter_pitch_y = det.sub_detectors[i].y_interpitch
            #     print(f'Sub-Detector {i} '
            #           f'(pitch_x: {pitch_x}, pitch_y: {pitch_y}, inter_x: {inter_pitch_x}, inter_y: {inter_pitch_y}) '
            #           f'x_mean: {x_mean}μm, y_mean: {y_mean}μm, x_std: {x_std}μm, y_std: {y_std}μm')
            #     pitches_x.append(pitch_x)
            #     pitches_y.append(pitch_y)
            #     inter_pitches_x.append(inter_pitch_x)
            #     inter_pitches_y.append(inter_pitch_y)
            #     res_xs.append(x_std)
            #     res_ys.append(y_std)
            #     resolutions.append(np.sqrt(x_std ** 2 + y_std ** 2))
            # # Sort by pitch_x
            # pitches_x, pitches_y, resolutions, res_xs, res_ys = zip(*sorted(zip(pitches_x, pitches_y, resolutions, res_xs, res_ys)))
            # print(pitches_x, pitches_y, resolutions, res_xs, res_ys)
            # # Write to file in out_dir
            # with open(f'{out_dir}{det.name}_res_vs_pitch.txt', 'w') as file:
            #     file.write(f'Pitch (mm)\tResolution (um)\tX Res (um)\tY Res (um)\n')
            #     for pitch_x, pitch_y, res, res_x, res_y in zip(pitches_x, pitches_y, resolutions, res_xs, res_ys):
            #         file.write(f'{pitch_x}\t{pitch_y}\t{res}\t{res_x}\t{res_y}\n')
            # print(det.name)
            # fig, ax = plt.subplots()
            # ax.plot(pitches_x, resolutions, marker='o', zorder=10)
            # ax.set_xlabel('Pitch (mm)')
            # ax.set_ylabel('Resolution (μm)')
            # ax.grid()
            # fig.tight_layout()

            all_figures = [plt.figure(num) for num in plt.get_fignums()]
            for fig_i, fig in enumerate(all_figures):
                fig_name = fig.axes[0].get_title() + f'_{fig_i}'
                fig_name = fig_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
                fig_name = fig_name.replace(':', '').replace('.', '')
                fig.savefig(f'{out_dir}{fig_name}.png')

            plt.show()
            input('Finished, press Enter to continue...')

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


def filter_ray_xy(ray_data, det_z, x_bnds, y_bnds):
    """
    Get event_nums of events in ray_data that are within x_bnds and y_bnds.
    :param ray_data:
    :param det_z:
    :param x_bnds:
    :param y_bnds:
    :return:
    """
    x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(det_z)

    x_indices = np.where((x_rays > x_bnds[0]) & (x_rays < x_bnds[1]))[0]
    y_indices = np.where((y_rays > y_bnds[0]) & (y_rays < y_bnds[1]))[0]
    xy_indices = np.intersect1d(x_indices, y_indices)

    return event_num_rays[xy_indices]


if __name__ == '__main__':
    main()

