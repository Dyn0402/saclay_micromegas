#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 09 4:03 PM 2024
Created in PyCharm
Created as saclay_micromegas/det_classes_test.py

@author: Dylan Neff, Dylan
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
from scipy.stats import skewnorm

from Detector_Classes.M3RefTracking import M3RefTracking
from Detector_Classes.DetectorConfigLoader import DetectorConfigLoader
from Detector_Classes.Detector import Detector
from Detector_Classes.DreamDetector import DreamDetector
from Detector_Classes.DreamData import DreamData
from Detector_Classes.BancoTelescope import BancoTelescope
from Measure import Measure


def main():
    # base_dir = 'F:/Saclay/cosmic_data/'
    # det_type_info_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    # out_dir = 'F:/Saclay/Analysis/Cosmic Bench/10-2-24/'
    base_dir = '/local/home/dn277127/Bureau/cosmic_data/'
    det_type_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    out_dir = '/local/home/dn277127/Bureau/cosmic_data/Analysis/12-3-24/'
    # run_name = 'new_strip_check_7-12-24'
    # run_name = 'ig1_test1'
    run_name = 'banco_flipped_7-8-24'
    # run_name = 'ig1_sg1_stats4'
    # run_name = 'sg1_stats_7-26-24'
    # run_name = 'urw_stats_10-31-24'
    run_dir = f'{base_dir}{run_name}/'
    # sub_run_name = 'hv1'
    # sub_run_name = 'new_detector_short'
    # sub_run_name = 'drift_600_resist_460'
    # sub_run_name = 'quick_test'
    sub_run_name = 'max_hv_long'
    # sub_run_name = 'max_hv_long_1'
    # sub_run_name = 'long_run'

    # det_single = 'asacusa_strip_1'
    # det_single = 'asacusa_strip_2'
    # det_single = 'strip_grid_1'
    # det_single = 'inter_grid_1'
    # det_single = 'urw_inter'
    det_single = 'urw_strip'
    # det_single = None

    # file_nums = 'all'
    file_nums = list(range(0, 25))
    # file_nums = list(range(0, 645))
    # file_nums = list(range(0, 100))
    # file_nums = list(range(100, 200))
    # file_nums = list(range(100, 500))
    # file_nums = list(range(100, 110))

    # chunk_size = 100  # Number of files to process at once
    chunk_size = 7  # Number of files to process at once

    read_good_banco_triggers = False  # If True, read good banco triggers from file, if False, get them from BancoTelescope
    realign_banco = True  # If False, read alignment from file, if True, realign Banco telescope
    realign_dream = True  # If False, read alignment from file, if True, realign Dream detector
    banco_filtered = False  # If True, use filtered data, if False, use full data root file

    banco_event_ns = [3]

    run_json_path = f'{run_dir}run_config.json'
    data_dir = f'{run_dir}{sub_run_name}/filtered_root/'
    ped_dir = f'{run_dir}{sub_run_name}/decoded_root/'
    banco_data_dir = f'{run_dir}{sub_run_name}/banco_data/'
    banco_noise_dir = f'{base_dir}banco_noise/'
    m3_dir = f'{run_dir}{sub_run_name}/m3_tracking_root/'

    alignment_dir = f'{run_dir}alignments/'
    select_triggers_dir = f'{run_dir}select_triggers/'
    out_dir = f'{out_dir}{det_single}/'
    try:
        os.mkdir(alignment_dir)
    except FileExistsError:
        pass
    try:
        os.mkdir(select_triggers_dir)
    except FileExistsError:
        pass
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    z_align_range = [5, 5]  # mm range to search for optimal z position

    det_config_loader = DetectorConfigLoader(run_json_path, det_type_info_dir)

    print(f'Getting ray data...')
    ray_data = M3RefTracking(m3_dir, single_track=True, file_nums=file_nums)

    # Load banco
    banco_telescope = BancoTelescope(det_config_loader, sub_run_name, banco_data_dir, banco_noise_dir)
    if not realign_banco:
        banco_telescope.read_ladder_alignments_from_file(alignment_dir)
    banco_triggers = None
    if read_good_banco_triggers:
        banco_triggers = banco_telescope.read_good_n_ladder_event_nums_from_file(f'{select_triggers_dir}good_n_ladder_event_nums.csv', banco_event_ns)
    banco_telescope.read_data(ray_data, filtered=banco_filtered, trigger_list=banco_triggers)
    # triggers = banco_telescope.get_all_banco_traversing_triggers(ray_data)
    # with open(f'{banco_data_dir}banco_triggers.txt', 'w') as file:
    #     for trigger in triggers:
    #         file.write(f'{trigger}\n')
    # input('Press Enter to continue...')
    # for ladder in banco_telescope.ladders:
    #     ladder.plot_cluster_centroids()
    # plt.show()
    if realign_banco:
        banco_telescope.align_ladders(ray_data)
        banco_telescope.write_ladder_alignments_to_file(alignment_dir)
    # plt.show()
    if not read_good_banco_triggers:
        banco_telescope.write_good_n_ladder_event_nums_to_file(f'{select_triggers_dir}good_n_ladder_event_nums.csv', [3, 4])
    # input('Press Enter to continue...')

    # Get list of triggers with rays traversing Banco telescope and one dream detector


    # for trigger in banco_telescope.four_ladder_triggers:
    #     print(f'Trigger: {trigger}')
    #     xs, ys, zs = [], [], []
    #     for ladder in banco_telescope.ladders:
    #         x, y, z = ladder.get_cluster_centroid_by_trigger(trigger)
    #         xs.append(x)
    #         ys.append(y)
    #         zs.append(z)
    #
    #     # Fit xs and yz as a function of z to a line, then plot the points and the lines in two 2D plots and 1 3D plot
    #     xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    #     popt_x, pcov_x = cf(linear, zs, xs)
    #     popt_y, pcov_y = cf(linear, zs, ys)
    #
    #     fig, ax = plt.subplots()
    #     ax.scatter(zs, xs, color='blue', label='X')
    #     ax.plot(zs, linear(zs, *popt_x), color='red', label='Fit')
    #     ax.set_xlabel('Z (mm)')
    #     ax.set_ylabel('X (mm)')
    #     ax.legend()
    #     ax.grid()
    #     fig.tight_layout()
    #
    #     fig, ax = plt.subplots()
    #     ax.scatter(zs, ys, color='blue', label='Y')
    #     ax.plot(zs, linear(zs, *popt_y), color='red', label='Fit')
    #     ax.set_xlabel('Z (mm)')
    #     ax.set_ylabel('Y (mm)')
    #     ax.legend()
    #     ax.grid()
    #     fig.tight_layout()
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(xs, ys, zs, color='blue')
    #     ax.plot(linear(zs, *popt_x), linear(zs, *popt_y), zs, color='red')
    #     ax.set_xlabel('X (mm)')
    #     ax.set_ylabel('Y (mm)')
    #     ax.set_zlabel('Z (mm)')
    #     fig.tight_layout()
    #
    #     plt.show()

    # plt.show()

    # input('Press Enter to continue...')

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
            if realign_dream:
                banco_dream_triggers = None
            else:
                banco_dream_triggers = np.array(banco_triggers) + 1
            det.load_dream_data(data_dir, ped_dir, 10, file_nums, chunk_size, banco_dream_triggers)
            print(f'Hits shape: {det.dream_data.hits.shape}')
            # det.dream_data.plot_noise_metric()
            # det.dream_data.plot_pedestals()
            det.dream_data.plot_hits_vs_strip(print_dead_strips=True)
            det.dream_data.plot_amplitudes_vs_strip()
            # # plt.show()
            # param_ranges = {'amplitude': [10, 5000]}
            # det.dream_data.plot_fit_param('amplitude', param_ranges)
            # det.dream_data.plot_fit_param('time_max')
            # # plt.show()
            det.make_sub_detectors()
            # event_nums = det.plot_xy_amp_sum_vs_event_num(True, 500, False, 15)
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
            # plt.show()

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
            plt.show()

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


def align_dream(det, ray_data, z_range=None, z_rot_range=None):
    x_res_i_mean, y_res_i_mean, x_res_i_std, y_res_i_std = get_residuals(det, ray_data)
    det.set_center(x=det.center[0] - x_res_i_mean, y=det.center[1] - y_res_i_mean)

    if z_range is None:
        z_range = [5, 5]
    zs = np.linspace(det.center[2] - z_range[0], det.center[2] + z_range[1], 30)
    # zs = [det.center[2]]
    x_residuals, y_residuals = [], []
    z_og = det.center[2]
    for z in zs:
        print(f'z: {z}')
        det.set_center(z=z)
        x_res_i_mean, y_res_i_mean, x_res_i_std, y_res_i_std = get_residuals(det, ray_data, plot=False)
        x_residuals.append(x_res_i_std)
        y_residuals.append(y_res_i_std)
    r_res = np.sqrt(np.array(x_residuals) ** 2 + np.array(y_residuals) ** 2)
    min_res = np.min(r_res)
    z_min = zs[np.argmin(r_res)]
    fig, ax = plt.subplots()
    ax.plot(zs, x_residuals, label='X Residuals')
    ax.plot(zs, y_residuals, label='Y Residuals')
    ax.scatter(z_min, min_res, color='red', marker='x', label='Min xy Residual')
    ax.axvline(z_og, color='green', linestyle='--', alpha=0.5, label='Original')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Residual (mm)')
    ax.legend()
    ax.grid()
    fig.tight_layout()

    det.set_center(z=z_min)

    if z_rot_range is None:
        z_rot_range = [-1, 1]
    z_rots = np.linspace(z_rot_range[0], z_rot_range[1], 30)
    x_residuals, y_residuals = [], []
    det.add_rotation(0, 'z')
    for z_rot in z_rots:
        print(f'z_rot: {z_rot}')
        det.replace_last_rotation(z_rot, 'z')
        x_res_i_mean, y_res_i_mean, x_res_i_std, y_res_i_std = get_residuals(det, ray_data)
        x_residuals.append(x_res_i_std)
        y_residuals.append(y_res_i_std)
    r_res = np.sqrt(np.array(x_residuals) ** 2 + np.array(y_residuals) ** 2)
    min_rot_res = np.min(r_res)
    z_rot_min = z_rots[np.argmin(r_res)]
    fig, ax = plt.subplots()
    ax.plot(z_rots, x_residuals, label='X Residuals')
    ax.plot(z_rots, y_residuals, label='Y Residuals')
    ax.scatter(z_rot_min, min_rot_res, color='red', marker='x', label='Min xy Residual')
    ax.axvline(0, color='green', linestyle='--', alpha=0.5, label='Original')
    ax.set_xlabel('z_rot (deg)')
    ax.set_ylabel('Residual (mm)')
    ax.grid()
    ax.legend()
    fig.tight_layout()

    det.replace_last_rotation(z_rot_min, 'z')

    x_res_i_mean, y_res_i_mean, x_res_i_std, y_res_i_std = get_residuals(det, ray_data)
    det.set_center(x=det.center[0] - x_res_i_mean, y=det.center[1] - y_res_i_mean)


def get_residuals(det, ray_data, sub_reses=False, plot=False, in_det=False, tolerance=0.0):
    x_res, y_res, = [], []
    x_subs_mean, x_subs_std, y_subs_mean, y_subs_std = [], [], [], []
    subs_centroids, subs_triggers = det.get_sub_centroids_coords()
    for sub_centroids, sub_triggers, sub_det in zip(subs_centroids, subs_triggers, det.sub_detectors):
        x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(det.center[2], list(sub_triggers))
        if in_det:
            x_rays, y_rays, event_num_rays = get_rays_in_sub_det(det, sub_det, x_rays, y_rays, event_num_rays, tolerance)

        if event_num_rays is None or len(event_num_rays) == 0:
            continue

        # Sort sub_triggers and sub_centroids together by sub_trigger
        sub_triggers, sub_centroids = zip(*sorted(zip(sub_triggers, sub_centroids)))
        sub_centroids, sub_triggers = np.array(sub_centroids), np.array(sub_triggers)

        # Sort x_rays, y_rays, and event_num_rays by event_num_rays
        event_num_rays, x_rays, y_rays = zip(*sorted(zip(event_num_rays, x_rays, y_rays)))
        event_num_rays, x_rays, y_rays = np.array(event_num_rays), np.array(x_rays), np.array(y_rays)

        # Find indices of sub_triggers in event_num_rays
        matched_indices = np.in1d(np.array(sub_triggers), np.array(event_num_rays)).nonzero()[0]

        if len(matched_indices) == 0:
            x_res.extend(None)
            y_res.extend(None)
            x_subs_mean.append(None)
            y_subs_mean.append(None)
            x_subs_std.append(None)
            y_subs_std.append(None)
            continue

        centroids_i_matched = sub_centroids[matched_indices]

        x_res_i = centroids_i_matched[:, 0] - x_rays
        y_res_i = centroids_i_matched[:, 1] - y_rays

        x_res.extend(x_res_i)
        y_res.extend(y_res_i)

        x_popt_i, y_popt_i = fit_residuals(x_res_i, y_res_i)
        x_subs_mean.append(x_popt_i[1])
        y_subs_mean.append(y_popt_i[1])
        x_subs_std.append(x_popt_i[2])
        y_subs_std.append(y_popt_i[2])

        if plot:
            title_post = sub_det.description
            plot_xy_residuals_2d(x_rays, y_rays, centroids_i_matched[:, 0], centroids_i_matched[:, 1], title_post)

    if sub_reses:
        return x_subs_mean, y_subs_mean, x_subs_std, y_subs_std

    x_popt, y_popt = fit_residuals(x_res, y_res)
    x_res_i_mean, y_res_i_mean = x_popt[1], y_popt[1]
    x_res_i_std, y_res_i_std = x_popt[2], y_popt[2]
    return x_res_i_mean, y_res_i_mean, x_res_i_std, y_res_i_std


def get_residuals_align(det, ray_data, triggers, sub_reses=False, plot=False):
    x_res, y_res, = [], []
    x_subs_mean, x_subs_std, y_subs_mean, y_subs_std = [], [], [], []
    subs_centroids, subs_triggers = det.get_sub_centroids_coords()
    for sub_centroids, sub_triggers in zip(subs_centroids, subs_triggers):
        x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(det.center[2], list(sub_triggers))

        # Find indices of sub_triggers in event_num_rays
        matched_indices = np.in1d(np.array(sub_triggers), np.array(event_num_rays)).nonzero()[0]

        centroids_i_matched = sub_centroids[matched_indices]

        x_res_i = centroids_i_matched[:, 0] - x_rays
        y_res_i = centroids_i_matched[:, 1] - y_rays

        x_res.extend(x_res_i)
        y_res.extend(y_res_i)

        x_popt_i, y_popt_i = fit_residuals(x_res_i, y_res_i)
        x_subs_mean.append(x_popt_i[1])
        y_subs_mean.append(y_popt_i[1])
        x_subs_std.append(x_popt_i[2])
        y_subs_std.append(y_popt_i[2])

        if plot:
            plot_xy_residuals_2d(x_rays, y_rays, centroids_i_matched[:, 0], centroids_i_matched[:, 1])

    if sub_reses:
        return x_subs_mean, y_subs_mean, x_subs_std, y_subs_std

    x_popt, y_popt = fit_residuals(x_res, y_res)
    x_res_i_mean, y_res_i_mean = x_popt[1], y_popt[1]
    x_res_i_std, y_res_i_std = x_popt[2], y_popt[2]
    return x_res_i_mean, y_res_i_mean, x_res_i_std, y_res_i_std


# def get_residuals_clust_sizes(det, ray_data, sub_reses=False, plot=False, in_det=False, tolerance=0.0):
#     x_res, y_res, = [], []
#     x_subs_mean, x_subs_std, y_subs_mean, y_subs_std = [], [], [], []
#     subs_centroids, subs_triggers = det.get_sub_centroids_coords()
#     for sub_centroids, sub_triggers, sub_det in zip(subs_centroids, subs_triggers, det.sub_detectors):
#         x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(det.center[2], list(sub_triggers))
#         if in_det:
#             x_rays, y_rays, event_num_rays = get_rays_in_sub_det(det, sub_det, x_rays, y_rays, event_num_rays, tolerance)
#
#         # Find indices of sub_triggers in event_num_rays
#         matched_indices = np.in1d(np.array(sub_triggers), np.array(event_num_rays)).nonzero()[0]
#
#         if len(matched_indices) == 0:
#             x_res.extend(None)
#             y_res.extend(None)
#             x_subs_mean.append(None)
#             y_subs_mean.append(None)
#             x_subs_std.append(None)
#             y_subs_std.append(None)
#             continue
#
#         centroids_i_matched = sub_centroids[matched_indices]
#
#         x_res_i = centroids_i_matched[:, 0] - x_rays
#         y_res_i = centroids_i_matched[:, 1] - y_rays
#
#         x_res.extend(x_res_i)
#         y_res.extend(y_res_i)
#
#         x_popt_i, y_popt_i = fit_residuals(x_res_i, y_res_i)
#         x_subs_mean.append(x_popt_i[1])
#         y_subs_mean.append(y_popt_i[1])
#         x_subs_std.append(x_popt_i[2])
#         y_subs_std.append(y_popt_i[2])
#
#         if plot:
#             title_post = sub_det.description
#             plot_xy_residuals_2d(x_rays, y_rays, centroids_i_matched[:, 0], centroids_i_matched[:, 1], title_post)
#
#     if sub_reses:
#         return x_subs_mean, y_subs_mean, x_subs_std, y_subs_std
#
#     x_popt, y_popt = fit_residuals(x_res, y_res)
#     x_res_i_mean, y_res_i_mean = x_popt[1], y_popt[1]
#     x_res_i_std, y_res_i_std = x_popt[2], y_popt[2]
#     return x_res_i_mean, y_res_i_mean, x_res_i_std, y_res_i_std


def get_banco_telescope_residuals(det, banco_telescope, banco_triggers=None, plot=False):
    """
    Get residuals between Banco telescope and detector.
    :param det:
    :param banco_telescope:
    :param banco_triggers:
    :param plot:
    :return:
    """
    if banco_triggers is None:
        banco_triggers = np.array(banco_telescope.four_ladder_triggers)  # banco starts at 0, dream starts at 1

    subs_centroids, subs_triggers = det.get_sub_centroids_coords()
    for sub_centroids, sub_triggers, sub_det in zip(subs_centroids, subs_triggers, det.sub_detectors):
        x_banco_rays_all, y_banco_rays_all = banco_telescope.get_xy_track_positions(det.center[2], banco_triggers)
        x_banco_rays, y_banco_rays, triggers_banco = get_rays_in_sub_det(det, sub_det, x_banco_rays_all,
                                                                         y_banco_rays_all, banco_triggers + 1,
                                                                         tolerance=0.0)
        fig, ax = plt.subplots()
        ax.scatter(x_banco_rays_all, y_banco_rays_all, color='blue', label='Banco Rays', marker='.', alpha=0.5)
        ax.scatter(sub_centroids[:, 0], sub_centroids[:, 1], color='red', label='Detector Centroids', marker='.', alpha=0.5)
        ax.scatter(x_banco_rays, y_banco_rays, color='green', label='Banco Rays in Detector', marker='.', alpha=0.5)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.legend()
        ax.set_title(f'2D Hit Centroids and Rays {sub_det.description}')
        fig.tight_layout()
        print(f'Centroids: {sub_centroids.shape}, Banco Rays: {len(x_banco_rays)}, Banco Triggers: {len(triggers_banco)}, Banco All: {len(x_banco_rays_all)}')

        matched_indices = np.in1d(np.array(sub_triggers), np.array(triggers_banco))
        centroids_i_matched = sub_centroids[matched_indices]
        print(f'Matched indices: {matched_indices.shape}, {len(matched_indices)}, matched centroids: {centroids_i_matched.shape}')
        if len(triggers_banco) > 0:
            print(f'Percent matched: {len(centroids_i_matched) / len(triggers_banco) * 100:.2f}%')

        if len(centroids_i_matched) == 0:
            print(f'No matched indices for {sub_det.description}')
            continue

        fig, ax = plt.subplots()
        ax.scatter(triggers_banco, [1] * len(triggers_banco), color='blue', label='Banco Triggers', marker='.')
        ax.scatter(sub_triggers, [2] * len(sub_triggers), color='red', label='Detector Triggers', marker='.')
        ax.scatter(np.array(sub_triggers)[matched_indices], [3] * len(centroids_i_matched), color='green',
                   label='Matched Triggers', marker='.')
        ax.set_xlabel('Trigger Number')
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Banco', 'Detector', 'Matched'])
        ax.legend()
        ax.set_title(f'Trigger Numbers {sub_det.description}')
        fig.tight_layout()

        matched_triggers = np.array(sub_triggers)[matched_indices]
        matched_x_indices = np.in1d(np.array(sub_det.x_cluster_triggers), matched_triggers)
        x_clust_sizes = np.array(sub_det.x_largest_cluster_sizes)[matched_x_indices]
        matched_y_indices = np.in1d(np.array(sub_det.y_cluster_triggers), matched_triggers)
        y_clust_sizes = np.array(sub_det.y_largest_cluster_sizes)[matched_y_indices]
        # Plot histograms of x and y cluster sizes, which will be integers
        fig, ax = plt.subplots()
        ax.hist(x_clust_sizes, bins=np.arange(-0.5, 10.5, 1), color='blue', alpha=0.5, label='X Cluster Sizes')
        ax.hist(y_clust_sizes, bins=np.arange(-0.5, 10.5, 1), color='red', alpha=0.5, label='Y Cluster Sizes')
        ax.set_xlabel('Cluster Size')
        ax.set_ylabel('Counts')
        ax.legend()
        ax.set_title(f'Cluster Sizes {sub_det.description}')
        fig.tight_layout()

        matched_banco_indices = np.in1d(np.array(triggers_banco), np.array(sub_triggers))
        x_banco_rays, y_banco_rays = np.array(x_banco_rays)[matched_banco_indices], np.array(y_banco_rays)[matched_banco_indices]

        if plot:
            title_post = sub_det.description
            # try:
            plot_xy_residuals_2d_2(x_banco_rays, y_banco_rays, centroids_i_matched[:, 0], centroids_i_matched[:, 1], 20,
                                   title_post)
            # except:
            #     print(f'Error plotting {sub_det.description}')


def get_efficiency(det, ray_data, hit_dist=1000, plot=False, in_det=False, tolerance=0.0, grid_size=5):
    """
    Get efficiency of detector by comparing ray hits to detector centroids.
    :param det:
    :param ray_data:
    :param hit_dist: If a detector centroid is within hit_dist of a ray hit, it is considered a hit
    :param plot:
    :param in_det:
    :param tolerance:
    :param grid_size: um per grid cell
    :return:
    """
    x_rays_all, y_rays_all, event_num_rays_all = ray_data.get_xy_positions(det.center[2])
    print(f'Pre-filtered rays: {len(x_rays_all)}')
    if in_det:
        x_bnds = det.center[0] - det.size[0] / 2, det.center[0] + det.size[0] / 2
        y_bnds = det.center[1] - det.size[1] / 2, det.center[1] + det.size[1] / 2
        ray_traversing_triggers = ray_data.get_traversing_triggers(det.center[2], x_bnds, y_bnds, expansion_factor=0.1)
        trigger_indices = np.in1d(np.array(event_num_rays_all), np.array(ray_traversing_triggers)).nonzero()[0]
        event_num_rays_all = event_num_rays_all[trigger_indices]
        x_rays_all, y_rays_all = x_rays_all[trigger_indices], y_rays_all[trigger_indices]
    print(f'All rays: {len(x_rays_all)}')
    detector_hits = [False] * len(x_rays_all)
    detector_x_hits, detector_y_hits = [False] * len(x_rays_all), [False] * len(x_rays_all)
    subs_centroids, subs_triggers = det.get_sub_centroids_coords()
    for sub_centroids, sub_triggers, sub_det in zip(subs_centroids, subs_triggers, det.sub_detectors):
        x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(det.center[2], list(sub_triggers))
        if in_det:
            x_rays, y_rays, event_num_rays = get_rays_in_sub_det(det, sub_det, x_rays, y_rays, event_num_rays, tolerance)

        if event_num_rays is None or len(event_num_rays) == 0:
            continue

        # Sort sub_triggers and sub_centroids together by sub_trigger
        sub_triggers, sub_centroids = zip(*sorted(zip(sub_triggers, sub_centroids)))
        sub_centroids, sub_triggers = np.array(sub_centroids), np.array(sub_triggers)

        # Sort x_rays, y_rays, and event_num_rays by event_num_rays
        event_num_rays, x_rays, y_rays = zip(*sorted(zip(event_num_rays, x_rays, y_rays)))
        event_num_rays, x_rays, y_rays = np.array(event_num_rays), np.array(x_rays), np.array(y_rays)

        # Find indices of sub_triggers in event_num_rays
        matched_indices = np.in1d(np.array(sub_triggers), np.array(event_num_rays)).nonzero()[0]

        if len(matched_indices) == 0:
            continue

        centroids_i_matched = sub_centroids[matched_indices]

        x_res_i = centroids_i_matched[:, 0] - x_rays
        y_res_i = centroids_i_matched[:, 1] - y_rays
        r_res_i = np.sqrt(x_res_i ** 2 + y_res_i ** 2)

        # For each r_res_i, if r_res_i < hit_dist, it is a hit. Update detector_hits for this trigger
        for i, (r_res, x_res, y_res) in enumerate(zip(r_res_i, x_res_i, y_res_i)):
            if r_res < hit_dist:
                event_num_indices = np.where(event_num_rays_all == event_num_rays[i])[0]
                if len(event_num_indices) != 1:
                    print(f'Bad event num indices: {event_num_indices}')
                else:
                    detector_hits[event_num_indices[0]] = True
            if x_res < hit_dist:
                event_num_indices = np.where(event_num_rays_all == event_num_rays[i])[0]
                if len(event_num_indices) != 1:
                    print(f'Bad event num indices: {event_num_indices}')
                else:
                    detector_x_hits[event_num_indices[0]] = True
            if y_res < hit_dist:
                event_num_indices = np.where(event_num_rays_all == event_num_rays[i])[0]
                if len(event_num_indices) != 1:
                    print(f'Bad event num indices: {event_num_indices}')
                else:
                    detector_y_hits[event_num_indices[0]] = True

    if plot:
        # Plot blue circles for hits, red for misses. Split into two lists for hits and misses
        x_hits, y_hits, x_misses, y_misses = [], [], [], []
        x_1d_hits, y_1d_hits, x_1d_misses, y_1d_misses = [], [], [], []
        for x_ray, y_ray, hit, hit_x, hit_y in zip(x_rays_all, y_rays_all, detector_hits, detector_x_hits, detector_y_hits):
            if hit:
                x_hits.append(x_ray)
                y_hits.append(y_ray)
            else:
                x_misses.append(x_ray)
                y_misses.append(y_ray)
            if hit_x:
                x_1d_hits.append(x_ray)
            else:
                x_1d_misses.append(x_ray)
            if hit_y:
                y_1d_hits.append(y_ray)
            else:
                y_1d_misses.append(y_ray)

        # Get corners of active area
        corners = [[0, 0, 0], [0, det.active_size[1], 0], [det.active_size[0], det.active_size[1], 0],
                   [det.active_size[0], 0, 0], [0, 0, 0]]
        corners = det.convert_coords_to_global(corners)

        fig, ax = plt.subplots()
        ax.scatter(x_misses, y_misses, color='red', label='Misses', marker='.', alpha=0.2)
        ax.scatter(x_hits, y_hits, color='blue', label='Hits', marker='.', alpha=0.2)
        # Make a thin box around active area based on det.active_size
        ax.plot([corner[0] for corner in corners], [corner[1] for corner in corners], color='black', linewidth=2)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.legend()
        ax.set_title('Detector Efficiency')
        fig.tight_layout()

        # Combine hits and misses to find the grid boundaries
        x_all = np.array(x_hits + x_misses)
        y_all = np.array(y_hits + y_misses)

        x_1d_all = np.array(x_1d_hits + x_1d_misses)
        y_1d_all = np.array(y_1d_hits + y_1d_misses)

        # Define grid resolution
        grid_x_bins = np.arange(x_all.min(), x_all.max() + grid_size, grid_size)
        grid_y_bins = np.arange(y_all.min(), y_all.max() + grid_size, grid_size)

        # Create 2D histogram for hits and total (hits + misses)
        hist_hits, x_edges, y_edges = np.histogram2d(x_hits, y_hits, bins=[grid_x_bins, grid_y_bins],
                                                     range=[[x_all.min(), x_all.max()], [y_all.min(), y_all.max()]])
        hist_total, _, _ = np.histogram2d(x_all, y_all, bins=[grid_x_bins, grid_y_bins],
                                          range=[[x_all.min(), x_all.max()], [y_all.min(), y_all.max()]])

        # Calculate efficiency: fraction of hits to total in each grid cell
        efficiency = np.divide(hist_hits, hist_total, out=np.zeros_like(hist_hits, dtype=float), where=hist_total > 0)

        # Plot efficiency map
        fig, ax = plt.subplots()
        c = ax.pcolormesh(x_edges, y_edges, efficiency.T, cmap='jet', shading='auto')
        # Make a thin box around active area based on det.active_size
        ax.plot([corner[0] for corner in corners], [corner[1] for corner in corners], color='white', linewidth=0.5)
        fig.colorbar(c, ax=ax, label='Efficiency')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title('Efficiency Map')
        fig.tight_layout()

        # Plot total hits and misses for statistics
        fig, ax = plt.subplots()
        c = ax.pcolormesh(x_edges, y_edges, hist_total.T, cmap='jet', shading='auto')
        # Make a thin box around active area based on det.active_size
        ax.plot([corner[0] for corner in corners], [corner[1] for corner in corners], color='white', linewidth=2)
        fig.colorbar(c, ax=ax, label='Number of Rays')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title('Ray Statistics Map')
        fig.tight_layout()

        # Histogram 1D hits/misses with numpy then plot
        histx_1d_hits, x_bins = np.histogram(x_1d_hits, bins=grid_x_bins)
        histx_1d_total, _ = np.histogram(x_1d_all, bins=grid_x_bins)
        histy_1d_hits, y_bins = np.histogram(y_1d_hits, bins=grid_y_bins)
        histy_1d_total, _ = np.histogram(y_1d_all, bins=grid_y_bins)

        efficiencyx_1d = np.divide(histx_1d_hits, histx_1d_total, out=np.zeros_like(histx_1d_hits, dtype=float), where=histx_1d_total > 0)
        efficiencyy_1d = np.divide(histy_1d_hits, histy_1d_total, out=np.zeros_like(histy_1d_hits, dtype=float), where=histy_1d_total > 0)

        fig, ax = plt.subplots()
        ax.plot(x_bins[:-1], efficiencyx_1d, color='blue', label='Efficiency X')
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Efficiency')
        ax.legend()
        ax.set_title('1D Efficiency X')

        fig, ax = plt.subplots()
        ax.plot(y_bins[:-1], efficiencyy_1d, color='red', label='Efficiency Y')
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Efficiency')
        ax.legend()
        ax.set_title('1D Efficiency Y')

        # Plot 1D histograms of hits and total
        fig, ax = plt.subplots()
        ax.plot(x_bins[:-1], histx_1d_hits, color='blue', label='Hits X')
        ax.plot(x_bins[:-1], histx_1d_total, color='blue', linestyle='--', label='Total X')
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Counts')
        ax.legend()
        ax.set_title('1D Hits X')

        fig, ax = plt.subplots()
        ax.plot(y_bins[:-1], histy_1d_hits, color='red', label='Hits Y')
        ax.plot(y_bins[:-1], histy_1d_total, color='red', linestyle='--', label='Total Y')
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Counts')
        ax.legend()
        ax.set_title('1D Hits Y')


def fit_residuals(x_res, y_res, n_bins=200):
    # Get residuals between 5th and 95th percentile
    x_res = np.array(x_res)
    y_res = np.array(y_res)
    x_res = x_res[(x_res > np.percentile(x_res, 5)) & (x_res < np.percentile(x_res, 95))]
    y_res = y_res[(y_res > np.percentile(y_res, 5)) & (y_res < np.percentile(y_res, 95))]

    # Bin residuals with numpy
    x_counts, x_bin_edges = np.histogram(x_res, bins=n_bins)
    y_counts, y_bin_edges = np.histogram(y_res, bins=n_bins)

    # Fit gaussians to residuals
    x_bins = (x_bin_edges[1:] + x_bin_edges[:-1]) / 2
    y_bins = (y_bin_edges[1:] + y_bin_edges[:-1]) / 2

    fit_bounds = [(-np.inf, -np.inf, 0), (np.inf, np.inf, np.inf)]
    p0_x = [np.max(x_counts), np.mean(x_res), np.std(x_res)]
    p0_y = [np.max(y_counts), np.mean(y_res), np.std(y_res)]
    try:
        x_popt, x_pcov = cf(gaus, x_bins, x_counts, p0=p0_x, bounds=fit_bounds)
        y_popt, y_pcov = cf(gaus, y_bins, y_counts, p0=p0_y, bounds=fit_bounds)

        return x_popt, y_popt
    except (RuntimeError, ValueError):  # Except runtime or value error
        return [np.max(x_counts), np.mean(x_res), np.std(x_res)], [np.max(y_counts), np.mean(y_res), np.std(y_res)]


def fit_residuals_return_err(x_res, y_res, n_bins=200):
    # Get residuals between 5th and 95th percentile
    x_res = np.array(x_res)
    y_res = np.array(y_res)
    x_res = x_res[(x_res > np.percentile(x_res, 5)) & (x_res < np.percentile(x_res, 95))]
    y_res = y_res[(y_res > np.percentile(y_res, 5)) & (y_res < np.percentile(y_res, 95))]

    # Bin residuals with numpy
    x_counts, x_bin_edges = np.histogram(x_res, bins=n_bins)
    y_counts, y_bin_edges = np.histogram(y_res, bins=n_bins)

    # Fit gaussians to residuals
    x_bins = (x_bin_edges[1:] + x_bin_edges[:-1]) / 2
    y_bins = (y_bin_edges[1:] + y_bin_edges[:-1]) / 2

    fit_bounds = [(-np.inf, -np.inf, 0), (np.inf, np.inf, np.inf)]
    p0_x = [np.max(x_counts), np.mean(x_res), np.std(x_res)]
    p0_y = [np.max(y_counts), np.mean(y_res), np.std(y_res)]
    try:
        x_popt, x_pcov = cf(gaus, x_bins, x_counts, p0=p0_x, bounds=fit_bounds)
        y_popt, y_pcov = cf(gaus, y_bins, y_counts, p0=p0_y, bounds=fit_bounds)

        return x_popt, y_popt, np.sqrt(np.diag(x_pcov)), np.sqrt(np.diag(y_pcov))
    except (RuntimeError, ValueError):  # Except runtime or value error
        return [np.max(x_counts), np.mean(x_res), np.std(x_res)], [np.max(y_counts), np.mean(y_res), np.std(y_res)], None, None


def get_rays_in_sub_det(det, sub_det, x_rays, y_rays, event_num_rays, tolerance=0.0):
    """
    Get rays that are within the sub-detector.
    :param det:
    :param sub_det:
    :param x_rays:
    :param y_rays:
    :param event_num_rays:
    :param tolerance: Tolerance in mm for ray to be in sub-detector.
    :return:
    """
    x_rays_sub, y_rays_sub, event_num_rays_sub = [], [], []
    for x_ray, y_ray, event_num_ray in zip(x_rays, y_rays, event_num_rays):
        if det.in_sub_det(sub_det.sub_index, x_ray, y_ray, det.center[2], tolerance):
            x_rays_sub.append(x_ray)
            y_rays_sub.append(y_ray)
            event_num_rays_sub.append(event_num_ray)
    return x_rays_sub, y_rays_sub, event_num_rays_sub


def plot_xy_residuals_2d(xs_ref, ys_ref, xs_meas, ys_meas, title_post=None):
    """
    Plot residuals of measured x and y positions vs reference x and y positions.
    :param xs_ref: Reference x positions.
    :param ys_ref: Reference y positions.
    :param xs_meas: Measured x positions.
    :param ys_meas: Measured y positions.
    :param title_post: Postfix for title containing sub-detector info
    :return:
    """

    # 2D plot of detector centroids and ray hits with red line connecting them
    fig, ax = plt.subplots()
    ax.scatter(xs_ref, ys_ref, color='blue', label='Reference', marker='.', alpha=0.5)
    ax.scatter(xs_meas, ys_meas, color='green', label='Measured', marker='.', alpha=0.5)
    for x_ref, y_ref, x_meas, y_meas in zip(xs_ref, ys_ref, xs_meas, ys_meas):
        ax.plot([x_ref, x_meas], [y_ref, y_meas], color='red', linewidth=0.5)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.legend()
    ax.set_title(f'2D Hit Centroids and Rays {title_post}')
    fig.tight_layout()

    # 2D plot of ray hits on x-axis and detector centroids on y-axis
    fig, axs = plt.subplots(ncols=2)
    axs[0].scatter(xs_ref, xs_meas, color='blue', label='X Residuals')
    axs[1].scatter(ys_ref, ys_meas, color='red', label='Y Residuals')
    axs[0].set_xlabel('Reference Position (mm)')
    axs[1].set_xlabel('Reference Position (mm)')
    axs[0].set_ylabel('Measured Position (mm)')
    axs[1].set_ylabel('Measured Position (mm)')
    axs[0].legend()
    axs[1].legend()
    axs[0].grid(zorder=0)
    axs[1].grid(zorder=0)
    fig.suptitle(f'Detector Centroids vs Ray Hits {title_post}')
    fig.tight_layout()

    x_res = xs_meas - xs_ref
    y_res = ys_meas - ys_ref

    if len(x_res) == 0 or len(y_res) == 0:
        print('No residuals to plot')
        return

    n_bins = len(x_res) // 5 if len(x_res) // 5 < 200 else 200
    n_bins = 10 if n_bins < 10 else n_bins
    x_popt, y_popt, x_perr, y_perr = fit_residuals_return_err(x_res, y_res, n_bins)


    if x_popt is None or y_popt is None or x_perr is None or y_perr is None:
        print(f'Error fitting residuals for {title_post}')
        return

    x_fit_str = f'Mean={Measure(x_popt[1], x_perr[1]) * 1000}μm\nWidth={Measure(x_popt[2], x_perr[2]) * 1000}μm'
    y_fit_str = f'Mean={Measure(y_popt[1], y_perr[1]) * 1000}μm\nWidth={Measure(y_popt[2], y_perr[2]) * 1000}μm'

    # Histogram of x residuals
    fig, ax = plt.subplots()
    ax.hist(x_res, bins=500, color='blue', histtype='step')
    ax_twin = ax.twinx()
    x_res = np.array(x_res)
    x_res = x_res[(x_res > np.percentile(x_res, 5)) & (x_res < np.percentile(x_res, 95))]
    if len(x_res) < 10:
        return
    x_counts, x_bin_edges = np.histogram(x_res, bins=n_bins)
    x_bins = (x_bin_edges[1:] + x_bin_edges[:-1]) / 2
    ax_twin.bar(x_bins, x_counts, width=x_bin_edges[1] - x_bin_edges[0], color='blue', alpha=0.5)
    x_plot_xs = np.linspace(x_bin_edges[0], x_bin_edges[-1], 1000)
    ax_twin.plot(x_plot_xs, gaus(x_plot_xs, *x_popt), color='red', linestyle='-', label='X Fit')
    ax_twin.set_xlim(x_popt[1] - 5 * x_popt[2], x_popt[1] + 5 * x_popt[2])
    ax.set_xlabel('X Residual (mm)')
    ax.set_ylabel('Events (Step Histogram)')
    ax_twin.set_ylabel('Events (Filled&Fit Histogram)')
    ax.set_title(f'X Residuals Histogram {title_post}')
    ax.annotate(x_fit_str, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    fig.tight_layout()

    print(f'X Residuals: Mean={int(x_popt[1] * 1000)}μm, Std={int(x_popt[2] * 1000)}μm')


    # Histogram of y residuals
    fig, ax = plt.subplots()
    ax.hist(y_res, bins=500, color='green', histtype='step')
    ax_twin = ax.twinx()
    y_res = np.array(y_res)
    y_res = y_res[(y_res > np.percentile(y_res, 5)) & (y_res < np.percentile(y_res, 95))]
    if len(y_res) < 10:
        return
    y_counts, y_bin_edges = np.histogram(y_res, bins=n_bins)
    y_bins = (y_bin_edges[1:] + y_bin_edges[:-1]) / 2
    ax_twin.bar(y_bins, y_counts, width=y_bin_edges[1] - y_bin_edges[0], color='green', alpha=0.5)
    y_plot_xs = np.linspace(y_bin_edges[0], y_bin_edges[-1], 1000)
    ax_twin.plot(y_plot_xs, gaus(y_plot_xs, *y_popt), color='red', linestyle='-', label='Y Fit')
    ax_twin.set_xlim(y_popt[1] - 5 * y_popt[2], y_popt[1] + 5 * y_popt[2])
    ax.set_xlabel('Y Residual (mm)')
    ax.set_ylabel('Events (Step Histogram)')
    ax_twin.set_ylabel('Events (Filled&Fit Histogram)')
    ax.set_title(f'Y Residuals Histogram {title_post}')
    ax.annotate(y_fit_str, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    fig.tight_layout()

    print(f'Y Residuals: Mean={int(y_popt[1] * 1000)}μm, Std={int(y_popt[2] * 1000)}μm')

    # Histogram of r residuals
    r_res = np.sqrt(x_res ** 2 + y_res ** 2)
    n_r_bins = 200
    fig, ax = plt.subplots()
    ax.hist(r_res, bins=500, color='purple', histtype='step')
    r_res = np.array(r_res)
    r_res = r_res[r_res < np.percentile(r_res, 95)]
    if len(r_res) > 10:
        return
    r_counts, r_bin_edges = np.histogram(r_res, bins=n_r_bins)
    r_bins = (r_bin_edges[1:] + r_bin_edges[:-1]) / 2
    ax.bar(r_bins, r_counts, width=r_bin_edges[1] - r_bin_edges[0], color='purple', alpha=0.5)
    func = lambda x, a, alpha, xi, omega: a * skewnorm.pdf(x, alpha, xi, omega)
    p0 = [np.max(r_counts) * n_r_bins / (2 * np.pi), 0, np.mean(r_bins), 50]
    p_names = ['a', 'alpha', 'xi', 'omega']
    try:
        r_popt, r_pcov = cf(func, r_bins, r_counts, p0=p0)
        r_plot_xs = np.linspace(r_bin_edges[0], r_bin_edges[-1], 1000)
        ax.plot(r_plot_xs, func(r_plot_xs, *r_popt), color='red', linestyle='-', label='R Fit')
        print(f'R Residuals: Mean={int(r_popt[2] * 1000)}μm')
    except:
        print('Error fitting skewnorm to r residuals')
    ax.set_xlim(0, np.max(r_res))
    ax.set_xlabel('R Residual (mm)')
    ax.set_ylabel('Events')
    ax.set_title(f'R Residuals Histogram {title_post}')
    fig.tight_layout()


def plot_xy_residuals_2d_2(xs_ref, ys_ref, xs_meas, ys_meas, n_bins=20, title_post=None):
    """
    Plot residuals of measured x and y positions vs reference x and y positions.
    :param xs_ref: Reference x positions.
    :param ys_ref: Reference y positions.
    :param xs_meas: Measured x positions.
    :param ys_meas: Measured y positions.
    :param n_bins: Number of bins for histogram.
    :param title_post: Postfix for title containing sub-detector info
    :return:
    """

    # 2D plot of detector centroids and ray hits with red line connecting them
    fig, ax = plt.subplots()
    ax.scatter(xs_ref, ys_ref, color='blue', label='Reference', marker='.', alpha=0.5)
    ax.scatter(xs_meas, ys_meas, color='green', label='Measured', marker='.', alpha=0.5)
    for x_ref, y_ref, x_meas, y_meas in zip(xs_ref, ys_ref, xs_meas, ys_meas):
        ax.plot([x_ref, x_meas], [y_ref, y_meas], color='red', linewidth=0.5)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.legend()
    ax.set_title(f'2D Hit Centroids and Rays {title_post}')
    fig.tight_layout()

    x_res = xs_meas - xs_ref
    y_res = ys_meas - ys_ref

    if len(x_res) == 0 or len(y_res) == 0:
        print('No residuals to plot')
        return

    print(f'x_res: {x_res}')
    print(f'y_res: {y_res}')

    try:
        x_popt, y_popt = fit_residuals(x_res, y_res, n_bins)

        # Histogram of x residuals
        fig, ax = plt.subplots()
        ax.hist(x_res, bins=500, color='blue', histtype='step')
        if len(x_res) < 10:
            return
        x_counts, x_bin_edges = np.histogram(x_res, bins=n_bins)
        x_plot_xs = np.linspace(x_bin_edges[0], x_bin_edges[-1], 1000)
        ax.plot(x_plot_xs, gaus(x_plot_xs, *x_popt), color='red', linestyle='-', label='X Fit')
        ax.set_xlim(x_popt[1] - 5 * x_popt[2], x_popt[1] + 5 * x_popt[2])
        ax.annotate(fr'$\sigma$={int(x_popt[2] * 1000)}μm', (0.2, 0.92), xycoords='axes fraction', fontsize=14,
                    bbox=dict(facecolor='wheat', alpha=0.5), ha='center', va='center')
        ax.set_xlabel('X Residual (mm)')
        ax.set_ylabel('Events')
        ax.set_title(f'X Residuals Histogram {title_post}')
        fig.tight_layout()

        print(f'X Residuals: Mean={int(x_popt[1] * 1000)}μm, Std={int(x_popt[2] * 1000)}μm')

        # Histogram of y residuals
        fig, ax = plt.subplots()
        ax.hist(y_res, bins=500, color='green', histtype='step')
        if len(y_res) < 10:
            return
        y_counts, y_bin_edges = np.histogram(y_res, bins=n_bins)
        y_plot_xs = np.linspace(y_bin_edges[0], y_bin_edges[-1], 1000)
        ax.plot(y_plot_xs, gaus(y_plot_xs, *y_popt), color='red', linestyle='-', label='Y Fit')
        ax.set_xlim(y_popt[1] - 5 * y_popt[2], y_popt[1] + 5 * y_popt[2])
        ax.annotate(fr'$\sigma$={int(y_popt[2] * 1000)}μm', (0.2, 0.92), xycoords='axes fraction', fontsize=14,
                    bbox=dict(facecolor='wheat', alpha=0.5), ha='center', va='center')
        ax.set_xlabel('Y Residual (mm)')
        ax.set_ylabel('Events')
        ax.set_title(f'Y Residuals Histogram {title_post}')
        fig.tight_layout()

        print(f'Y Residuals: Mean={int(y_popt[1] * 1000)}μm, Std={int(y_popt[2] * 1000)}μm')

        # Histogram of r residuals
        r_res = np.sqrt(x_res ** 2 + y_res ** 2)
        n_r_bins = n_bins
        fig, ax = plt.subplots()
        ax.hist(r_res, bins=500, color='purple', histtype='step')
        r_res = np.array(r_res)
        r_res = r_res[r_res < np.percentile(r_res, 95)]
        if len(r_res) > 10:
            return
        r_counts, r_bin_edges = np.histogram(r_res, bins=n_r_bins)
        r_bins = (r_bin_edges[1:] + r_bin_edges[:-1]) / 2
        ax.bar(r_bins, r_counts, width=r_bin_edges[1] - r_bin_edges[0], color='purple', alpha=0.5)
        func = lambda x, a, alpha, xi, omega: a * skewnorm.pdf(x, alpha, xi, omega)
        p0 = [np.max(r_counts) * n_r_bins / (2 * np.pi), 0, np.mean(r_bins), 50]
        p_names = ['a', 'alpha', 'xi', 'omega']
        try:
            r_popt, r_pcov = cf(func, r_bins, r_counts, p0=p0)
            r_plot_xs = np.linspace(r_bin_edges[0], r_bin_edges[-1], 1000)
            ax.plot(r_plot_xs, func(r_plot_xs, *r_popt), color='red', linestyle='-', label='R Fit')
            print(f'R Residuals: Mean={int(r_popt[2] * 1000)}μm')
        except:
            print('Error fitting skewnorm to r residuals')
        ax.set_xlim(0, np.max(r_res))
        ax.set_xlabel('R Residual (mm)')
        ax.set_ylabel('Events')
        ax.set_title(f'R Residuals Histogram {title_post}')
        fig.tight_layout()
    except Exception as e:
        print(f'Error: {e}')


def plot_ray_hits_2d(det, ray_data):
    """
    Plot xy position of ray tracks when detector registers a hit.
    :param det: Detector object.
    :param ray_data: M3RefTracking object.
    :return:
    """
    all_x_rays, all_y_rays = [], []
    for sub_det in det.sub_detectors:
        event_nums_i, centroids_i = sub_det.get_event_centroids()
        event_nums_i = list(event_nums_i)
        x_rays, y_rays, event_num_rays = ray_data.get_xy_positions(det.center[2], event_nums_i)
        all_x_rays.extend(list(x_rays))
        all_y_rays.extend(list(y_rays))

    fig, ax = plt.subplots()
    ax.set_title(f'{det.name} hits')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    # Draw a thin square at -100, -100, 100, 100
    # ax.plot([-100, 100, 100, -100, -100], [-100, -100, 100, 100, -100], color='black', linewidth=0.5)
    # Make a thin box around active area based on det.active_size
    corners = [[0, 0, 0], [0, det.active_size[1], 0], [det.active_size[0], det.active_size[1], 0], [det.active_size[0], 0, 0], [0, 0, 0]]
    corners = det.convert_coords_to_global(corners)
    ax.plot([corner[0] for corner in corners], [corner[1] for corner in corners], color='black', linewidth=0.5)

    ax.scatter(all_x_rays, all_y_rays, marker='o', color='blue', alpha=0.05)
    ax.scatter(0, 0, marker='x', color='red')
    text = f'{len(all_x_rays)} hits'
    ax.annotate(text, (0.2, 0.92), xycoords='axes fraction', fontsize=14, bbox=dict(facecolor='wheat', alpha=0.5),
                ha='center', va='center')
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    fig.tight_layout()


def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def linear(x, a, b):
    return a * x + b


if __name__ == '__main__':
    main()
