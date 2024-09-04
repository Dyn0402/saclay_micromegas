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

from M3RefTracking import M3RefTracking
from DetectorConfigLoader import DetectorConfigLoader
from Detector import Detector
from DreamDetector import DreamDetector
from DreamData import DreamData


def main():
    # base_dir = 'F:/Saclay/cosmic_data/'
    # det_type_info_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    # out_dir = 'F:/Saclay/Analysis/Cosmic Bench/9-3-24/'
    base_dir = '/local/home/dn277127/Bureau/cosmic_data/'
    det_type_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    out_dir = '/local/home/dn277127/Bureau/cosmic_data/Analysis/'
    # run_name = 'new_strip_check_7-12-24'
    # run_name = 'ig1_test1'
    # run_name = 'banco_flipped_7-8-24'
    # run_name = 'ig1_sg1_stats4'
    run_name = 'sg1_stats_7-26-24'
    run_dir = f'{base_dir}{run_name}/'
    # sub_run_name = 'hv1'
    # sub_run_name = 'new_detector_short'
    # sub_run_name = 'drift_600_resist_460'
    # sub_run_name = 'quick_test'
    # sub_run_name = 'max_hv_long'
    sub_run_name = 'max_hv_long_1'

    # det_single = 'asacusa_strip_1'
    # det_single = 'asacusa_strip_2'
    # det_single = 'strip_grid_1'
    # det_single = 'inter_grid_1'
    det_single = 'urw_inter'
    # det_single = 'urw_strip'
    # det_single = None

    # file_nums = 'all'
    # file_nums = list(range(0, 645))
    file_nums = list(range(100, 210))

    chunk_size = 5  # Number of files to process at once

    run_json_path = f'{run_dir}run_config.json'
    data_dir = f'{run_dir}{sub_run_name}/filtered_root/'
    ped_dir = f'{run_dir}{sub_run_name}/decoded_root/'
    m3_dir = f'{run_dir}{sub_run_name}/m3_tracking_root/'
    out_dir = f'{out_dir}{det_single}/'
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    z_align_range = [5, 5]  # mm range to search for optimal z position

    print(f'Getting ray data...')
    ray_data = M3RefTracking(m3_dir, single_track=True, file_nums=file_nums)

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
            det.load_dream_data(data_dir, ped_dir, 10, file_nums, chunk_size)
            print(f'Hits shape: {det.dream_data.hits.shape}')
            # det.dream_data.plot_noise_metric()
            det.dream_data.plot_pedestals()
            det.dream_data.plot_hits_vs_strip(print_dead_strips=True)
            det.dream_data.plot_amplitudes_vs_strip()
            plt.show()
            param_ranges = {'amplitude': [10, 5000]}
            det.dream_data.plot_fit_param('amplitude', param_ranges)
            det.dream_data.plot_fit_param('time_max')
            # plt.show()
            det.make_sub_detectors()
            event_nums = det.plot_xy_amp_sum_vs_event_num(True, 500, False, 15)
            det.plot_amplitude_sum_vs_event_num()
            det.plot_num_hit_xy_hist()
            print(f'Det data: {len(det.dream_data.data_amps)}')
            print(f'Ray data: {len(ray_data.ray_data)}')
            det.plot_centroids_2d()
            plot_ray_hits_2d(det, ray_data)
            det.add_rotation(90, 'z')
            det.plot_centroids_2d()
            det.plot_centroids_2d_heatmap()
            det.plot_centroids_2d_scatter_heat()
            plot_ray_hits_2d(det, ray_data)
            plt.show()

            align_dream(det, ray_data, z_align_range)

            get_residuals(det, ray_data, plot=True)

            x_subs_mean, y_subs_mean, x_subs_std, y_subs_std = get_residuals(det, ray_data, plot=False, sub_reses=True)
            pitches, resolutions, res_xs, res_ys = [], [], [], []
            for i, (x_mean, y_mean, x_std, y_std) in enumerate(zip(x_subs_mean, y_subs_mean, x_subs_std, y_subs_std)):
                x_mean, y_mean = int(x_mean * 1000), int(y_mean * 1000)
                x_std, y_std = int(x_std * 1000), int(y_std * 1000)
                pitch = det.sub_detectors[i].x_pitch
                print(f'Sub-Detector {i} (pitch: {pitch}) '
                      f'x_mean: {x_mean}μm, y_mean: {y_mean}μm, x_std: {x_std}μm, y_std: {y_std}μm')
                pitches.append(pitch)
                res_xs.append(x_std)
                res_ys.append(y_std)
                resolutions.append(np.sqrt(x_std ** 2 + y_std ** 2))
            # Sort by pitch
            pitches, resolutions, res_xs, res_ys = zip(*sorted(zip(pitches, resolutions, res_xs, res_ys)))
            print(pitches, resolutions, res_xs, res_ys)
            # Write to file in out_dir
            with open(f'{out_dir}{det.name}_res_vs_pitch.txt', 'w') as file:
                file.write(f'Pitch (mm)\tResolution (um)\tX Res (um)\tY Res (um)\n')
                for pitch, res, res_x, res_y in zip(pitches, resolutions, res_xs, res_ys):
                    file.write(f'{pitch}\t{res}\t{res_x}\t{res_y}\n')
            print(det.name)
            fig, ax = plt.subplots()
            ax.plot(pitches, resolutions, marker='o', zorder=10)
            ax.set_xlabel('Pitch (mm)')
            ax.set_ylabel('Resolution (μm)')
            ax.grid()
            fig.tight_layout()

            all_figures = [plt.figure(num) for num in plt.get_fignums()]
            for fig_i, fig in enumerate(all_figures):
                fig_name = fig.axes[0].get_title() + f'_{fig_i}'
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


def align_dream(det, ray_data, z_range):
    x_res_i_mean, y_res_i_mean, x_res_i_std, y_res_i_std = get_residuals(det, ray_data)
    det.set_center(x=det.center[0] - x_res_i_mean, y=det.center[1] - y_res_i_mean)

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
    min_res = np.min(np.sqrt(np.array(x_residuals) ** 2 + np.array(y_residuals) ** 2))
    z_min = zs[np.argmin(np.sqrt(np.array(x_residuals) ** 2 + np.array(y_residuals) ** 2))]
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

    z_rots = np.linspace(-1, 1, 30)
    x_residuals, y_residuals = [], []
    det.add_rotation(0, 'z')
    for z_rot in z_rots:
        print(f'z_rot: {z_rot}')
        det.replace_last_rotation(z_rot, 'z')
        x_res_i_mean, y_res_i_mean, x_res_i_std, y_res_i_std = get_residuals(det, ray_data)
        x_residuals.append(x_res_i_std)
        y_residuals.append(y_res_i_std)
    min_rot_res = np.min(np.sqrt(np.array(x_residuals) ** 2 + np.array(y_residuals) ** 2))
    z_rot_min = z_rots[np.argmin(np.sqrt(np.array(x_residuals) ** 2 + np.array(y_residuals) ** 2))]
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


def get_residuals(det, ray_data, sub_reses=False, plot=False):
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


def fit_residuals(x_res, y_res):
    # Get residuals between 5th and 95th percentile
    x_res = np.array(x_res)
    y_res = np.array(y_res)
    x_res = x_res[(x_res > np.percentile(x_res, 5)) & (x_res < np.percentile(x_res, 95))]
    y_res = y_res[(y_res > np.percentile(y_res, 5)) & (y_res < np.percentile(y_res, 95))]

    # Bin residuals with numpy
    x_counts, x_bin_edges = np.histogram(x_res, bins=200)
    y_counts, y_bin_edges = np.histogram(y_res, bins=200)

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
    except RuntimeError:
        return [np.max(x_counts), np.mean(x_res), np.std(x_res)], [np.max(y_counts), np.mean(y_res), np.std(y_res)]


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

    fig, ax = plt.subplots()
    ax.scatter(xs_ref, xs_meas, color='blue', label='X Residuals')
    ax.scatter(ys_ref, ys_meas, color='red', label='Y Residuals')
    ax.set_xlabel('Reference Position (mm)')
    ax.set_ylabel('Measured Position (mm)')
    ax.legend()
    ax.grid()
    fig.tight_layout()

    x_res = xs_meas - xs_ref
    y_res = ys_meas - ys_ref

    x_popt, y_popt = fit_residuals(x_res, y_res)

    fig, ax = plt.subplots()
    ax.hist(x_res, bins=500, color='blue', histtype='step')
    x_res = np.array(x_res)
    x_res = x_res[(x_res > np.percentile(x_res, 5)) & (x_res < np.percentile(x_res, 95))]
    x_counts, x_bin_edges = np.histogram(x_res, bins=200)
    x_bins = (x_bin_edges[1:] + x_bin_edges[:-1]) / 2
    ax.bar(x_bins, x_counts, width=x_bin_edges[1] - x_bin_edges[0], color='blue', alpha=0.5)
    x_plot_xs = np.linspace(x_bin_edges[0], x_bin_edges[-1], 1000)
    ax.plot(x_plot_xs, gaus(x_plot_xs, *x_popt), color='red', linestyle='-', label='X Fit')
    ax.set_xlabel('X Residual (mm)')
    ax.set_ylabel('Events')
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.hist(y_res, bins=500, color='green', histtype='step')
    y_res = np.array(y_res)
    y_res = y_res[(y_res > np.percentile(y_res, 5)) & (y_res < np.percentile(y_res, 95))]
    y_counts, y_bin_edges = np.histogram(y_res, bins=200)
    y_bins = (y_bin_edges[1:] + y_bin_edges[:-1]) / 2
    ax.bar(y_bins, y_counts, width=y_bin_edges[1] - y_bin_edges[0], color='green', alpha=0.5)
    y_plot_xs = np.linspace(y_bin_edges[0], y_bin_edges[-1], 1000)
    ax.plot(y_plot_xs, gaus(y_plot_xs, *y_popt), color='red', linestyle='-', label='Y Fit')
    ax.set_xlabel('Y Residual (mm)')
    ax.set_ylabel('Events')
    fig.tight_layout()

    print(f'X Residuals: Mean={int(x_popt[1] * 1000)}μm, Std={int(x_popt[2] * 1000)}μm')
    print(f'Y Residuals: Mean={int(y_popt[1] * 1000)}μm, Std={int(y_popt[2] * 1000)}μm')


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
    ax.plot([-100, 100, 100, -100, -100], [-100, -100, 100, 100, -100], color='black', linewidth=0.5)
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


if __name__ == '__main__':
    main()
