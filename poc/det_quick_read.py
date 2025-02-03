#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 02 18:18 2024
Created in PyCharm
Created as saclay_micromegas/det_quick_read

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt

from Detector_Classes.M3RefTracking import M3RefTracking
from Detector_Classes.DetectorConfigLoader import DetectorConfigLoader
from Detector_Classes.DreamDetector import DreamDetector


def main():
    base_dir = 'F:/Saclay/cosmic_data/'
    det_type_info_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    # out_dir = 'F:/Saclay/Analysis/Cosmic Bench/9-24-24/'
    # base_dir = '/local/home/dn277127/Bureau/cosmic_data/'
    # det_type_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    # out_dir = '/local/home/dn277127/Bureau/cosmic_data/Analysis/9-11-24/'
    run_name = 'urw_inter_test_10-2-24'
    run_dir = f'{base_dir}{run_name}/'
    sub_run_name = 'test_1'

    # det_single = 'asacusa_strip_1'
    # det_single = 'asacusa_strip_2'
    # det_single = 'strip_grid_1'
    # det_single = 'inter_grid_1'
    det_single = 'urw_inter'
    # det_single = 'urw_strip'
    # det_single = None

    file_nums = 'all'

    chunk_size = 5  # Number of files to process at once

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
        if det_config['det_type'] == 'm3':
            continue
        if det_config['det_type'] == 'banco':
            continue
        else:  # Dream
            det = DreamDetector(config=det_config)
            # ray_data.plot_xy(det.center[2])
            # plt.show()
            print(f'FEU Num: {det.feu_num}')
            print(f'FEU Channels: {det.feu_connectors}')
            print(f'HV: {det.hv}')
            det.load_dream_data(data_dir, ped_dir, 10, file_nums, chunk_size, hist_raw_amps=True)
            print(f'Hits shape: {det.dream_data.hits.shape}')
            # det.dream_data.plot_noise_metric()
            # det.dream_data.plot_pedestals()
            det.dream_data.plot_hits_vs_strip(print_dead_strips=True)
            det.dream_data.plot_amplitudes_vs_strip()
            det.dream_data.plot_raw_amps_2d_hist()
            plt.show()
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

            align_dream(det, ray_data, z_align_range)
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
            get_banco_telescope_residuals(det, banco_telescope, plot=True)
            # plt.show()

            x_subs_mean, y_subs_mean, x_subs_std, y_subs_std = get_residuals(det, ray_data, plot=False, sub_reses=True)
            pitches_x, pitches_y, inter_pitches_x, inter_pitches_y = [], [], [], []
            resolutions, res_xs, res_ys = [], [], []
            for i, (x_mean, y_mean, x_std, y_std) in enumerate(zip(x_subs_mean, y_subs_mean, x_subs_std, y_subs_std)):
                x_mean, y_mean = int(x_mean * 1000), int(y_mean * 1000)
                x_std, y_std = int(x_std * 1000), int(y_std * 1000)
                pitch_x = det.sub_detectors[i].x_pitch
                pitch_y = det.sub_detectors[i].y_pitch
                inter_pitch_x = det.sub_detectors[i].x_interpitch
                inter_pitch_y = det.sub_detectors[i].y_interpitch
                print(f'Sub-Detector {i} '
                      f'(pitch_x: {pitch_x}, pitch_y: {pitch_y}, inter_x: {inter_pitch_x}, inter_y: {inter_pitch_y}) '
                      f'x_mean: {x_mean}μm, y_mean: {y_mean}μm, x_std: {x_std}μm, y_std: {y_std}μm')
                pitches_x.append(pitch_x)
                pitches_y.append(pitch_y)
                inter_pitches_x.append(inter_pitch_x)
                inter_pitches_y.append(inter_pitch_y)
                res_xs.append(x_std)
                res_ys.append(y_std)
                resolutions.append(np.sqrt(x_std ** 2 + y_std ** 2))
            # Sort by pitch_x
            pitches_x, pitches_y, resolutions, res_xs, res_ys = zip(
                *sorted(zip(pitches_x, pitches_y, resolutions, res_xs, res_ys)))
            print(pitches_x, pitches_y, resolutions, res_xs, res_ys)
            # Write to file in out_dir
            with open(f'{out_dir}{det.name}_res_vs_pitch.txt', 'w') as file:
                file.write(f'Pitch (mm)\tResolution (um)\tX Res (um)\tY Res (um)\n')
                for pitch_x, pitch_y, res, res_x, res_y in zip(pitches_x, pitches_y, resolutions, res_xs, res_ys):
                    file.write(f'{pitch_x}\t{pitch_y}\t{res}\t{res_x}\t{res_y}\n')
            print(det.name)
            fig, ax = plt.subplots()
            ax.plot(pitches_x, resolutions, marker='o', zorder=10)
            ax.set_xlabel('Pitch (mm)')
            ax.set_ylabel('Resolution (μm)')
            ax.grid()
            fig.tight_layout()

            all_figures = [plt.figure(num) for num in plt.get_fignums()]
            for fig_i, fig in enumerate(all_figures):
                fig_name = fig.axes[0].get_title() + f'_{fig_i}'
                fig_name = fig_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
                fig_name = fig_name.replace(':', '').replace('.', '')
                fig.savefig(f'{out_dir}{fig_name}.png')

            plt.show()
            input('Finished, press Enter to continue...')
    print('donzo')


if __name__ == '__main__':
    main()
