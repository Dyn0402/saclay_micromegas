#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 11 7:27 PM 2025
Created in PyCharm
Created as saclay_micromegas/online_qa_plots.py

@author: Dylan Neff, Dylan
"""

import os
import resource
import sys
import numpy as np
import matplotlib.pyplot as plt

from Detector_Classes.DetectorConfigLoader import DetectorConfigLoader
from Detector_Classes.DreamDetector import DreamDetector


def main():
    # Example: limit to 2 GB of memory, kill process if exceeded
    # limit_memory(2000)

    # base_dir = 'F:/Saclay/cosmic_data/'
    # base_dir = '/local/home/dn277127/Bureau/cosmic_data/'
    base_dir = '/mnt/cosmic_data/Run/'
    # det_type_info_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    # det_type_info_dir = '/local/home/dn277127/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    det_type_info_dir = '/mnt/cosmic_data/config/detectors/'
    # out_dir = 'F:/Saclay/Analysis/Cosmic Bench/11-5-24/'
    # out_dir = '/local/home/dn277127/Bureau/cosmic_data/Analysis/'
    out_dir = '/mnt/cosmic_data/Analysis/'
    create_dir_if_not_exist(out_dir)
    # chunk_size = 5  # Number of files to process at once
    chunk_size = 0.2  # Number of files to process at once
    # base_dir = '/media/ucla/Saclay/cosmic_data/'
    # det_type_info_dir = '/home/dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    # out_dir = '/media/ucla/Saclay/Analysis/Cosmic Bench/11-5-24/'
    # chunk_size = 5  # Number of files to process at once

    # run_name = 'beam_test_fe_zs_test_10-10-25'
    # run_name = 'rd542_plein_vfp_1_fe_test_10-16-25'
    run_name = 'rd542_plein_vfp_1__10-16-25'
    run_dir = f'{base_dir}{run_name}/'
    # sub_run_name = 'quick_test_440V'
    sub_run_name = 'quick_test'

    event_nums = np.arange(0, 1000)

    out_dir = f'{out_dir}{run_name}/{sub_run_name}/'
    create_dir_if_not_exist(out_dir)

    det_single = 'rd5_plein_vfp_1_co2_fe55_10-28-25'

    file_nums = 'all'

    noise_sigma = 4

    run_json_path = f'{run_dir}run_config.json'
    data_dir = f'{run_dir}{sub_run_name}/filtered_root/'
    ped_dir = f'{run_dir}{sub_run_name}/decoded_root/'
    alignment_dir = f'{run_dir}alignments/'

    try:
        os.mkdir(alignment_dir)
    except FileExistsError:
        pass

    det_config_loader = DetectorConfigLoader(run_json_path, det_type_info_dir)

    det_config = det_config_loader.get_det_config(det_single, sub_run_name=sub_run_name)
    det = DreamDetector(config=det_config)
    print(f'FEU Num: {det.feu_num}')
    print(f'FEU Channels: {det.feu_connectors}')
    print(f'HV: {det.hv}')

    det.load_dream_data(data_dir, ped_dir, noise_sigma, file_nums, chunk_size, hist_raw_amps=True, save_waveforms=True,
                        waveform_fit_func='parabola_vectorized', trigger_list=event_nums)
    print(f'Hits shape: {det.dream_data.hits.shape}')

    det.dream_data.plot_pedestals()
    det.dream_data.plot_noise_metric()
    det.dream_data.filter_sparks(spark_filter_sigma=8)
    det.dream_data.plot_noise_metric()

    det.dream_data.plot_hits_vs_strip(print_dead_strips=True)
    det.dream_data.plot_raw_amps_2d_hist(combine_y=10)
    det.dream_data.plot_amplitudes_vs_strip()

    # plt.show()
    det.make_sub_detectors()

    det.plot_hits_1d()

    det.plot_centroids_2d()
    det.plot_xy_hit_map()

    det.dream_data.correct_for_fine_timestamps()

    sigma_x, sigma_x_err = det.dream_data.plot_event_time_maxes(max_channel=True, channels=np.arange(0, int(256 / 2)),
                                                                min_amp=None, plot=True)
    plt.title(f'Time of Max for X (Top) Strips')

    sigma_y, sigma_y_err = det.dream_data.plot_event_time_maxes(max_channel=True, channels=np.arange(int(256 / 2), 256),
                                                                min_amp=None, plot=True)
    plt.title(f'Time of Max for Y (Bottom) Strips')

    min_amp = 600
    sigma_x, sigma_x_err = det.dream_data.plot_event_time_maxes(max_channel=True, channels=np.arange(0, int(256 / 2)),
                                                                min_amp=min_amp, plot=True)
    plt.title(f'Time of Max for X (Top) Strips Min Amp {min_amp}')

    sigma_y, sigma_y_err = det.dream_data.plot_event_time_maxes(max_channel=True, channels=np.arange(int(256 / 2), 256),
                                                                min_amp=min_amp, plot=True)
    plt.title(f'Time of Max for Y (Bottom) Strips Min Amp {min_amp}')

    # Save all open plots
    save_all_figures(out_dir)

    # plt.show()

    print('donzo')


def limit_memory(max_mem_mb: int):
    """Set a memory limit (in MB) for this process."""
    soft, hard = max_mem_mb * 1024 * 1024, max_mem_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def create_dir_if_not_exist(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        os.chmod(dir_path, 0o777)


def save_all_figures(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for num in plt.get_fignums():
        fig = plt.figure(num)
        raw_title = None

        # Try figure suptitle
        if hasattr(fig, "_suptitle") and fig._suptitle is not None:
            raw_title = fig._suptitle.get_text()

        # Try first axis title if no suptitle
        if (not raw_title or not raw_title.strip()) and fig.axes:
            # Check for axes title (preferred) or ylabel/xlabel if empty
            ax = fig.axes[0]
            raw_title = ax.get_title() or ax.get_ylabel() or ax.get_xlabel()

        # Fallback: use figure label
        if not raw_title or not raw_title.strip():
            raw_title = fig.get_label() or f'figure_{num}'

        # Clean up title for filename
        fig_title = raw_title.strip().replace(' ', '_').replace('/', '-')
        if not fig_title:
            fig_title = f'figure_{num}'

        # Save files
        for ext in ('png', 'pdf'):
            path = os.path.join(out_dir, f"{fig_title}.{ext}")
            fig.savefig(path, bbox_inches='tight')
            print(f"Saved {path}")


if __name__ == '__main__':
    main()
