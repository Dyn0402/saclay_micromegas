#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 05 11:29 2023
Created in PyCharm
Created as saclay_micromegas/main

@author: Dylan Neff, dn277127
"""

from matplotlib.backends.backend_pdf import PdfPages

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from fe_analysis import *


def main():
    # base_path = '/local/home/dn277127/Documents/'
    # base_path = '/media/ucla/Saclay/TestBeamData/2023_July_Saclay/dec6/'
    base_path = 'F:/Saclay/'
    data_base = f'{base_path}TestBeamData/2023_July_Saclay/'
    fdf_dir = f'{base_path}fdfs/'
    raw_root_dir = f'{data_base}raw_root/'
    ped_flag = 'ped'
    connected_channels = load_connected_channels()  # Hard coded into function

    # process_fdfs(fdf_dir, raw_root_dir)
    # run_full_analysis(base_path, raw_root_dir, ped_flag, connected_channels)
    # plot_peaks_from_file(base_path, raw_root_dir, ped_flag)
    single_file_analysis(raw_root_dir, ped_flag)
    # plot_p2_coverage(raw_root_dir, ped_flag)
    # get_run_periods(fdf_dir, ped_flag)

    print('donzo')


def run_full_analysis(base_path, raw_root_dir, ped_flag, p2_connected_channels):
    num_threads = 15
    chunk_size = 10000
    print(f'{num_threads} threads, {chunk_size} chunk size')
    # run_files = ['P22_P2_2_ME_400_P2_2_DR_1000']  # If 'all' run all files found
    run_files = 'all'  # If 'all' run all files found
    urw_flag = 'URW_'
    ped_times = {'p2': '_231206_14H51_', 'urw': '_231124_16H27_'}
    edge_strips = {'p2': None, 'urw': np.array([[0, 0], [0, 1], [1, 62], [1, 63], [2, 0], [2, 1], [3, 62], [3, 63]])}
    noise_sigmas = {'p2': 3, 'urw': 1}
    connected_channels = {'p2': p2_connected_channels, 'urw': None}
    out_directory = f'{base_path}Analysis/'
    out_file_path = f'{out_directory}analysis_data.txt'
    signal_event_out_dir = f'{base_path}TestBeamData/2023_July_Saclay/signal_events/'
    read_events_from_file = True

    distance_map_dt_strp = '%y-%m-%d %H'
    distance_mapping = {
        '23-12-06 18': {'distance': 8, 'aluminum': False, 'det': 'p2'},
        '23-12-12 13': {'distance': 2, 'aluminum': False, 'det': 'p2'},
        '23-12-12 17': {'distance': 4, 'aluminum': False, 'det': 'p2'},
        '23-12-13 11': {'distance': 4, 'aluminum': True, 'det': 'p2'},
        '23-12-13 16': {'distance': 14, 'aluminum': False, 'det': 'p2'},
        '23-11-24 18': {'distance': 14, 'aluminum': False, 'det': 'urw'},
        '23-11-28 13': {'distance': 8, 'aluminum': False, 'det': 'urw'},
        '23-11-29 17': {'distance': 2, 'aluminum': False, 'det': 'urw'},
        '23-11-30 12': {'distance': 28, 'aluminum': False, 'det': 'urw'},
        '23-12-01 14': {'distance': 4, 'aluminum': False, 'det': 'urw'},
    }

    num_detectors = 2

    run_periods = get_run_periods(raw_root_dir, ped_flag, plot=True)
    distance_map_run_periods = {get_run_period(datetime.strptime(date, distance_map_dt_strp), run_periods): date
                                for date in distance_mapping.keys()}

    ped_files = [file for file in os.listdir(raw_root_dir) if file.endswith('.root') and ped_flag in file.lower()]
    pedestals, noise_thresholds = {}, {}
    for ped_file in ped_files:
        for ped_type, ped_time in ped_times.items():
            if ped_time in ped_file:
                # Get pedestal data
                ped_root_path = os.path.join(raw_root_dir, ped_file)
                peds, noise_theshs = run_pedestal(ped_root_path, num_detectors, noise_sigmas[ped_type],
                                                  connected_channels[ped_type])
                pedestals.update({ped_type: peds})
                noise_thresholds.update({ped_type: noise_theshs})

    # Get pedestal data
    # ped_root_path = os.path.join(raw_root_dir, ped_file)
    # pedestals, noise_thresholds = run_pedestal(ped_root_path, num_detectors, noise_sigmas, connected_channels)

    data_files = [os.path.join(raw_root_dir, file) for file in os.listdir(raw_root_dir)
                  if file.endswith('.root') and ped_flag not in file and
                  (run_files == 'all' or any(run_file in file for run_file in run_files))]

    process_data = [(file, pedestals, noise_thresholds, num_detectors, connected_channels, urw_flag in file,
                     edge_strips, chunk_size, out_directory, run_periods, distance_mapping, distance_map_run_periods,
                     signal_event_out_dir, read_events_from_file)
                    for file in data_files]

    # for params in process_data:
    #     analyze_file(*params)
    file_data = []
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        # with tqdm(total=len(process_data), desc='Processing Trees') as pbar:
        for file_res in tqdm(executor.map(analyze_file, *zip(*process_data)), total=len(process_data),
                             desc='Processing Trees'):
            if file_res is not None:
                file_data.append(file_res)
                # pbar.update(1)

    write_to_file(file_data, out_file_path)

    peak_analysis(file_data, run_periods)

    plt.show()


def single_file_analysis(raw_root_dir, ped_flag):
    chunk_size = 10000  # 15000
    # file_name = 'P22_P2_2_ME_400_P2_2_DR_1000_231213_15H46'  # Easier one
    # file_name = 'P22_P2_2_ME_390_P2_2_DR_990_231212'  # P2 2cm
    file_name = 'P22_P2_2_ME_390_P2_2_DR_990_231213_15'  # P2 14cm
    # file_name = 'URW_STRIPMESH_390_STRIPDRIFT_600_231130_12H51'  # URW 28cm
    # file_name = 'URW_STRIPMESH_390_STRIPDRIFT_600_231201'  # URW 4cm
    # file_name = 'URW_STRIPMESH_390_STRIPDRIFT_600_231130'  # URW 28cm
    # file_name = 'URW_STRIPMESH_390_STRIPDRIFT_600_231124'  # URW 14cm
    urw_flag = 'URW_'
    multi = False
    save_path = 'F:/Saclay/Analysis/'
    plot_pedestals = False

    if urw_flag in file_name:
        num_detectors = 4
        ped_time = '_231124_16H27_'
        connected_channels = None  # All channels connected for URW
        edge_strips = np.array([[0, 0], [0, 1], [1, 62], [1, 63], [2, 0], [2, 1], [3, 62], [3, 63]])
        noise_sigmas = 1
    else:
        num_detectors = 2
        ped_time = '_231206_14H51_'
        connected_channels = load_connected_channels()  # Hard coded into function
        # edge_strips = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 8], [0, 12],
        #                         [0, 16], [0, 20], [0, 24], [0, 28], [0, 29], [0, 30], [0, 31],
        #                         [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
        #                         [1, 10], [1, 19], [1, 20], [1, 29], [1, 30], [1, 39], [1, 40], [1, 49]])
        edge_strips = None
        noise_sigmas = 3

    ped_files = [file for file in os.listdir(raw_root_dir) if file.endswith('.root') and ped_flag in file.lower()]
    ped_file = ped_files[0] if len(ped_files) == 1 else [file for file in ped_files if ped_time in file][0]
    if len(ped_files) > 1:
        print(f'Warning: Multiple ped files found: {ped_files}.\nUsing {ped_file}')

    # Get pedestal data
    ped_root_path = os.path.join(raw_root_dir, ped_file)
    pedestals, noise_thresholds = run_pedestal(ped_root_path, num_detectors, noise_sigmas, connected_channels,
                                               plot_pedestals)
    # plt.show()

    if not multi:
        data_files = [os.path.join(raw_root_dir, file) for file in os.listdir(raw_root_dir)
                      if file.endswith('.root') and file_name in file]
        data_file = data_files[0]
        if len(data_files) > 1:
            print(f'Warning: Multiple data files found: {data_files}.\nUsing {data_file}')

        analyze_file_qa(data_file, pedestals, noise_thresholds, num_detectors, connected_channels, edge_strips,
                        chunk_size)

    # run_periods = get_run_periods(raw_root_dir, ped_flag, plot=True)
    # data_files = [os.path.join(raw_root_dir, file) for file in os.listdir(raw_root_dir)
    #               if file.endswith('.root') and ped_flag not in file]
    #
    # for data_file in data_files:
    #     if 'urw_' not in data_file.lower():
    #         continue
    #     mesh_voltage, drift_voltage, run_date = interpret_file_name(data_file, 'urw)
    #     run_period = get_run_period(run_date, run_periods)
    #     title = f'{mesh_voltage}V Mesh, {drift_voltage}V Drift, Run #{run_period}'
    #     analyze_spectra(data_file, pedestals, noise_thresholds, num_detectors, connected_channels, chunk_size, title)

    if multi:
        run_periods = get_run_periods(raw_root_dir, ped_flag, plot=True)
        strip_vs = np.arange(260, 450, 10)
        strip_vs_plot, mus_plot, mu_errs_plot = [], [], []
        for strip_v in strip_vs:
            v_file_name = file_name.replace('390', str(strip_v))
            data_files = [os.path.join(raw_root_dir, file) for file in os.listdir(raw_root_dir)
                          if file.endswith('.root') and v_file_name in file]
            if len(data_files) == 0:
                print(f'Warning: No data files found for {strip_v}V')
                continue
            data_file = data_files[0]
            if len(data_files) > 1:
                print(f'Warning: Multiple data files found: {data_files}.\nUsing {data_file}')
            mesh_voltage, drift_voltage, run_date = interpret_file_name(data_file, 'urw')
            run_period = get_run_period(run_date, run_periods)
            title = f'{mesh_voltage}V Mesh, {drift_voltage}V Drift, Run #{run_period}'
            if save_path is not None:
                save_path_file = f'{save_path}{v_file_name}.png'
            else:
                save_path_file = None
            peak_mu = analyze_spectra(data_file, pedestals, noise_thresholds, num_detectors, connected_channels,
                                      edge_strips, chunk_size, title, save_path_file)
            if peak_mu is None:
                continue
            if len(peak_mu) > 1:
                peak_mu = peak_mu[0]
            strip_vs_plot.append(strip_v)
            mus_plot.append(peak_mu.val)
            mu_errs_plot.append(peak_mu.err)

        fig, ax = plt.subplots()
        ax.errorbar(strip_vs_plot, mus_plot, yerr=mu_errs_plot, fmt='o')
        ax.set_xlabel('Strip Voltage (V)')
        ax.set_ylabel(r'Peak $\mu$ (ADC)')
        ax.set_title(r'Peak $\mu$ vs Strip Voltage')
        ax.grid()
        fig.tight_layout()

    plt.show()


def plot_p2_coverage(raw_root_dir, ped_flag):
    chunk_size = 60000
    mesh_v = 410
    drift_v = mesh_v + 600
    file_name = f'P22_P2_2_ME_{mesh_v}_P2_2_DR_{drift_v}_'
    save_path = 'F:/Saclay/Analysis/'
    plot_pedestals = False

    distance_map_dt_strp = '%y-%m-%d %H'
    distance_mapping = {
        '23-12-06 18': {'distance': 8, 'aluminum': False},
        '23-12-12 13': {'distance': 2, 'aluminum': False},
        '23-12-12 17': {'distance': 4, 'aluminum': False},
        '23-12-13 11': {'distance': 4, 'aluminum': True},
        '23-12-13 16': {'distance': 14, 'aluminum': False},
    }

    num_detectors = 2
    ped_time = '_231206_14H51_'
    connected_channels = load_connected_channels()  # Hard coded into function
    noise_sigmas = 3

    ped_files = [file for file in os.listdir(raw_root_dir) if file.endswith('.root') and ped_flag in file.lower()]
    ped_file = ped_files[0] if len(ped_files) == 1 else [file for file in ped_files if ped_time in file][0]
    if len(ped_files) > 1:
        print(f'Warning: Multiple ped files found: {ped_files}.\nUsing {ped_file}')

    # Get pedestal data
    ped_root_path = os.path.join(raw_root_dir, ped_file)
    pedestals, noise_thresholds = run_pedestal(ped_root_path, num_detectors, noise_sigmas, connected_channels,
                                               plot_pedestals)

    run_periods = get_run_periods(raw_root_dir, ped_flag, plot=True)

    data_files = [os.path.join(raw_root_dir, file) for file in os.listdir(raw_root_dir)
                  if file.endswith('.root') and file_name in file]
    distance_map_run_periods = {get_run_period(datetime.strptime(date, distance_map_dt_strp), run_periods): date
                                for date in distance_mapping.keys()}
    print(distance_map_run_periods)
    distance_fig_map = {}
    for data_file in data_files:
        mesh_voltage, drift_voltage, run_date = interpret_file_name(data_file)
        run_period = get_run_period(run_date, run_periods)
        print(data_file, run_period)
        file_run_info = distance_mapping[distance_map_run_periods[run_period]]

        title = f'{file_run_info["distance"]}cm {mesh_voltage}V Mesh, {drift_voltage}V Drift'
        title = title if not file_run_info['aluminum'] else f'{title}, Aluminum'
        fig_2d = analyze_file_p2_coverage(data_file, pedestals, noise_thresholds, num_detectors, connected_channels,
                                          chunk_size, title)
        distance = file_run_info['distance'] if not file_run_info['aluminum'] else file_run_info['distance'] + 1
        distance_fig_map.update({distance: fig_2d})  # Add 1 to distance if aluminum just for ordering

    with PdfPages(f'{save_path}P2_2_coverage_meshv_{mesh_v}.pdf') as pdf:
        for distance, fig_2d in sorted(distance_fig_map.items()):
            pdf.savefig(fig_2d)

    plt.show()


def process_fdfs(fdf_dir, raw_root_dir):
    num_threads = 15
    overwrite = False

    fdf_files = [file for file in os.listdir(fdf_dir) if file.endswith('.fdf')]
    fdf_data_list = [(file, fdf_dir, raw_root_dir, overwrite, file_i) for file_i, file in enumerate(fdf_files)]
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        with tqdm(total=len(fdf_files), desc='Processing fdfs') as pbar:
            for root_name in executor.map(process_fdf, *zip(*fdf_data_list)):
                pbar.update(1)


def plot_peaks_from_file(base_path, raw_root_dir, ped_flag):
    file_path = f'{base_path}Analysis/analysis_data.txt'
    file_data = read_from_file(file_path)

    run_periods = get_run_periods(raw_root_dir, ped_flag, plot=True)
    print(run_periods)

    peak_analysis(file_data, run_periods)


if __name__ == '__main__':
    main()
