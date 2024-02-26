#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 16 1:09 PM 2024
Created in PyCharm
Created as saclay_micromegas/convert_fdf_to_root.py

@author: Dylan Neff, Dylan
"""

import os
import sys
from datetime import datetime
import shutil
from time import sleep


def main():
    """
    Currently have to push to server and run this manually
    :return:
    """
    run_type = 'convert_batch'
    # run_type = 'convert_on_fly'
    input_fdf_dir = '/mnt/nas_clas12/DATA/CosmicBench/2024/W05/'
    output_root_dir = '/local/home/usernsw/dylan/output_root/'
    tracking_run_dir = '/local/home/usernsw/dylan/tracking/'
    signal_run_dir = '/local/home/usernsw/dylan/signal/'
    tracking_sh_file = '/local/home/usernsw/dylan/tracking/run_tracking_single.sh'
    signal_sh_file = '/local/home/usernsw/dylan/signal/run_data_reader_single.sh'

    start_date = datetime(2024, 2, 20, 15, 30)
    end_date = datetime(2024, 2, 20, 17, 0)
    ref_fdf = 1
    signal_feus = ['03']

    if run_type == 'convert_batch':
        convert_batch(input_fdf_dir, output_root_dir, tracking_run_dir, signal_run_dir, tracking_sh_file,
                      signal_sh_file, start_date, end_date, ref_fdf, signal_feus)
    elif run_type == 'convert_on_fly':
        convert_on_fly(input_fdf_dir, output_root_dir, start_date, tracking_sh_file, signal_sh_file, tracking_run_dir,
                       signal_run_dir, ref_fdf, signal_feus)

    print('donzo')


def convert_batch(input_fdf_dir, output_root_dir, tracking_run_dir, signal_run_dir, tracking_sh_file, signal_sh_file,
                  start_date, end_date, ref_fdf, signal_feus):
    verbose = False
    if len(sys.argv) == 2:
        verbose = bool(sys.argv[1])

    fdf_files = get_files(input_fdf_dir, start_date, end_date)
    ref_fdf_files = [file for file in fdf_files if get_feu_num_from_file_name(file) == ref_fdf]
    fdf_runs = []
    for fdf_file in ref_fdf_files:
        # fdf_run = fdf_file[:fdf_file.rfind('_', 0, fdf_file.rfind('_'))].split('/')[-1]
        fdf_run = get_run_name_from_file_name(fdf_file)
        fdf_runs.append(fdf_run)

    for fdf_run in fdf_runs:
        n_files = len([file for file in ref_fdf_files if fdf_run in file])
        file_nums = range(n_files)
        print(f'Processing {fdf_run} with {n_files} files')
        get_rays_from_fdf(fdf_run, tracking_sh_file, file_nums, output_root_dir, tracking_run_dir, verbose)
        get_signal_from_fdf(fdf_run, signal_sh_file, file_nums, output_root_dir, signal_run_dir, signal_feus, verbose)

    # signal_fdf_files = [file for file in fdf_files if get_fdf_num_from_file_name(file) in signal_fdfs]


def convert_on_fly(fdf_dir, out_root_dir, start_date, tracking_sh_file, signal_sh_file, tracking_run_dir,
                   signal_run_dir, ref_fdf, signal_feus, check_wait=5):
    while True:
        existing_root_files = get_files(out_root_dir, start_date, file_type='root', flag=None)
        new_fdfs = check_for_new_fdfs(fdf_dir, existing_root_files, start_date)
        ref_fdf_files = [file for file in fdf_files if get_feu_num_from_file_name(file) == ref_fdf]
        new_run_names_nums = get_run_names_nums_from_files(new_fdfs)
        for run_name, file_nums in new_run_names_nums.items():
            get_rays_from_fdf(run_name, tracking_sh_file, file_nums, out_root_dir, tracking_run_dir)
            get_signal_from_fdf(run_name, signal_sh_file, file_nums, out_root_dir, signal_run_dir, signal_feus)
        sleep(check_wait * 60)  # Wait for check_wait minutes before checking again for new fdfs.


def get_run_names_nums_from_files(file_names):
    run_names_nums = {}
    for file_name in file_names:
        run_name = get_run_name_from_file_name(file_name)
        file_num = get_file_num_from_file_name(file_name)
        if run_name not in run_names_nums:
            run_names_nums.update({run_name: [file_num]})
        else:
            run_names_nums[run_name].append(file_num)

    return run_names_nums


def check_for_new_fdfs(fdf_dir, existing_root_files, start_date):
    fdf_files = get_files(fdf_dir, start_date, file_type='fdf', flag='_datrun_')
    new_fdfs = list(set(fdf_files).difference(existing_root_files))

    return new_fdfs


def get_rays_from_fdf(fdf_run, tracking_sh_file, file_nums, output_root_dir, run_dir, verbose=False):
    """
    Get rays from fdf files and write to root file.
    :param fdf_run:
    :param tracking_sh_file:
    :param file_nums:
    :param output_root_dir:
    :param run_dir:
    :param verbose:
    :return:
    """
    os.chdir(run_dir)
    for i in file_nums:
        print(f'Processing file {i}')
        temp_sh_file = make_temp_sh_file(fdf_run, tracking_sh_file, i, 'tracking')
        cmd = f'{temp_sh_file}'
        if not verbose:
            cmd += ' > /dev/null'
        os.system(cmd)
        shutil.move(f'output_{i:03d}.root', f'{output_root_dir}{fdf_run}_{i:03d}_rays.root')


def get_signal_from_fdf(fdf_run, signal_sh_file, file_nums, output_root_dir, run_dir, signal_feus, verbose=False):
    """
    Get signal from fdf files and write to root file.
    :param fdf_run:
    :param signal_sh_file:
    :param file_nums:
    :param output_root_dir:
    :param run_dir:
    :param signal_feus:
    :param verbose:
    :return:
    """
    os.chdir(run_dir)
    for feu in signal_feus:
        for i in file_nums:
            print(f'Processing feu {feu} file {i}')
            temp_sh_file = make_temp_sh_file(fdf_run, signal_sh_file, i, 'signal', feu)
            cmd = f'{temp_sh_file}'
            if not verbose:
                cmd += ' > /dev/null'
            os.system(cmd)
            shutil.move(f'output_signal_{i:03d}_{feu}.root', f'{output_root_dir}{fdf_run}_{i:03d}_{feu}.root')
        shutil.move(f'output_ped_{feu}.root', f'{output_root_dir}{fdf_run.replace("_datrun_", "_pedthr_")}_{feu}.root')


def make_temp_sh_file(fdf_run, ref_sh_file, file_num, sh_file_type='tracking', feu='01'):
    """
    Make the tracking shell script file to run the tracking program from reference file.
    :param fdf_run:
    :param ref_sh_file:
    :param file_num:
    :param sh_file_type:
    :param feu:
    :return:
    """
    # Copy tracking_sh_file to new file
    temp_file_name = ref_sh_file.replace('.sh', f'_{sh_file_type}_temp.sh')
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)
    shutil.copy(ref_sh_file, temp_file_name)
    with open(temp_file_name, 'r') as file:
        file_text = file.read()
    # Replace reference fdf file in tracking_sh_file with ref_fdf_file
    file_text = file_text.replace('CosTb_380V_stats_datrun_240212_11H42', fdf_run)
    file_text = file_text.replace('CosTb_380V_stats_pedthr_240212_11H42',
                                  fdf_run.replace('_datrun_', '_pedthr_'))
    file_text = file_text.replace('file_num=0', f'file_num={file_num}')
    if 'feu="03"' in file_text:
        file_text = file_text.replace('feu="03"', f'feu="{feu}"')
    # Write new file
    with open(temp_file_name, 'w') as file:
        file.write(file_text)
    # Make file executable
    os.system(f'chmod +x {temp_file_name}')
    return temp_file_name


def make_signal_sh_file(fdf_run, signal_sh_file, n_files):
    """
    Make the shell script file to run the signal program from fdf files.
    :param fdf_run:
    :param signal_sh_file:
    :param n_files:
    :return:
    """
    # Copy tracking_sh_file to new file
    temp_file_name = signal_sh_file.replace('.sh', '_temp.sh')
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)
    shutil.copy(signal_sh_file, temp_file_name)
    with open(temp_file_name, 'r') as file:
        file_text = file.read()
    # Replace reference fdf file in tracking_sh_file with ref_fdf_file
    file_text = file_text.replace('CosTb_380V_stats_datrun_240212_11H42', fdf_run)
    file_text = file_text.replace('CosTb_380V_stats_pedthr_240212_11H42',
                                  fdf_run.replace('_datrun_', '_pedthr_'))
    file_text = file_text.replace('nfiles=1', f'nfiles={n_files}')
    # Write new file
    with open(temp_file_name, 'w') as file:
        file.write(file_text)
    # Make file executable
    os.system(f'chmod +x {temp_file_name}')
    return temp_file_name


def get_files(in_dir, start_date=None, end_date=None, file_type='fdf', flag='_datrun_'):
    files = []
    for file in os.listdir(in_dir):
        file_path = in_dir + file
        good_file = os.path.isfile(file_path) and file_path.endswith(f'.{file_type}')
        if flag is not None:
            good_file = good_file and flag in file_path
        if good_file:
            file_date = get_date_from_file_name(file)
            if start_date is not None and file_date < start_date:
                continue
            if end_date is not None and file_date > end_date:
                continue
            files.append(file_path)
    return files


def get_date_from_file_name(file_name):
    """
    Get date from file name with format ...xxx_xxx_240212_11H42_000_01.xxx
    :param file_name:
    :return:
    """
    date_str = file_name.split('_')[-4] + ' ' + file_name.split('_')[-3]
    date = datetime.strptime(date_str, '%y%m%d %HH%M')
    return date


def get_feu_num_from_file_name(file_name):
    """
    Get fdf style feu number from file name with format ...xxx_xxx_240212_11H42_000_01.xxx
    :param file_name:
    :return:
    """
    fdf_num = int(file_name.split('_')[-1].split('.')[0])
    return fdf_num


def get_file_num_from_file_name(file_name):
    """
    Get fdf style file number from file name with format ...xxx_xxx_240212_11H42_000_01.xxx
    :param file_name:
    :return:
    """
    file_num = int(file_name.split('_')[-2])
    return file_num


def get_run_name_from_file_name(file_name):
    # Should work for fdfs, maybe other file types
    run_name = file_name[:file_name.rfind('_', 0, file_name.rfind('_'))].split('/')[-1]
    return run_name


if __name__ == '__main__':
    main()
