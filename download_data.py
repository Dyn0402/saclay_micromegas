#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 14 10:38 AM 2023
Created in PyCharm
Created as saclay_micromegas/download_data.py

@author: Dylan Neff, Dylan
"""

import os
from subprocess import Popen, PIPE
from datetime import datetime


def main():
    download()
    print('donzo')


def download():
    data_set = 'Fe_data'
    data_sets = {'Fe_data': {'remote_path_suf': '', 'local_path': 'F:/Saclay/TestBeamData/2023_July_Saclay/dec6/'}
                 }

    bw_limit = None  # bandwidth limit per energy in Mbps or None
    size_tolerance = 0.001  # percentage tolerance between remote and local sizes, re-download if different
    earliest_date = datetime(2023, 12, 6)

    remote_path = 'banco_ext:/mnt/TestBeamData/2023_July_Saclay'

    remote_path += data_sets[data_set]['remote_path_suf']
    local_path = data_sets[data_set]['local_path']

    missing_files, missing_size = [], 0
    expected_files = get_expected_list(remote_path)
    expected_fdf_files = {file: file_data for file, file_data in expected_files.items() if file.endswith('.fdf')}
    for file, file_data in expected_fdf_files.items():
        file_size = file_data['size']
        local_file_path = f'{local_path}{file}'
        if os.path.isfile(local_file_path):
            local_size = os.path.getsize(local_file_path)
            size_frac = (local_size - file_size) / file_size if file_size > 0 else 1 if local_size > 0 else 0
            if abs(size_frac) <= size_tolerance:
                continue
        if file_data['date'] < earliest_date:
            continue
        missing_files.append(file)
        missing_size += file_size
    total_missing = len(missing_files)
    print(f'{total_missing} of {len(expected_fdf_files)} files missing, {missing_size / 1e9} GB')

    if total_missing > 0:
        res = input(f'\nDownload {total_missing} missing files? Enter yes to download all'
                    f' or anything else to quit: \n')
        if res.strip().lower() in ['yes', 'y']:
            start_download_sftp(missing_files, remote_path, local_path, bw_limit)


def get_expected_list(remote_path):
    cmd = f'\'ls -l\'|sftp {remote_path}'
    process = Popen(["powershell", cmd], stdout=PIPE, stderr=PIPE, text=True)
    stdout, stderr = process.communicate()

    files_str = stdout.split('\n')
    files_dict = {}
    for file in files_str:
        file = file.split()
        if len(file) == 9:
            # Will break when the year changes
            date = datetime.strptime(f'{file[5]} {file[6]} 2023 {file[7]}', '%b %d %Y %H:%M')
            files_dict.update({file[-1]: {'size': int(file[4]), 'date': date}})

    return files_dict


def start_download_sftp(files, remote_path, local, bw_limit=None):
    remote_host, remote_path = remote_path.split(':')
    bat_file_name = f'sftp_file.bat'

    sftp_file_name = f'sftp_file.txt'
    sftp_gets = [f'get {remote_path}/{file} {local}{file}\n' for file in files]
    sftp_gets.insert(0, 'progress\n')  # Show progress of downloads
    with open(sftp_file_name, 'w') as temp_txt:
        temp_txt.writelines(sftp_gets)

    bw_limit_str = '' if bw_limit is None else f'-l {int(bw_limit * 1000)}'
    command = f'sftp -b {sftp_file_name} {bw_limit_str} {remote_host}'

    with open(bat_file_name, 'w') as temp_bat:  # Need bat file to delete itself and sftp batch
        temp_bat.write(f'{command}\n')  # Run sftp batch file
        temp_bat.write(f'del {sftp_file_name}\n')  # Delete sftp batch file
        temp_bat.write(f'start /b "" cmd /c del {bat_file_name}&exit /b')  # Have bat file delete itself

    info = f'{len(files)} files:'
    print(f'{info} {command}')
    print('  '.join(sftp_gets))

    os.system(f'start cmd /c {bat_file_name}')  # Run batch file in new terminal


if __name__ == '__main__':
    main()
