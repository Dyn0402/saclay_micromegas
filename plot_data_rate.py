#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 21 11:24 2024
Created in PyCharm
Created as saclay_micromegas/plot_data_rate

@author: Dylan Neff, dn277127
"""

from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


def main():
    data_files_string = """
    -rw-rw-r--. 1 1024 users 954M Feb 20 18:52 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_000_01.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 20 18:52 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_000_03.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 20 20:07 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_001_01.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 20 20:07 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_001_03.fdf
    -rw-rw-r--. 1 1024 users 955M Feb 20 21:23 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_002_01.fdf
    -rw-rw-r--. 1 1024 users 955M Feb 20 21:23 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_002_03.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 20 22:40 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_003_01.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 20 22:40 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_003_03.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 20 23:57 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_004_01.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 20 23:57 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_004_03.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 01:12 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_005_01.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 01:12 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_005_03.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 02:28 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_006_01.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 02:28 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_006_03.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 03:44 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_007_01.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 03:44 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_007_03.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 05:00 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_008_01.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 05:00 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_008_03.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 06:16 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_009_01.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 06:16 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_009_03.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 07:32 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_010_01.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 07:32 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_010_03.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 08:47 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_011_01.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 08:47 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_011_03.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 10:02 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_012_01.fdf
    -rw-rw-r--. 1 1024 users 954M Feb 21 10:02 CosTb_top_400V_bot_390V_long_run_datrun_240220_17H28_012_03.fdf
    """
    feu_dates = {}
    for line in data_files_string.split('\n'):
        if len(line.split()) == 9:
            file_name = line.split()[-1]
            file_size = int(line.split()[4][:-1])  # Assumes M
            date_str = ' '.join(line.split()[5:8])
            date = datetime.strptime(date_str, '%b %d %H:%M')
            feu = int(file_name[-6:-4])
            file_num = int(file_name[-10:-7])
            if feu in feu_dates:
                feu_dates[feu].append({'date': date, 'size': file_size, 'num': file_num})
            else:
                feu_dates[feu] = [{'date': date, 'size': file_size, 'num': file_num}]

    fig, ax = plt.subplots()
    for feu, feu_data in feu_dates.items():
        feu_data = sorted(feu_data, key=lambda x: x['date'])
        data = pd.DataFrame(feu_data)
        ax.plot(data['date'], data['size'], label=f'FEU {feu}', marker='o')
        ax.legend()
        total_size = (data['size'].sum() - data['size'].iloc[0]) / 1024  # GB
        total_time = (data['date'].iloc[-1] - data['date'].iloc[0]).total_seconds() / 3600  # hours
        rate = total_size / total_time
        print(f'FEU {feu} rate: {rate} GB/h')
    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()
