#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 03 1:01 PM 2024
Created in PyCharm
Created as saclay_micromegas/m3_ref_hv_cal.py

@author: Dylan Neff, Dylan
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak
import vector

from dream_functions import *


def main():
    vector.register_awkward()
    hvs = [440, 445, 450, 455, 460]
    # hvs = [440]
    root_dir = 'C:/Users/Dylan/Desktop/m3_ref_hv_cal/'
    noise_sigmas = 5

    data_files, ped_files = {}, {}
    for file_name in os.listdir(root_dir):
        for hv in hvs:
            if f'_mesh{hv}_' in file_name and '_array.root' in file_name:
                if '_pedthr_' in file_name:
                    ped_files[hv] = f'{root_dir}{file_name}'
                elif '_datrun_' in file_name:
                    data_files[hv] = f'{root_dir}{file_name}'
                else:
                    print('?????')
    for hv in hvs:
        print(f'Mesh {hv}V')
        peds = read_det_data(ped_files[hv], variable_name='amp', tree_name='nt')
        peds = np.reshape(peds, (peds.shape[0], 8, peds.shape[1] // 8, peds.shape[2]))
        # print(peds.shape)
        pedestals = get_pedestals(peds)
        # print(pedestals)
        # print(pedestals.shape)
        ped_common_noise = get_common_noise(peds, pedestals)
        ped_fits = get_pedestal_fits(peds, common_noise=ped_common_noise)
        pedestals, ped_rms = ped_fits['mean'], ped_fits['sigma']
        # noise_thresholds = get_noise_thresholds(ped_rms, noise_sigmas=noise_sigmas)

        data = read_det_data(data_files[hv], variable_name='amp', tree_name='nt')
        # print(data.shape)
        data = np.reshape(data, (data.shape[0], 8, data.shape[1] // 8, data.shape[2]))
        # print(data.shape)
        data = subtract_pedestal(data, pedestals)
        # print(data.shape)
        sample_maxes = get_sample_max(data)
        # print(sample_maxes.shape)
        event_maxes = get_sample_max(sample_maxes)
        # print(event_maxes.shape)

        fig, ax = plt.subplots()
        for dream_i, dream in enumerate(np.transpose(event_maxes)):
            ax.hist(dream, bins=np.arange(-0.5, 4200.5, 50), histtype='step', label=f'Dream {dream_i}')
        ax.legend()
        ax.set_yscale('log')
        ax.set_xlabel('ADC')
        ax.set_ylabel('Events')
        ax.set_title(f'Mesh {hv}V Event Maxes')
        fig.tight_layout()
    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
