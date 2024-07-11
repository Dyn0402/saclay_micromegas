#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 10 3:09 PM 2024
Created in PyCharm
Created as saclay_micromegas/add_sub_detector_to_det_maps.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    map_dir = 'C:/Users/Dylan/PycharmProjects/Cosmic_Bench_DAQ_Control/config/detectors/'
    asacusa(map_dir)
    print('donzo')


def asacusa(map_dir):
    with open(f'{map_dir}asacusa_map.txt', 'r') as file:
        lines = file.readlines()

    with open(f'{map_dir}asacusa_map_new.txt', 'w') as file:
        for i, line in enumerate(lines):
            if i == 0:
                file.write('sub_detector,' + line)
            else:
                sub_det = line.split(',')[0]
                file.write(f'{sub_det},' + line)


def inter(map_dir):
    with open(f'{map_dir}inter_map.txt', 'r') as file:
        lines = file.readlines()



    with open(f'{map_dir}inter_map_new.txt', 'w') as file:
        for i, line in enumerate(lines):
            if i == 0:
                file.write('sub_detector,' + line)
            else:
                sub_det = line.split(',')[0]
                file.write(f'{sub_det},' + line)


if __name__ == '__main__':
    main()
