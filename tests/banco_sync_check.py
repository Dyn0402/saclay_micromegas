#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 15 9:26 PM 2025
Created in PyCharm
Created as saclay_micromegas/banco_sync_check.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import uproot


def main():
    base_path = 'C:/Users/Dylan/Desktop/test/'

    ladders = [160, 163, 162, 157]
    for ladder in ladders:
        file_path = f'{base_path}multinoiseScan_251015_174423-B0-ladder{ladder}.root'
        with uproot.open(file_path) as f:
            tree = f['pixTree']
            df = tree['fData'].array(library='np')
            print(f'Ladder {ladder} shape: {df.shape}')
            print(df['trgNum'])
            # Np histogram of all trgNum
            hist, edges = np.histogram(df['trgNum'], bins=np.arange(-0.5, np.max(df['trgNum']) + 1.5, 1))
            print(hist)
    print('donzo')


if __name__ == '__main__':
    main()
