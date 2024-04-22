#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 22 10:31 AM 2024
Created in PyCharm
Created as saclay_micromegas/decoder_comparison.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time

import uproot
import awkward as ak
import vector


def main():
    damien_path = 'C:/Users/Dylan/Desktop/decoder_test/damien.root'
    francesco_path = 'C:/Users/Dylan/Desktop/decoder_test/francesco.root'
    vector.register_awkward()

    damien_vars = ['StripAmpl']
    damien_start = time()
    with uproot.open(damien_path) as damien_file:
        print(damien_file.keys())
        damien_tree = damien_file['T;18']
        print(damien_tree.keys())
        damien_data = ak.to_numpy(damien_tree[damien_vars[0]].array())
    print(f'Damien time: {time() - damien_start}s')
    print(damien_data)
    print(damien_data.shape)

    francesco_vars = ['amplitude', 'sample', 'channel']
    francesco_start = time()
    with uproot.open(francesco_path) as francesco_file:
        print(francesco_file.keys())
        francesco_tree = francesco_file['nt;18']
        print(francesco_tree.keys())
        francesco_data = francesco_tree.arrays(francesco_vars)
    print(f'Francesco time: {time() - francesco_start}s')
    print(francesco_data)
    print(francesco_data['amplitude'])
    print(len(francesco_data['amplitude']))
    print([len(arr) for arr in francesco_data['amplitude']])
    print(ak.to_numpy(francesco_data['amplitude']))
    print(ak.to_numpy(francesco_data['amplitude']).shape)
    print(ak.to_numpy(francesco_data['sample']))
    print(ak.to_numpy(francesco_data['sample']).shape)
    print(ak.to_numpy(francesco_data['channel']))
    print(ak.to_numpy(francesco_data['channel']).shape)

    print('donzo')


if __name__ == '__main__':
    main()
