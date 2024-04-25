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
    vector.register_awkward()
    # comp_damien_francesco()
    comp_damien_francesco_array()
    # comp_new_old()
    # comp_zs_new_old()
    print('donzo')


def comp_damien_francesco():
    damien_path = 'C:/Users/Dylan/Desktop/decoder_test/damien.root'
    francesco_path = 'C:/Users/Dylan/Desktop/decoder_test/francesco.root'

    damien_vars = ['StripAmpl']
    damien_start = time()
    with uproot.open(damien_path) as damien_file:
        print(damien_file.keys())
        damien_tree = damien_file['T;18']
        print(damien_tree.keys())
        # damien_data = ak.to_numpy(damien_tree[damien_vars[0]].array())
        damien_data = damien_tree[damien_vars[0]].array()
        numpy_convert_start = time()
        damien_data = ak.to_numpy(damien_data)
        print(f'Numpy convert time: {time() - numpy_convert_start}s')
    print(f'Damien time: {time() - damien_start}s')
    # print(damien_data)
    # print(damien_data.shape)

    francesco_vars = ['amplitude', 'sample', 'channel']
    francesco_start = time()
    with uproot.open(francesco_path) as francesco_file:
        print(francesco_file.keys())
        francesco_tree = francesco_file['nt;18']
        print(francesco_tree.keys())
        francesco_data = francesco_tree.arrays(francesco_vars)
    print(f'Francesco time: {time() - francesco_start}s')
    # print(francesco_data)
    print(francesco_data['amplitude'])
    print(len(francesco_data['amplitude']))
    print([len(arr) for arr in francesco_data['amplitude']])
    print(ak.to_numpy(francesco_data['amplitude']))
    print(ak.to_numpy(francesco_data['amplitude']).shape)
    print(ak.to_numpy(francesco_data['sample']))
    print(ak.to_numpy(francesco_data['sample']).shape)
    print(ak.to_numpy(francesco_data['channel']))
    print(ak.to_numpy(francesco_data['channel']).shape)


def comp_damien_francesco_array():
    damien_path = 'C:/Users/Dylan/Desktop/decoder_test/damien.root'
    francesco_path = 'C:/Users/Dylan/Desktop/decoder_test/francesco_array.root'

    damien_vars = ['StripAmpl']
    damien_start = time()
    with uproot.open(damien_path) as damien_file:
        print(damien_file.keys())
        damien_tree = damien_file['T;18']
        print(damien_tree.keys())
        # damien_data = ak.to_numpy(damien_tree[damien_vars[0]].array())
        damien_data = damien_tree[damien_vars[0]].array()
        numpy_convert_start = time()
        damien_data = ak.to_numpy(damien_data)
        print(f'Numpy convert time: {time() - numpy_convert_start}s')
    print(f'Damien time: {time() - damien_start}s')
    # print(damien_data)
    print(damien_data.shape)

    francesco_vars = ['amp']
    francesco_start = time()
    with uproot.open(francesco_path) as francesco_file:
        print(francesco_file.keys())
        francesco_tree = francesco_file['nt;17']
        print(francesco_tree.keys())
        francesco_data = francesco_tree[francesco_vars[0]].array()
        numpy_convert_start = time()
        francesco_data = ak.to_numpy(francesco_data)
        print(f'Numpy convert time: {time() - numpy_convert_start}s')
    print(f'Francesco time: {time() - francesco_start}s')
    print(francesco_data.shape)


def comp_new_old():
    francesco_path = 'C:/Users/Dylan/Desktop/decoder_test/francesco.root'
    francesco_new_path = 'C:/Users/Dylan/Desktop/decoder_test/francesco_new.root'

    francesco_vars = ['amplitude', 'sample', 'channel']
    with uproot.open(francesco_path) as francesco_file:
        print(francesco_file.keys())
        francesco_tree = francesco_file['nt;18']
        print(francesco_tree.keys())
        francesco_data = francesco_tree.arrays(francesco_vars)
    print(francesco_data)
    francesco_numpy = {var: ak.to_numpy(francesco_data[var]) for var in francesco_vars}
    with uproot.open(francesco_new_path) as francesco_new_file:
        print(francesco_new_file.keys())
        francesco_new_tree = francesco_new_file['nt;18']
        print(francesco_new_tree.keys())
        francesco_new_data = francesco_new_tree.arrays(francesco_vars)
    print(francesco_new_data)
    francesco_new_numpy = {var: ak.to_numpy(francesco_new_data[var]) for var in francesco_vars}

    for var in francesco_vars:
        print(var)
        print(np.array_equal(francesco_numpy[var], francesco_new_numpy[var]))


def comp_zs_new_old():
    francesco_path = 'C:/Users/Dylan/Desktop/decoder_test/ftest_zs_oldcode.root'
    francesco_new_path = 'C:/Users/Dylan/Desktop/decoder_test/ftest_zs_mycode.root'

    francesco_vars = ['amplitude', 'sample', 'channel']
    with uproot.open(francesco_path) as francesco_file:
        print(francesco_file.keys())
        francesco_tree = francesco_file['nt;3']
        print(francesco_tree.keys())
        francesco_data = francesco_tree.arrays(francesco_vars)
    print(francesco_data)
    print(len(francesco_data))

    with uproot.open(francesco_new_path) as francesco_new_file:
        print(francesco_new_file.keys())
        francesco_new_tree = francesco_new_file['nt;3']
        print(francesco_new_tree.keys())
        francesco_new_data = francesco_new_tree.arrays(francesco_vars)
    print(francesco_new_data)
    print(len(francesco_new_data))

    any_difference, j = False, 0
    for i in range(len(francesco_data)):
        if i % 500 == 0:
            print(i)
        francesco_numpy = {var: ak.to_numpy(francesco_data[i][var]) for var in francesco_vars}
        francesco_new_numpy = {var: ak.to_numpy(francesco_new_data[i][var]) for var in francesco_vars}
        events_equal = True
        for var in francesco_vars:
            if not np.array_equal(francesco_numpy[var], francesco_new_numpy[var]):
                events_equal = False
                any_difference = True
        if not events_equal:
            print(f'Event {i} not equal')
        j += 1

    print(f'Any difference: {any_difference} in {j} events')


if __name__ == '__main__':
    main()
