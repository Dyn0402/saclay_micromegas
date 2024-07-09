#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 09 6:23 PM 2024
Created in PyCharm
Created as saclay_micromegas/root_struct_comp.py

@author: Dylan Neff, Dylan
"""

import numpy as np

import uproot
import awkward as ak
import vector


def main():
    old_file_path = ('F:/Saclay/TestBeamData/2023_July_Saclay/raw_root/'
                     'URW_STRIPMESH_400_STRIPDRIFT_600_231201_14H31_000_01.root')
    new_file_path = ('F:/Saclay/cosmic_data/hv_scan_7-6-24/hv1/filtered_root/'
                     'CosTb_hv1_datrun_240706_21H39_000_03_decoded_array_filtered.root')
    new_ped_file_path = ('F:/Saclay/cosmic_data/hv_scan_7-6-24/hv1/decoded_root/'
                         'CosTb_hv1_pedthr_240706_21H39_000_03_decoded_array.root')

    # old_data = read_det_data(old_file_path, variable_name='StripAmpl', tree_name='T')

    print()

    new_data = read_det_data(new_file_path)

    print()

    new_ped_data = read_det_data(new_ped_file_path)

    print('donzo')


def read_det_data(file_path, variable_name='amplitude', tree_name='nt'):
    vector.register_awkward()
    # Open the ROOT file with uproot
    root_file = uproot.open(file_path)

    # Access the tree in the file
    tree = root_file[tree_name]

    print(tree[variable_name].array())
    print(len(tree[variable_name].array()))
    print(len(tree[variable_name].array()[0]))
    print()

    # Get the variable data from the tree
    variable_data = ak.to_numpy(tree[variable_name].array())
    root_file.close()

    return variable_data


if __name__ == '__main__':
    main()
