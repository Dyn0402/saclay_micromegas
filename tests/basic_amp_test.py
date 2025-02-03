#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 03 10:34 2025
Created in PyCharm
Created as saclay_micromegas/basic_amp_test

@author: Dylan Neff, dn277127
"""

import sys
import os

sys.path.append(os.path.abspath("../Detector_Classes"))

from DreamData import DreamData


def main():
    fdf_path = '/local/home/banco/dylan/tests/audrey_test/selfTPOTFe_testrdmKel_datrun_250129_17H08_000_05.fdf'  # If going from fdf, set path here and make root_path None
    root_path = None  # If going from root, set path here and make fdf_path None
    decode_exe_path = '/local/home/banco/dylan/decode/decode'
    convert_exe_path = '/local/home/banco/dylan/decode/convert_vec_tree_to_array'
    feu_number = 5
    feu_connectors = [1, 2, 3, 4, 5, 6, 7, 8]

    if fdf_path is not None:
        os.chdir(os.path.dirname(fdf_path))
        root_path = decode_fdf(fdf_path, decode_exe_path)
        root_path = convert_to_array(root_path, convert_exe_path)

    os.chdir(os.path.dirname(root_path))
    root_dir = os.path.dirname(root_path)
    data = DreamData(root_dir, feu_number, feu_connectors)
    data.read_data()
    data.plot_hits_vs_strip()
    data.plot_amplitudes_vs_strip()

    print('donzo')


def decode_fdf(fdf_path, decode_exe_path):
    """
    Decode fdf file to root file using decode executable.
    Args:
        fdf_path: Path to fdf file to decode.
        decode_exe_path: Path to compiled decode executable.

    Returns:

    """
    fdf_name = os.path.basename(fdf_path)
    out_name = fdf_name.replace('.fdf', '.root')
    command = f'{decode_exe_path} {fdf_path} {out_name}'
    print(f'Decoding {fdf_name} to {out_name}')
    os.system(command)
    out_path = f'{fdf_path.replace(fdf_name, out_name)}'
    os.chmod(f'{out_path}', 0o777)

    return out_path


def convert_to_array(root_path, convert_exe_path):
    """
    Convert vector root file to array format using convert executable
    Args:
        root_path:
        convert_exe_path:

    Returns:

    """
    root_name = os.path.basename(root_path)
    out_name = root_name.replace('.root', '_array.root')
    command = f'{convert_exe_path} {root_path} {out_name}'
    print(f'Converting {root_name} to {out_name}')
    os.system(command)
    out_path = f'{root_path.replace(root_name, out_name)}'
    os.chmod(f'{out_path}', 0o777)

    return out_path

if __name__ == '__main__':
    main()
