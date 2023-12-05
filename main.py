#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 05 11:29 2023
Created in PyCharm
Created as saclay_micromegas/main

@author: Dylan Neff, dn277127
"""

import os
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import ROOT
import uproot
import awkward as ak

# Suppress the specified warning messages
# ROOT.gErrorIgnoreLevel = ROOT.kError + ROOT.kBreak
ROOT.gErrorIgnoreLevel = -1


def main():
    fdf_dir = 'test_data/fdf/'
    raw_root_dir = 'test_data/raw_root/'
    ped_file = 'selfTPOTFe_proba_pedthr_230801_17H17_000_05.root'
    files = ['selfTPOTFe_proba_datrun_230801_17H17_000_05']

    read_fdfs = False
    num_detectors = 4

    if read_fdfs:
        for file in files + [ped_file]:
            read_fdf_to_root(file, fdf_dir, raw_root_dir)

    ped_root_path = os.path.join(raw_root_dir, ped_file)
    ped_data = read_det_data(ped_root_path, num_detectors)
    pedestals = get_pedestals(ped_data)

    for file in files:
        raw_root_path = os.path.join(raw_root_dir, file + ".root")
        data = read_det_data(raw_root_path, num_detectors)
        # plot_combined_time_series(data)
        ped_sub_data = subtract_pedestal(data, pedestals)
        plot_combined_time_series(ped_sub_data)
        max_data = get_sample_max(ped_sub_data)
        plot_position_data(max_data)

    plt.show()

    print('donzo')


def read_fdf_to_root(file, fdf_dir, raw_root_dir):
    os.system(f'./DreamDataReader {os.path.join(fdf_dir, file + ".fdf")}')
    raw_root_path = os.path.join(raw_root_dir, file + ".root")
    shutil.move('output.root', raw_root_path)
    return raw_root_path


def plot_adc_pyroot(file_path):
    # Create a ROOT TFile object to open the file
    root_file = ROOT.TFile(file_path)

    # Access the tree in the file
    tree = root_file.Get('T')

    # Create a canvas to draw the histogram
    canvas = ROOT.TCanvas("canvas", "Histogram Canvas", 800, 600)

    # Create a histogram from the tree variable
    variable_name = 'StripAmpl'
    tree.Draw(variable_name)
    histogram = ROOT.gPad.GetPrimitive("htemp")

    # Set histogram attributes
    histogram.SetTitle(f"Histogram of {variable_name}")
    histogram.GetXaxis().SetTitle(variable_name)
    histogram.GetYaxis().SetTitle("Frequency")

    # Draw the histogram on the canvas
    canvas.Draw()

    # Keep the program running to display the canvas
    ROOT.gApplication.Run()
    input()


def plot_adc_uproot(file_path):
    # Open the ROOT file with uproot
    root_file = uproot.open(file_path)

    # Access the tree in the file
    tree = root_file['T']

    # Get the variable data from the tree
    variable_name = 'StripAmpl'
    variable_data = ak.flatten(tree[variable_name].array(), axis=None)
    print(variable_data)

    # Create a histogram using Matplotlib
    plt.hist(variable_data, bins=50, edgecolor='black')

    # Set plot labels and title
    plt.xlabel(variable_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {variable_name}")

    # Display the plot
    plt.show()


def read_det_data(file_path, num_detectors=None, variable_name='StripAmpl', tree_name='T'):
    # Open the ROOT file with uproot
    root_file = uproot.open(file_path)

    # Access the tree in the file
    tree = root_file[tree_name]

    # Get the variable data from the tree
    variable_data = ak.to_numpy(tree[variable_name].array())
    root_file.close()

    if num_detectors is not None:
        variable_data = variable_data[:, :num_detectors]

    return variable_data


def get_pedestals(data):
    data_event_concat = np.concatenate(data, axis=2)
    strip_means = np.mean(data_event_concat, axis=2)
    strip_medians = np.median(data_event_concat, axis=2)

    return strip_medians


def plot_combined_time_series(data):
    n_events, n_samples_per_event = data.shape[0], data.shape[-1]
    for det_num, det in enumerate(np.concatenate(data, axis=2)):
        fig, ax = plt.subplots()
        for strip in det:
            ax.plot(range(len(strip)), strip)

        # Set plot labels and title
        for event_i in range(n_events):
            ax.axvline(event_i * n_samples_per_event, color='black', ls='--')
        ax.set_xlabel('Sample Number')
        ax.set_ylabel("ADC")
        ax.set_title(f"Detector #{det_num}")


def subtract_pedestal(data, pedestal):
    return data - pedestal[np.newaxis, :, :, np.newaxis]


def get_sample_max(data):
    return np.max(data, axis=-1)


def plot_position_data(data):
    print(data.shape)
    for event_num, event in enumerate(data):
        fig, ax = plt.subplots()
        for det_num, det in enumerate(event):
            ax.plot(range(len(det)), det, label=f'Detector #{det_num}')
        ax.set_xlabel('Position')
        ax.set_ylabel("Max ADC")
        ax.set_title(f"Event #{event_num}")
        ax.legend()
    data_transpose = np.transpose(data, (1, 0, 2))
    print(data_transpose.shape)
    for det_num, det in enumerate(data_transpose):
        fig, ax = plt.subplots()
        for event_num, event in enumerate(det):
            ax.plot(range(len(event)), event, label=f'Event #{event_num}')
        ax.set_xlabel('Position')
        ax.set_ylabel("Max ADC")
        ax.set_title(f"Detector #{det_num}")
        ax.legend()


if __name__ == '__main__':
    main()
