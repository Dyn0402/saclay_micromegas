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

# matplotlib.use('Qt5Agg')


def main():
    fdf_dir = 'test_data/fdf/'
    raw_root_dir = 'test_data/raw_root/'
    # files = ['selfTPOTFe_proba_pedthr_230801_17H17_000_05', 'selfTPOTFe_proba_datrun_230801_17H17_000_05']
    files = ['selfTPOTFe_proba_pedthr_230801_17H17_000_05']
    for file in files:
        raw_root_path = os.path.join(raw_root_dir, file + ".root")
        # raw_root_path = read_fdf_to_root(file, fdf_dir, raw_root_dir)
        # plot_adc_pyroot(raw_root_path)
        # plot_adc_uproot(raw_root_path)
        plot_time_samples(raw_root_path)
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


def plot_time_samples(file_path):
    # Open the ROOT file with uproot
    root_file = uproot.open(file_path)

    # Access the tree in the file
    tree = root_file['T']

    # Get the variable data from the tree
    variable_name = 'StripAmpl'
    variable_data = ak.to_numpy(tree[variable_name].array())
    print(variable_data.shape)

    for det_num, det in enumerate(np.concatenate(variable_data, axis=2)):
        print(det_num)
        print(det)
        fig, ax = plt.subplots()
        for strip in det:
            ax.plot(range(len(strip)), strip)

        # Set plot labels and title
        ax.set_xlabel('Sample Number')
        ax.set_ylabel("ADC")
        ax.set_title(f"Detector #{det_num}")

    # Display the plot
    plt.show()


if __name__ == '__main__':
    # pass
    main()
