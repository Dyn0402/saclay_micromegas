#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 04 12:41 AM 2024
Created in PyCharm
Created as saclay_micromegas/dream_functions.py

@author: Dylan Neff, Dylan
"""

import numpy as np
from scipy.optimize import curve_fit as cf

import uproot
import awkward as ak
import vector


def read_det_data(file_path, num_detectors=None, variable_name='amplitude', tree_name='T'):
    # Open the ROOT file with uproot
    root_file = uproot.open(file_path)

    # Access the tree in the file
    tree = root_file[tree_name]

    # Get the variable data from the tree
    variable_data = ak.to_numpy(tree[variable_name].array())
    root_file.close()

    if num_detectors is not None:
        if isinstance(num_detectors, list) and len(num_detectors) == 2:
            variable_data = variable_data[:, num_detectors[0]:num_detectors[-1]]
        elif isinstance(num_detectors, int):
            variable_data = variable_data[:, :num_detectors]

    return variable_data


def get_pedestals(data):
    strip_medians = np.median(data, axis=3)  # Median of samples in each strip for each event
    strip_medians_transpose = np.transpose(strip_medians, axes=(1, 2, 0))  # Concatenate strips for each event
    strip_means = np.mean(strip_medians_transpose, axis=2)  # Mean of strip medians over all events

    # This takes the median of samples over all events for each strip
    # data_event_concat = np.concatenate(data, axis=2)
    # strip_means = np.mean(data_event_concat, axis=2)
    # strip_medians = np.median(data_event_concat, axis=2)

    return strip_means


def subtract_pedestal(data, pedestal):
    return data - pedestal[np.newaxis, :, :, np.newaxis]


def get_pedestals_rms(ped_data, ped_means):
    ped_zeroed = subtract_pedestal(ped_data, ped_means)  # Subtract averages from pedestal data
    ped_zeroed_concat = np.concatenate(ped_zeroed, axis=2)  # Concatenate all events
    ped_rms = np.std(ped_zeroed_concat, axis=2)  # Get RMS of pedestal data for each strip
    # Existing code averages over all strips, but will see if we can get away with strip by strip
    return ped_rms


def get_pedestal_fits(ped_data, common_noise=None):
    if common_noise is not None:
        ped_data = ped_data - common_noise[:, :, np.newaxis, :]
    ped_concat = np.concatenate(ped_data, axis=2)  # Concatenate all events
    ped_fits = np.apply_along_axis(fit_pedestals, -1, ped_concat)
    ped_fits = np.transpose(ped_fits, axes=(2, 0, 1))
    ped_fits = dict(zip(['mean', 'sigma', 'mean_err', 'sigma_err'], ped_fits))
    # Existing code averages over all strips, but will see if we can get away with strip by strip
    return ped_fits


def fit_pedestals(strip_samples):
    bin_edges = np.arange(-0.5, 4097.5, 1)
    mean = np.mean(strip_samples)
    sd = np.std(strip_samples)
    hist, _ = np.histogram(strip_samples, bins=bin_edges, density=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    popt, pcov = cf(gaussian_density, bin_centers, hist, p0=[mean, sd])
    perr = np.sqrt(np.diag(pcov))
    return *popt, *perr


def get_common_noise(data, pedestals):
    data_ped_sub = data - pedestals[np.newaxis, :, :, np.newaxis]
    data_ped_sub_trans = np.transpose(data_ped_sub, axes=(0, 1, 3, 2))
    common_noise = np.nanmedian(data_ped_sub_trans, axis=-1)

    return common_noise


def get_noise_thresholds(ped_rms, noise_sigmas=5):
    return noise_sigmas * ped_rms


def get_sample_max(data):
    return np.max(data, axis=-1)


def gaussian_density(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))