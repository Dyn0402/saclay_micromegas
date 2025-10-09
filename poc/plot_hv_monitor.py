#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 02 11:08 PM 2025
Created in PyCharm
Created as saclay_micromegas/plot_hv_monitor.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # file_path = 'F:/Saclay/cosmic_data/rd542_plein_3_first_test_10-2-25/long_test_450V/hv_monitor.csv'
    # file_path = 'F:/Saclay/cosmic_data/rd542_strip_2_first_test_10-27-25/long_test_450V/hv_monitor.csv'
    # file_path = '/local/home/dn277127/Bureau/cosmic_data/rd542_plein_3_first_test_10-2-25/long_test_450V/hv_monitor.csv'
    file_path = 'F:/Saclay/cosmic_data/rd542_strip_2_co2_10-8-25/long_test_460V/hv_monitor.csv'
    plot_hv_monitor(file_path)
    plt.show()
    print('donzo')


def plot_hv_monitor(file_path):
    # Load CSV
    df = pd.read_csv(file_path, parse_dates=["timestamp"])

    # Extract all card:slot identifiers
    card_slots = sorted(set(col.split()[0] for col in df.columns if ":" in col))

    # --- Voltage plot (v0 and vmon for each channel) ---
    fig_v, ax_v = plt.subplots(figsize=(12, 6))
    for cs in card_slots:
        # ax_v.plot(df["timestamp"], df[f"{cs} v0"], linestyle="--", label=f"{cs} v0 (set)")
        ax_v.plot(df["timestamp"], df[f"{cs} vmon"], label=f"{cs} vmon (measured)")
    ax_v.set_xlabel("Time")
    ax_v.set_ylabel("Voltage [V]")
    ax_v.set_title("HV vs Time (all cards/slots)")
    ax_v.legend(ncol=3, fontsize="small")
    ax_v.grid(True)

    # --- Current plot (imon for each channel) ---
    fig_i, ax_i = plt.subplots(figsize=(12, 6))
    for cs in card_slots:
        ax_i.plot(df["timestamp"], df[f"{cs} imon"], label=f"{cs} imon")
    ax_i.set_xlabel("Time")
    ax_i.set_ylabel("Current [µA]")
    ax_i.set_title("Current vs Time (all cards/slots)")
    ax_i.legend(ncol=3, fontsize="small")
    ax_i.grid(True)


if __name__ == '__main__':
    main()
