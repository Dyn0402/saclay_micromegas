#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 24 9:25 PM 2024
Created in PyCharm
Created as saclay_micromegas/filter_banco.py

@author: Dylan Neff, Dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import ROOT


def main():
    triggers_path = 'F:/Saclay/cosmic_data/sg1_stats_7-26-24/max_hv_long_1/banco_data/banco_triggers.txt'
    banco_dir = 'F:/Saclay/cosmic_data/sg1_stats_7-26-24/max_hv_long_1/banco_data'

    triggers = read_triggers(triggers_path)
    for file in os.listdir(banco_dir):
        if file.endswith('.root') and '_filtered' not in file:
            input_file = os.path.join(banco_dir, file)
            output_file = os.path.join(banco_dir, file.replace('.root', '_filtered.root'))
            filter_tree(input_file, output_file, triggers)
            print(f'Filtered {input_file} to {output_file}')

    print('donzo')


def read_triggers(file_path):
    with open(file_path, 'r') as file:
        triggers = [int(line.strip()) for line in file]
    return triggers


def filter_tree(input_file, output_file, trigger_list):
    # Open the input ROOT file and retrieve the TTree
    input_root = ROOT.TFile.Open(input_file, "READ")

    # Get the list of keys in the file and find the first TTree
    keys = input_root.GetListOfKeys()
    tree_name = None

    for key in keys:
        obj = key.ReadObj()
        if isinstance(obj, ROOT.TTree):  # Check if the object is a TTree
            tree_name = key.GetName()
            break

    if tree_name is None:
        print("No TTree found in the file.")
        return

    # Retrieve the TTree
    pix_tree = input_root.Get(tree_name)

    # Create the output ROOT file and a new TTree for filtered events
    output_root = ROOT.TFile(output_file, "RECREATE")
    filtered_tree = pix_tree.CloneTree(0)  # Create an empty tree structure

    # Loop over the input tree and filter events based on trgNum
    trigger_list.sort()
    trigger_idx = 0
    num_triggers = len(trigger_list)
    entries = pix_tree.GetEntries()

    for entry in range(entries):
        pix_tree.GetEntry(entry)

        # Get trg_num from the TTree
        trg_num = getattr(pix_tree, 'fData.trgNum')
        if trg_num % 10000 == 0:
            print(f'Entry: {entry}/{entries}, Trigger: {trg_num}')

        # Move the pointer in the trigger_list until we either find a match or surpass trg_num
        while trigger_idx < num_triggers and trigger_list[trigger_idx] < trg_num:
            trigger_idx += 1

        # If we find a match, fill the filtered tree
        if trigger_idx < num_triggers and trigger_list[trigger_idx] == trg_num:
            filtered_tree.Fill()

        # Exit early if we've passed all the possible matches
        if trigger_idx >= num_triggers:
            break

    # Write the filtered tree to the output file
    output_root.cd()
    filtered_tree.Write()

    # Close the files
    input_root.Close()
    output_root.Close()

if __name__ == '__main__':
    main()
