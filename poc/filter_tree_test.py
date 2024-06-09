#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 07 11:54 PM 2024
Created in PyCharm
Created as saclay_micromegas/filter_tree_test.py

@author: Dylan Neff, Dylan
"""

import numpy as np

import uproot
import awkward as ak

import ROOT


def main():
    in_path = 'C:/Users/Dylan/Desktop/banco_test3/CosTb_HV7_datrun_240514_23H53_000_rays.root'
    out_path = 'C:/Users/Dylan/Desktop/banco_test3/CosTb_HV7_datrun_240514_23H53_000_rays_filtered.root'
    events = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # filter_dream_file_uproot(in_path, events, out_path)
    filter_dream_file_pyroot(in_path, events, out_path)
    print('donzo')


def filter_dream_file_uproot(file_path, events, out_file_path, event_branch_name='evn'):
    """
    Filter a dream file based on a mask.
    :param file_path: Path to dream file to filter
    :param events: List of events to keep
    :param out_file_path: Path to write filtered dream file
    :param event_branch_name: Name of event branch in dream file
    :return:
    """
    with uproot.open(file_path) as file:
        tree_name = f"{file.keys()[0].split(';')[0]};{max([int(key.split(';')[-1]) for key in file.keys()])}"
        tree = file[tree_name]
        print(tree.arrays())
        print(tree.arrays()[0])
        print(tree.arrays()[0]['X_Up'])
        # data = tree
        data = tree.arrays(library='ak')
        print(data)
        tree_events = ak.to_numpy(data[event_branch_name])
        mask = np.isin(events, tree_events)
        data = data[mask]
        print(data)
        # for key in data.keys():
        #     data[key] = data[key][mask]
        with uproot.recreate(out_file_path) as out_file:
            out_file[tree_name] = data


def filter_dream_file_pyroot(file_path, events, out_file_path, event_branch_name='evn'):
    """
    Filter a dream file based on a mask.
    :param file_path: Path to dream file to filter
    :param events: List of events to keep
    :param out_file_path: Path to write filtered dream file
    :param event_branch_name: Name of event branch in dream file
    :return:
    """
    in_file = ROOT.TFile.Open(file_path, 'READ')
    in_tree_name = [x.ReadObj().GetName() for x in in_file.GetListOfKeys()][0]
    in_tree = in_file.Get(in_tree_name)

    out_file = ROOT.TFile(out_file_path, 'RECREATE')
    out_tree = in_tree.CloneTree(0)

    event_num_branch = in_tree.GetBranch(event_branch_name)

    for i in range(in_tree.GetEntries()):
        in_tree.GetEntry(i)
        event_id = event_num_branch.GetLeaf(event_branch_name).GetValue()
        if event_id in events:
            out_tree.Fill()

    out_file.Write()
    out_file.Close()
    in_file.Close()


if __name__ == '__main__':
    main()
