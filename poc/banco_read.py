#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 03 12:03 PM 2024
Created in PyCharm
Created as saclay_micromegas/banco_read.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak
import vector


def main():
    vector.register_awkward()
    file_path = 'C:/Users/Dylan/Desktop/banco_test2/fdec.root'
    with uproot.open(file_path) as file:
        print(file.keys())
        tree = file['events;1']
        print(tree.keys())
        data = tree.arrays()
        print(data)
        print(len(data))
        print(data[0])
        print(data[0].fields)
        for field in data[0].fields:
            print(f'Field {field}: {data[0][field]}')
        clust_rows, clust_cols, clust_sizes = [], [], []
        for event in data:
            print(f'Event {event["eventId"]} has {len(event["clusters.size"])} clusters of size {event["clusters.size"]}')
            # if len(event['clusters.size']) == 0:
            #     continue
            event_clust_rows = list(event['clusters.rowCentroid'])
            event_clust_cols = list(event['clusters.colCentroid'])
            event_clust_sizes = list(event['clusters.size'])
            bad_i = None
            for i in range(len(event_clust_rows)):
                if event_clust_rows[i] == 102 and event_clust_cols[i] == 3739.5:
                    bad_i = i
            if bad_i is not None:
                event_clust_rows.pop(bad_i)
                event_clust_cols.pop(bad_i)
                event_clust_sizes.pop(bad_i)
            if len(event_clust_rows) == 0:
                continue
            print(f'Event {event["eventId"]} has {len(event_clust_sizes)} clusters of size {event_clust_sizes}')
            clust_rows.append(event_clust_rows[0])
            clust_cols.append(event_clust_cols[0])
            clust_sizes.append(event_clust_sizes[0])
        print(clust_rows)
        print(clust_cols)
        print(clust_sizes)
        print(len(clust_sizes))
        # Histogram of cluster sizes
        fig, ax = plt.subplots()
        plt.hist(clust_sizes)
        plt.title('Cluster Size Histogram')
        plt.xlabel('Cluster Size')
        plt.ylabel('Counts')
        fig.tight_layout()
        # Histogram of cluster rows
        fig, ax = plt.subplots()
        plt.hist(clust_rows)
        plt.title('Cluster Row Histogram')
        plt.xlabel('Cluster Row')
        plt.ylabel('Counts')
        fig.tight_layout()
        # Histogram of cluster cols
        fig, ax = plt.subplots()
        plt.hist(clust_cols)
        plt.title('Cluster Col Histogram')
        plt.xlabel('Cluster Col')
        plt.ylabel('Counts')
        fig.tight_layout()

        plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
