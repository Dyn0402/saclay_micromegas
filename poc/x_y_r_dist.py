#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 21 10:44 PM 2024
Created in PyCharm
Created as saclay_micromegas/x_y_r_dist.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 1, 1000)
    r = np.sqrt(x**2 + y**2)
    d =
    fig, ax = plt.subplots()
    plt.hist(r, bins=50)
    plt.xlabel('Radius')
    plt.ylabel('Counts')
    plt.title('Distribution of Radii')
    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()
