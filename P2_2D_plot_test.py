#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 22 5:10 PM 2023
Created in PyCharm
Created as saclay_micromegas/P2_2D_plot_test.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Create two sample arrays
    array1 = np.random.rand(10, 5)
    array2 = np.random.rand(8, 4)

    # Determine the common vmin and vmax for both arrays
    vmin = min(np.min(array1), np.min(array2))
    vmax = max(np.max(array1), np.max(array2))

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the first array in the upper part of the figure
    # im1 = ax.imshow(array1, cmap='plasma', extent=[0, 5, 8, 18], vmin=vmin, vmax=vmax)
    im1 = ax.imshow(array1.transpose(), cmap='plasma', extent=[0, 10, 5, 10], vmin=vmin, vmax=vmax)
    ax.set_title('Array 1')

    # Plot the second array in the lower part of the figure with adjusted extent and aspect
    # im2 = ax.imshow(array2, cmap='plasma', extent=[0, 5, 0, 8], vmin=vmin, vmax=vmax, aspect='auto')
    im2 = ax.imshow(array2.transpose(), cmap='plasma', extent=[0, 10, 0, 5], vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title('Array 2')

    # Adjust the aspect ratio and limits to make it look like a single image
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Create a single colorbar for both plots
    cbar = fig.colorbar(im1, ax=ax, orientation='vertical', pad=0.1)
    cbar.set_label('Value')

    # Create a simple 2D array
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

    fig, ax = plt.subplots()

    plt.imshow(data, origin='lower')  # The default is origin='lower'
    plt.colorbar()
    plt.title('imshow with origin="lower"')

    fig, ax = plt.subplots()

    plt.imshow(data, origin='upper')
    plt.colorbar()
    plt.title('imshow with origin="upper"')

    large_pixels = np.arange(32)
    large_pixels = large_pixels.reshape(8, 4)
    print(large_pixels)
    large_pixels = large_pixels.transpose()
    print(large_pixels)
    small_pixels = np.arange(50).reshape(5, 10)
    print(small_pixels)
    print(f'large_pixels: {large_pixels}')
    print(f'small_pixels: {small_pixels}')

    vmin, vmax = 0, 50

    fig, ax = plt.subplots()
    im_small = ax.imshow(small_pixels, cmap='plasma', extent=[0, 10, 5, 10], vmin=vmin, vmax=vmax, origin='upper')
    im_large = ax.imshow(large_pixels, cmap='plasma', extent=[0, 10, 0, 5], vmin=vmin, vmax=vmax, origin='upper')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('Origin: upper')

    # Annotate each pixel with its value
    for pixel_set, y_offset, pos_scale in zip([large_pixels, small_pixels], [0, 5], [5. / 4, 1]):
        for i in range(pixel_set.shape[0]):
            for j in range(pixel_set.shape[1]):
                plt.annotate(f'{pixel_set[i, j]}', np.array((j + 0.5, i + y_offset + 0.5)) * pos_scale,
                             color='white', ha='center', va='center')

    # Create a single colorbar for both plots
    cbar = fig.colorbar(im_small, ax=ax, orientation='vertical', pad=0.1)
    cbar.set_label('Value')

    fig, ax = plt.subplots()
    im_small = ax.imshow(small_pixels, cmap='plasma', extent=[0, 10, 5, 10], vmin=vmin, vmax=vmax, origin='lower')
    im_large = ax.imshow(large_pixels, cmap='plasma', extent=[0, 10, 0, 5], vmin=vmin, vmax=vmax, origin='lower')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('Origin: lower')

    # Annotate each pixel with its value
    for pixel_set, y_offset, pos_scale in zip([large_pixels, small_pixels], [0, 5], [5. / 4, 1]):
        for i in range(pixel_set.shape[0]):
            for j in range(pixel_set.shape[1]):
                plt.annotate(f'{pixel_set[i, j]}', np.array((j + 0.5, i + y_offset + 0.5)) * pos_scale,
                             color='white', ha='center', va='center')

    # Create a single colorbar for both plots
    cbar = fig.colorbar(im_small, ax=ax, orientation='vertical', pad=0.1)
    cbar.set_label('Value')

    fig, ax = plt.subplots()
    im_small = ax.imshow(small_pixels, cmap='plasma', extent=[0, 10, 5, 10], vmin=vmin, vmax=vmax)
    im_large = ax.imshow(large_pixels, cmap='plasma', extent=[0, 10, 0, 5], vmin=vmin, vmax=vmax)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title('Origin: default')

    # Annotate each pixel with its value
    for pixel_set, y_offset, pos_scale in zip([large_pixels, small_pixels], [0, 5], [5. / 4, 1]):
        for i in range(pixel_set.shape[0]):
            for j in range(pixel_set.shape[1]):
                plt.annotate(f'{pixel_set[i, j]}', np.array((j + 0.5, i + y_offset + 0.5)) * pos_scale,
                             color='white', ha='center', va='center')

    # Create a single colorbar for both plots
    cbar = fig.colorbar(im_small, ax=ax, orientation='vertical', pad=0.1)
    cbar.set_label('Value')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()
