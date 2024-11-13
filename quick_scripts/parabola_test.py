#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 13 12:00 AM 2024
Created in PyCharm
Created as saclay_micromegas/parabola_test.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    # Convert all inputs to float to ensure floating-point division
    x1, y1, x2, y2, x3, y3 = map(float, (x1, y1, x2, y2, x3, y3))

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)

    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom

    B = (x3 ** 2 * (y1 - y2) + x2 ** 2 * (y3 - y1) + x1 ** 2 * (y2 - y3)) / denom

    C1 = x2 * x3 * (x2 - x3) * y1
    C2 = x3 * x1 * (x3 - x1) * y2
    C3 = x1 * x2 * (x1 - x2) * y3
    C4 = C1 + C2 + C3

    C = C4 / denom

    xv = -B / (2 * A)
    yv = C - B ** 2 / (4 * A)

    if yv <= 0:
        print(f"Warning: denom={denom} a={A} b={B} c={C} c1={C1} c2={C2} c3={C3} c4={C4}")

    return xv, yv

# Test the function with some example points
x1, y1 = 0, 0
x2, y2 = 1, 2
x3, y3 = 2, 1.5

# Calculate the vertex
xv, yv = calc_parabola_vertex(x1, y1, x2, y2, x3, y3)

# Calculate parabola coefficients
denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
b = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

# Generate points for plotting the parabola
x = np.linspace(min(x1, x2, x3) - 0.5, max(x1, x2, x3) + 0.5, 100)
y = a * x**2 + b * x + c

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='Parabola')
plt.plot([x1, x2, x3], [y1, y2, y3], 'ro', label='Input Points')
plt.plot(xv, yv, 'g*', markersize=15, label='Vertex')

plt.annotate(f'({xv:.2f}, {yv:.2f})', (xv, yv), xytext=(5, 5), textcoords='offset points')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Parabola and its Vertex')
plt.legend()
plt.grid(True)

plt.show()

print(f"Vertex coordinates: ({xv:.4f}, {yv:.4f})")
