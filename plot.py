# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:11:38 2022

@author: Carlos de Cloet
"""
import matplotlib.pyplot as plt


PC = [0.07, 0.07, 0.11, 0.39, 0.57, 0.87, 0.95 , 0.98, 1]
PQ = [0.055, 0.049, 0.045, 0.039, 0.025, 0.003, 0.00045, 0, 0]
F1 = [0.056, 0.055, 0.0644, 0.072, 0.047, 0.0069, 0.001, 0.001, 0]

FOC = [0.0005, 0.00051, 0.0007, 0.003, 0.007, 0.076, 0.26, 0.62, 1]

plt.plot(FOC,PC)
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Completeness')
plt.show()

plt.plot(FOC,PQ)
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Quality')
plt.show()

plt.plot(FOC,F1)
plt.xlabel('Fraction of Comparisons')
plt.ylabel('F1*-measure')
plt.show()