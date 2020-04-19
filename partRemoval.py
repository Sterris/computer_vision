#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 00:15:22 2020

@author: kuba
"""

import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt

A = np.array(PyntCloud.from_file("andreadm2.ply").points)[:,0:3]

plt.figure()
ax = plt.subplot(1, 1, 1, projection='3d')
ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='r',s=0.05)


#matrix format 
    #x1 y1 z1
    #x2 y2 z2
B = A[A[:,0] > np.mean(A, axis = 0)[0]]


ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='g',s=0.05)