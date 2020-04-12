#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:40:05 2020

@author: kuba
"""

import matplotlib.pyplot as plt
import numpy as np
from pyntcloud import PyntCloud
from sklearn.decomposition import PCA
import random

src = "andreadm2.ply"
A = np.array(PyntCloud.from_file(src).points)[:,0:3]
sampleA = random.sample(range(A.shape[0]), 10000)
A= np.array([A[i] for i in sampleA])
r = np.array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
               [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
A = np.transpose( np.dot(r,np.transpose(A)))
pca = PCA(n_components=2)
pca.fit(A)


"""
x = np.linspace(-0.3244755*100,0,1000) + np.mean(A, axis = 0)[0]
y = np.linspace(0,0.90055925*100,1000)+ np.mean(A, axis = 0)[1]
z = np.linspace(0,0.28932402*100,1000)+ np.mean(A, axis = 0)[2]


x2 = np.linspace(-0.94195*100,0,1000) + np.mean(A, axis = 0)[0]
y2 = np.linspace(-0.33553877*100,0,1000)+ np.mean(A, axis = 0)[1]
z2 = np.linspace(-0.01198535*100,0,1000)+ np.mean(A, axis = 0)[2]
"""



v1 = np.array(pca.components_[0])
v2 = np.array(pca.components_[1])
v3 = np.cross(v1,v2)

tonew = np.transpose(np.array([v2,v1,v3]))

back = np.linalg.inv(tonew)
R3 = [[-1 ,0 ,0],[ 0,-1,0],[0,0,1]]
R2 = [[1 ,0 ,0],[ 0,-1,0],[0,0,-1]]
rotated3 = np.transpose(np.dot(back,np.dot(R3,np.dot(tonew,np.transpose(A)))))
rotated2 = np.transpose(np.dot(back,np.dot(R2,np.dot(tonew,np.transpose(A)))))

pca = PCA(n_components=2)
pca.fit(rotated2)
v2 = np.array(pca.components_[0])
print("A: ",v1)
print("rot2: ",v2)
a = v1/np.linalg.norm(v1)
b = v2/np.linalg.norm(v2)
n = np.cross(a,b)
I = np.array([[1 ,0 ,0],[ 0,1,0],[0,0,1]])
nc =np.array( [[ 0,-n[2] ,n[1]],[ n[2],0,-n[0]],[-n[1],n[0],0]])
initR = I + nc + np.divide(np.matmul(nc,nc),(1+np.dot(a,b)))

rotated2_2 = np.dot(initR, A.T).T

pca = PCA(n_components=2)
pca.fit(rotated2_2)
print("rot2_2",np.array(pca.components_[0]))
ax = plt.subplot(1, 1, 1, projection='3d')
ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='r',s=0.05)
#ax.scatter(rotated3[:, 0], rotated3[:, 1], rotated3[:, 2], c='g',s=0.05)
ax.scatter(rotated2[:, 0], rotated2[:, 1], rotated2[:, 2], c='b',s=0.05)
ax.scatter(rotated2_2[:, 0], rotated2_2[:, 1], rotated2_2[:, 2], c='g',s=0.05)
ax.scatter(x,y,z,c='b')
ax.scatter(x2,y2,z2,c='y')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
plt.pause(0.5)
plt.show(block = False)