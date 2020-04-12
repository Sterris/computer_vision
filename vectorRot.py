#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:59:57 2020

@author: kuba

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = [2,7,9]
b = [-1,12,5]

a = a/np.linalg.norm(a)
b = b/np.linalg.norm(b)


n = np.cross(a,b)
I = np.array([[1 ,0 ,0],[ 0,1,0],[0,0,1]])
nc =np.array( [[ 0,-n[2] ,n[1]],[ n[2],0,-n[0]],[-n[1],n[0],0]])
initR = I + nc + np.divide(np.matmul(nc,nc),(1+np.dot(a,b)))

c = np.dot(initR,a)/np.linalg.norm(np.dot(initR,a))

print("a: ",a)
print("b: ",b)
print("c: ",c)


"""
origin = [0], [0], [0]
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.quiver(a[0],a[1],a[2],1,1,1, length=0.1, normalize=True)
ax.quiver(b[0],b[1],b[2],1,1,1, length=0.1, normalize=True)
ax.quiver(c[0],c[1],c[2],1,1,1, length=0.1, normalize=True)
plt.show()


x = np.linspace(0,a[0]*100,1000)
y = np.linspace(0,a[1]*100,1000)
z = np.linspace(0,a[2]*100,1000)


x2 = np.linspace(b[0]*100,0,1000)
y2 = np.linspace(0,b[1]*100,1000)
z2 = np.linspace(0,b[2]*100,1000)

x3 = np.linspace(c[0]*100,0,1000)
y3 = np.linspace(0,c[1]*100,1000)
z3 = np.linspace(0,c[2]*100,1000)


ax = plt.subplot(1, 1, 1, projection='3d')
ax.scatter(x,y,z,c='b')
ax.scatter(x2,y2,z2,c='y')
ax.scatter(x3,y3,z3,c='g')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
plt.pause(0.5)
plt.show(block = False)
"""