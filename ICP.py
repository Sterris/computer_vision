#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 00:04:51 2020

@author: kuba
"""
import numpy as np
import random
import matplotlib.pyplot as plt
from optimal_transform import optimal_transform 
from nearest_neighbor import nearest_neighbor, nearest_neighbor_2
from pyntcloud import PyntCloud
from sklearn.decomposition import PCA
import math

def icp(src, dst, maxIteration=500, tolerance=0.01, controlPoints=10000):
    r1 = random.random()*2*math.pi
    r2 = random.random()*2*math.pi
    r3 = random.random()*2*math.pi
    
    
    #random rotation for testing
    rx = np.array([[ 1, 0,  0],
               [ 0,  math.cos(r1),  -math.sin(r1)],
               [ 0,  math.sin(r1),  math.cos(r1)]])
    ry = np.array([[  math.cos(r2), 0,   math.sin(r2)],
               [ 0,  1,  0],
               [ -math.sin(r2),  0,  math.cos(r2)]])
    
    rz = np.array([[ math.cos(r3), -math.sin(r3),  0],
               [ math.sin(r3),  math.cos(r3),  0],
               [ 0,  0,  1]])
    r =np.dot(rx, np.dot(ry,rz))
    
    
    
    A = np.array(PyntCloud.from_file(src).points)[:,0:3]
    original = A
    B = np.array(PyntCloud.from_file(dst).points)[:,0:3]
    
    #applying random rotation
    A = np.dot(r, A.T).T + 200
    
    
    #plotting the initial positions
    plt.figure()
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='r',s=0.05)
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='g',s=0.05)
    plt.savefig("plots/"+src.split(".")[0]+"_0")
    plt.pause(0.5)
    
    
    #initial rotation based on the principal vectors
    pca = PCA(n_components=2)
    pca.fit(A)
    v1 = np.array(pca.components_[0])
    a = v1/np.linalg.norm(v1)
    pca = PCA(n_components=2)
    pca.fit(B)
    v2 = np.array(pca.components_[0])
    b = v2/np.linalg.norm(v2)
    n = np.cross(a,b)
    
    I = np.array([[1 ,0 ,0],[ 0,1,0],[0,0,1]])
    nc =np.array( [[ 0,-n[2] ,n[1]],[ n[2],0,-n[0]],[-n[1],n[0],0]])
    initR = I + nc + np.divide(np.matmul(nc,nc),(1+np.dot(a,b)))
    
    A = np.dot(initR, A.T).T
    T = np.transpose( np.mean(B, axis = 0) - np.mean(A, axis = 0))
    A = A + np.array([T for j in range(A.shape[0])])
 

    
    lastErr = np.inf
    rotcounter = 0
    length = min(A.shape[0], B.shape[0],controlPoints)
    
    # sampling random indeces for points in the point clouds
    sampleA = random.sample(range(A.shape[0]), length)
    sampleB = random.sample(range(B.shape[0]), length)
    
    # extracting the random points
    P = np.array([A[i] for i in sampleA])
    Q = np.array([B[i] for i in sampleB])

    for i in range(maxIteration):
        print("Iteration : " + str(i) + " with Err : " + str(lastErr))
        dis, index = nearest_neighbor_2(P, Q)
        R, T = optimal_transform(P, Q[index,:])
        
        # applying the step rotation and translation found
        A = np.dot(R, A.T).T + np.array([T for j in range(A.shape[0])])
        P = np.dot(R, P.T).T + np.array([T for j in range(P.shape[0])])
        

        meanErr = np.sum(dis) / dis.shape[0]
        print(lastErr - meanErr)
        if abs(lastErr - meanErr) < tolerance:
            #Local minima problem
            if lastErr > 5:
                
                #PCA analysis
                pca = PCA(n_components=3)
                pca.fit(A)
                v1 = np.array(pca.components_[0])
                v2 = np.array(pca.components_[1])
                v3 = np.array(pca.components_[2])

                
                #Using principal vectors as new basis
                tonew = np.transpose(np.array([v2,v1,v3]))
                back = np.linalg.inv(tonew)
                
                #Rotation in the new basis system
                R2 = [[1 ,0 ,0],[ 0,-1,0],[0,0,-1]]
                R3 = [[-1 ,0 ,0],[ 0,-1,0],[0,0,1]]
                
                if rotcounter == 1:
                    Rot = R2
                else:
                    Rot = R3
                
                A = np.transpose(np.dot(back,np.dot(Rot,np.dot(tonew,np.transpose(A)))))
                P = np.transpose(np.dot(back,np.dot(Rot,np.dot(tonew,np.transpose(P)))))
                
                
                #Translation
                T = np.transpose( np.mean(B, axis = 0) - np.mean(A, axis = 0))
                A = A + np.array([T for j in range(A.shape[0])])
                P = P + np.array([T for j in range(P.shape[0])])
                rotcounter = rotcounter + 1
                if rotcounter > 3:
                    print("Face not matched")
                    break
            else:
                print("Face matched")
                break
        lastErr = meanErr
        
        
        
        # visualization
        ax = plt.subplot(1, 1, 1, projection='3d')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='r',s=0.05)
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c='g',s=0.05)
        plt.savefig("plots/"+src.split(".")[0]+"_"+str(i+1))
        plt.pause(0.5)
    plt.show(block = False)
        
    # backtracking the overall transformation
    R, T = optimal_transform(original,A)
    
    return R, T, A
