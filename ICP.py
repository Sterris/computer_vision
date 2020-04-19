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
from sklearn.decomposition import PCA

def icp(src, dst, maxIteration=100, tolerance=0.01, controlPoints=10000):
    
    A = src
    B = dst
    #plotting the initial positions
    plt.figure()
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='r',s=0.05)
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='g',s=0.05)
    plt.savefig("plots/"+"plot_0")
    plt.pause(0.5)
    
    
    #initial rotation based on the principal vectors
    
    for i in range (0,2):
        pca = PCA(n_components=2)
        pca.fit(A)
        v1 = np.array(pca.components_[i])
        a = v1/np.linalg.norm(v1)
        pca = PCA(n_components=2)
        pca.fit(B)
        v2 = np.array(pca.components_[i])
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
            if lastErr > 13:
                
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
                    print("Not the same face, faces not matched")
                    break
                
            elif lastErr < 13 and lastErr > 4:
                print("Not the same face, faces  matched")
                break
                
            else:
                print("Same face, faces matched")
                break
        lastErr = meanErr
        
        
        
        # visualization
        ax = plt.subplot(1, 1, 1, projection='3d')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='r',s=0.05)
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c='g',s=0.05)
        plt.savefig("plots/"+"plot_"+str(i+1))
        plt.pause(0.5)
    plt.show(block = False)
        
    # backtracking the overall transformation
    R, T = optimal_transform(src,A)
    
    return R, T
