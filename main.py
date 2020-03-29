#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:38:49 2020

@author: kuba
"""
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from optimal_transform import optimal_transform 
from nearest_neighbor import nearest_neighbor, nearest_neighbor_2
from pyntcloud import PyntCloud
import sys
import time




#matrix format 
    #x1 y1 z1
    #x2 y2 z2
    

def icp(src, dst, maxIteration=500, tolerance=0.01, controlPoints=40000):
    r = np.array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
               [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    
    A = np.array(PyntCloud.from_file(src).points)[:,0:3]
    #np.savetxt('starting_matrix.txt',A,fmt='%.2f')
    B = np.array(PyntCloud.from_file(dst).points)[:,0:3]
    B = np.dot(r, B.T).T + 200
    
    
    lastErr = np.inf
    
    length = min(A.shape[0], B.shape[0],controlPoints)
    
    # sampling random indeces for points in the point clouds
    sampleA = random.sample(range(A.shape[0]), length)
    sampleB = random.sample(range(B.shape[0]), length)
    
    # extracting the random points
    P = np.array([A[i] for i in sampleA])
    Q = np.array([B[i] for i in sampleB])
    
    """
    #point cloud for plotting - not needed with the new fast near neighbour
    sampleApl = random.sample(range(A.shape[0]), int(min(A.shape[0], B.shape[0])/3))
    sampleBpl = random.sample(range(B.shape[0]),int(min(A.shape[0], B.shape[0])/3))
    Ppl = np.array([A[i] for i in sampleApl])
    Qpl = np.array([B[i] for i in sampleBpl])
    """
    
    
    """  
    else:
        length = A.shape[0]
        if (length > controlPoints): 
            sampleA = random.sample(range(A.shape[0]), length)
            sampleB = random.sample(range(B.shape[0]), length)
            P = np.array([A[i] for i in sampleA])
            Q = np.array([B[i] for i in sampleB])
        else :
            P = A
            Q = B
    """
    plt.figure()
    for i in range(maxIteration):
        print("Iteration : " + str(i) + " with Err : " + str(lastErr))
        dis, index = nearest_neighbor_2(P, Q)
        #print("this:",Q)
        R, T = optimal_transform(P, Q[index,:])
        
        # applying the step rotation and translation found
        A = np.dot(R, A.T).T + np.array([T for j in range(A.shape[0])])
        
        P = np.dot(R, P.T).T + np.array([T for j in range(P.shape[0])])
        

        meanErr = np.sum(dis) / dis.shape[0]
        print(lastErr - meanErr)
        if abs(lastErr - meanErr) < tolerance:
            if lastErr > 5:
                R = [[-1 ,0 ,0],[ 0,-1,0],[0,0,1]]
                A = np.dot(R, A.T).T
                P = np.dot(R, P.T).T
            else:
                break
        lastErr = meanErr
        
        
        
        # visualization
        ax = plt.subplot(1, 1, 1, projection='3d')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='r',s=0.05)
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c='g',s=0.05)
        plt.pause(0.5)
    plt.show(block = False)
        
    # backtracking the overall transformation
    R, T = optimal_transform(A, np.array(A))
    
    return R, T, A




start_time = time.time()
R, T, A = icp("test.ply","test.ply")
#np.savetxt('result_matrix.txt',A,fmt='%.2f')
print("--- %s seconds ---" % (time.time() - start_time))
print(R)
