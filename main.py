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
import matplotlib
from optimal_transform import optimal_transform 
from nearest_neighbor import nearest_neighbor, nearest_neighbor_2
from pyntcloud import PyntCloud
import sys
from sklearn.decomposition import PCA
import imageio
import os
import re

#filenames ordering for the correct gif order
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#matrix format 
    #x1 y1 z1
    #x2 y2 z2
    

def icp(src, dst, maxIteration=20, tolerance=0.01, controlPoints=40000):
    r = np.array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
               [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    
    A = np.array(PyntCloud.from_file(src).points)[:,0:3]
    original = A
    B = np.array(PyntCloud.from_file(dst).points)[:,0:3]
    B = np.dot(r, B.T).T + 200
    
    
    #plotting
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
        R, T = optimal_transform(P, Q[index,:])
        
        # applying the step rotation and translation found
        A = np.dot(R, A.T).T + np.array([T for j in range(A.shape[0])])
        P = np.dot(R, P.T).T + np.array([T for j in range(P.shape[0])])
        

        meanErr = np.sum(dis) / dis.shape[0]
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
        errs[i] = meanErr

        
        # visualization
        ax = plt.subplot(1, 1, 1, projection='3d')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='r',s=0.05)
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c='g',s=0.05)
        plt.savefig("plots/"+src.split(".")[0]+"_"+str(i+1))
        plt.pause(0.5)
    plt.show(block = False)
        
    # backtracking the overall transformation
    R, T = optimal_transform(original,A)
    
    return R, T, A, errs


def plot(errs, time):
    np.trim_zeros(errs)
    itrs = np.linspace(1, len(errs), num=len(errs))
    plt.plot(itrs, errs)
    plt.xlabel('Iteration number')
    plt.xticks()
    plt.ylabel('Error')
    print("Total iterations: ", len(errs), "----- Total time: ", time, "----- Final error: ", errs[-1])
    #plt.text(0.5, 0.5, str(time))
    plt.show()


start_time = time.time()
R, T, A = icp("andreadm2.ply","andreadm2.ply")
print("--- %s seconds ---" % (time.time() - start_time))
time = time.time() - start_time
plot(errs, time)


#creates gif
numbers = re.compile(r'(\d+)')
with imageio.get_writer('gifs/andreadm2.gif', mode='I',duration = 0.5) as writer:
    files = sorted(os.listdir('plots'), key= numericalSort)
    if files.count(".DS_Store")!=0:
        files.remove(".DS_Store")
    for filename in files:
        image = imageio.imread('plots/'+filename)
        writer.append_data(image)
