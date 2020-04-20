#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:38:49 2020

@author: kuba
"""
import time
import imageio
import os
import re
from ICP import icp 
import numpy as np
import random
import math
from pyntcloud import PyntCloud

#filenames ordering for the correct gif order
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


for file in os.listdir("face_clouds"):
    for file2 in os.listdir("face_clouds"):
        
        
        #matrix format 
            #x1 y1 z1
            #x2 y2 z2
            
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
        
        np.savetxt("rotation_matrices/"+file +"_"+ file2 + "_initial.txt", r, delimiter=',')
        
        A = np.array(PyntCloud.from_file("face_clouds/"+file).points)[:,0:3]
        B = np.array(PyntCloud.from_file("face_clouds/"+file2).points)[:,0:3]
        #noise
        noise = np.random.rand(A.shape[0],A.shape[1])*5
        #applying random rotation
        A = np.dot(r, A.T).T + 200 + noise
        
        start_time = time.time()
        R, T, error, i,cat = icp(A,B)
        print("--- %s seconds ---" % (time.time() - start_time))
        np.savetxt("rotation_matrices/"+file +"_"+ file2 + "_guess.txt", R, delimiter=',')
        np.savetxt("errors/"+file +"_"+ file2 + ".txt", np.array([float(i),error,float(cat)]), delimiter=',')
        
        #creates gif
        numbers = re.compile(r'(\d+)')
        with imageio.get_writer("gifs/"+ file +"_"+ file2 + ".gif", mode='I',duration = 0.5) as writer:
            files = sorted(os.listdir('plots'), key= numericalSort)
            if files.count(".DS_Store")!=0:
                files.remove(".DS_Store")
            for filename in files:
                image = imageio.imread('plots/'+filename)
                writer.append_data(image)
            for f in files:
                os.remove('plots/'+f)