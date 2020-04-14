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

#filenames ordering for the correct gif order
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#matrix format 
    #x1 y1 z1
    #x2 y2 z2
    


start_time = time.time()
R, T, A = icp("andreadm2.ply","andreadm2.ply")
print("--- %s seconds ---" % (time.time() - start_time))


#creates gif
numbers = re.compile(r'(\d+)')
with imageio.get_writer('gifs/andreadm3.gif', mode='I',duration = 0.5) as writer:
    files = sorted(os.listdir('plots'), key= numericalSort)
    if files.count(".DS_Store")!=0:
        files.remove(".DS_Store")
    for filename in files:
        image = imageio.imread('plots/'+filename)
        writer.append_data(image)