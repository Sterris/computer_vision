import numpy as np

# susceptible to local minima

def optimal_transform(A, B):
    
    #matrix format 
    #x1 y1 z1
    #x2 y2 z2
    
    #finding the centroids
    meanA = np.mean(A, axis = 0)
    meanB = np.mean(B, axis = 0)
    
    #moving centroids to 0,0,0 origin
    Acen = A - meanA
    Bcen = B - meanB

    # rotation matrix
    W = np.dot(np.transpose(Bcen), Acen)
    U, S, VT = np.linalg.svd(W)
    Rot = np.dot(U, VT)
    
    #reflection case
    if np.linalg.det(Rot) < 0:
       Rot[2, :] = Rot[2, :]* (-1)

    T = np.transpose(meanB) - np.dot(Rot, np.transpose(meanA))
    return Rot, T