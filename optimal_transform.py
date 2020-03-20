import numpy as np

# susceptible to local minima
"""
A = np.array([[1,1,1],[1,0,0],[0,0,1],[0,1,0],[1,0,1],[0,0,0],[1,1,0],[0,1,1]]) 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

meanQ = np.mean(A, axis = 0)
print(A)
print("Q", meanQ)

Q_ = A - meanQ
print("Q2",Q_)
print("A0",Q_[:,0])

ax.scatter(A[:,0],A[:,1],A[:,2], c = 'r', marker='o')
ax.scatter(Q_[:,0],Q_[:,1],Q_[:,2], c = 'b', marker='o')

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')

"""

def optimal_transform(A, B):
    
    #matrix format 
    #x1 y1 z1
    #x2 y2 z2
    
    #finding the centroids
    meanA = np.mean(A, axis = 0)
    #print(B)
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
       
   #translation
    T = np.transpose(meanB) - np.dot(Rot, np.transpose(meanA))
    return Rot, T