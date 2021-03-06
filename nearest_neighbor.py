#from plyfile import PlyData
import numpy as np
from sklearn.neighbors import NearestNeighbors

"""
#Loading point clouds from file
plydata = PlyData.read('test.ply')
x_points = plydata['vertex']['x']
y_points = plydata['vertex']['y']
z_points = plydata['vertex']['z']
p = np.array([x_points, y_points, z_points]).T

plydata = PlyData.read('test2.ply')
x_points = plydata['vertex']['x']
y_points = plydata['vertex']['y']
z_points = plydata['vertex']['z']

q = np.array([x_points, y_points, z_points]).T


##Could be bad if points are sorted by position. Could maybe get away with shuffling array.
def sample(point_cloud, n):# n sample size
    return point_cloud[0:n, :]

"""

#Comparing each point in one point cloud to all in other
def nearest_neighbor(P, Q):
    dis = np.zeros(P.shape[0]) # Dist is array of distances
    index = np.zeros(Q.shape[0], dtype = np.int)
    for i in range(P.shape[0]):
        minDis = np.inf #Initially minimum distance = infinity
        for j in range(Q.shape[0]):
            tmp = np.linalg.norm(P[i] - Q[j], ord = 2) #Fast way to calculate distance
            if minDis > tmp: 
                minDis = tmp
                index[i] = j
        dis[i] = minDis
    return dis, index
# returns the minimal distances for all P points to the Q points and the indeces of Q points


def nearest_neighbor_2(P,Q):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    nbrs.fit(Q)
    distances, indices = nbrs.kneighbors(P, return_distance=True)
    return distances.ravel(), indices.ravel()





"""

q =sample(q,50)
p =sample(p,50)
dis, index = nearest_neighbor(p,q)
print(np.mean(dis))
"""