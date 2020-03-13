from plyfile import PlyData, PlyElement  


from pyntcloud import PyntCloud

cloud = PyntCloud.from_file("test.ply")

print(cloud)
cloud.plot()

