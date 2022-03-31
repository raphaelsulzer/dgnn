import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import seaborn as sns
from matplotlib.colors import ListedColormap

fig = plt.figure(figsize=(4, 4), dpi=1000)
plt.tight_layout()
plt.xlim(-1, 15)
# plt.ylim(-0.5, 2.05)
plt.ylim(-1, 13)
plt.axis('off')

px = [0,1,2,3,4,5,9,10,10,11,11,14,14]
py = [6,12,2,8,4,2,8,4,12,0,10,2,4]
points=np.stack((px,py),axis=1)


### delaunay
tri = Delaunay(points)
# centers = np.sum(points[tri.simplices], axis=1)/3.0

# mask = np.zeros(len(tri.simplices))
# mask[4]=1
# mask[7]=1
# mask[9]=1

plt.triplot(points[:,0], points[:,1], color='0.75', linewidth=1.5)

# plt.scatter(points[:,0], points[:,1], color='r', marker='o', s=10, zorder=10)

plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/gnn.svg")

plt.show(block=True)