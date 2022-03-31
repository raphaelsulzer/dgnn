import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import seaborn as sns
from matplotlib.colors import ListedColormap



def plot_points_with_noise(px, py, nx, ny):
    fig = plt.figure(figsize=(4, 2), dpi=1000)
    plt.tight_layout()
    plt.xlim(-2.7, 2.7)
    # plt.ylim(-0.5, 2.05)
    plt.ylim(-0.10, 1.1)
    plt.axis('off')

    plt.plot(px, py, color='r', marker='o', markersize=13, linewidth=4.5)


    plt.scatter(px, py, color='r', marker='o', s=180, zorder=10)

    plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/points/noise.png")
    plt.close(fig)

### correct surface
plt.figure(figsize=(4,3),dpi=1000)
plt.tight_layout()
plt.xlim(-2.7,2.7)
plt.ylim(-0.5,1.1)
plt.axis('off')

### real surface


### 1
# px = [-3.7, -3.0, -2.5, -1.5, -1.2, -0.5, 0.05, 0.5, 1.9,  6.0]
# py = [-1.0,  0.0, -0.4, -0.3, -0.1, 0.05,  0.2, 0.5, 0.9, -1.0]
# plt.plot(px, py, color='r', marker='o', markersize=10, linewidth=3)
# px.append(0)
# py.append(0.7)
# px.append(3.3)
# py.append(0.3)
# px.append(2.3)
# py.append(-0.4)

### 1
# px = [-3.7, -3.0, -1.5, -1.0, -0.4, -0.5, 0.05, 0.5, 1.9, 2.2, 6.0]
# py = [-1.0,  0.0, -0.3, -0.1, -0.3, 0.05,  0.2, -0.4, -0.4, -0.9, 1.0]
# plt.plot(px, py, color='r', marker='o', markersize=10, linewidth=3)
# px.append(-0.1)
# py.append(0.4)


### 2
px = [-3.7, -1.4, -1.2, -0.6, -0.1, 0.05, 0.6, 1.9, 2.2, 6.0,7.0]
py = [-1.0,  -0.1, -0.3, 0.28,  0.27, -0.04, -0.3, -0.24, 1.9, 1.0,-2.0]
plt.plot(px, py, color='r', marker='o', markersize=10, linewidth=3)
px.append(-0.51)
py.append(-0.4)
px.append(-3.3)
py.append(0.0)
px.append(0.3)
py.append(-0.4)


points = np.stack((px, py), axis=1)
plt.scatter(px, py, color='r', marker='o', s=180, zorder=10)
# plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/points.pdf")

# add noise

### correct surface
# px=[-5, -2.5, 6*np.pi/4, 10*np.pi/4, 14*np.pi/4, 4*np.pi, 7*np.pi, 9*np.pi]
# py=[-1, -1.07, -0.95, 1, -1, 0, 0, 2.3]
# points=np.stack((px,py),axis=1)
# # py=np.sin(px)
# # plt.plot(px,py,color='g',marker='o')



### delaunay
tri = Delaunay(points)
centers = np.sum(points[tri.simplices], axis=1)/3.0

fcolors=[]
for i,c in enumerate(centers):
    ### get the function index
    if c[0] < 0:
        fcolors.append(1.0)
    else:
        fcolors.append(0.0)
fcolors=np.array(fcolors)
flatui = ['#d4efdf', "#3498db", "#95a5a6", "#e74c3c", "#34495e", '#fab0e4', '#52be80']
# flatui = ['#ffd8d8', "#3498db", "#95a5a6", "#e74c3c", "#34495e", '#fab0e4', '#c43131']
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())
plt.tripcolor(points[:,0], points[:,1], tri.simplices.copy(), facecolors=fcolors, edgecolors='0.75', linewidth=2, cmap=my_cmap)

# plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy(), color='0.75', linewidth=2)

plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/train_graphs/train_graph2.svg")


plt.show(block=True)

