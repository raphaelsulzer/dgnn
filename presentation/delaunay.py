import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import seaborn as sns
from matplotlib.colors import ListedColormap
import os

path = "/home/adminlocal/PhD/python/dgnn/presentation/toy"

# flatui = ['#d4efdf', "#3498db", "#95a5a6", "#e74c3c", "#34495e", '#fab0e4', '#52be80']
# flatui = ['#d4efdf', '#ff0000', '#52be80']
flatui = ['#d4efdf', '#52be80']
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())


def function(x,noise=0):

    if noise:
        noisey = (np.random.random_sample() - 0.5) / 10
    else:
        noisey = 0

    if(x<=-3):
        return x+3
    elif(x>-3 and x<-2):
        return 0
    elif(x>=-2 and x<=0):
        return 0.25*(x+2)**2
    elif(x>0 and x<0.7):
        return -1.5*x+1
    else:
        return -0.25*(x-2.5)**2+1

def sdf(x,y):

    z = np.empty(x.shape)
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            z[i,j]=y[i,j]-function(x[i,j])


    return z


def plot_points(px,i,scatter=False):

    fig=plt.figure(figsize=(4, 2),dpi=1000)
    plt.tight_layout()
    plt.xlim(-2.7, 2.7)
    # plt.ylim(-0.5, 2.05)
    plt.ylim(-0.10, 1.1)
    plt.axis('off')
    py = []
    for x in px:
        py.append(function(x, 1))
    if(scatter):
        plt.scatter(px, py, color='r', marker='o',s=180,zorder=10)
        plt.savefig(os.path.join(path,"points","scatter.png"))
        plt.close(fig)
    else:
        plt.plot(px, py, color='r', marker='o', markersize=13, linewidth=4.5)
        plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/points/"+str(i)+".png")
        plt.close(fig)

def plot_points_with_noise(px,nx,ny):

    fig = plt.figure(figsize=(4, 2), dpi=1000)
    plt.tight_layout()
    plt.xlim(-2.7, 2.7)
    # plt.ylim(-0.5, 2.05)
    plt.ylim(-0.10, 1.1)
    plt.axis('off')
    py = []
    px = [-3.7, -3, -2.5, -1.5, -1, 1.9, 2.5, 3.3, 6]
    for x in px:
        py.append(function(x, 1))

    plt.plot(px, py, color='r', marker='o', markersize=13, linewidth=4.5)

    px = [-3.7, -3, -2.5, -1.5, 0.5, -1, 0.05, 1.9, 2.5, 3.3, 6]
    py = []
    for x in px:
        py.append(function(x, 1))
    plt.scatter(px, py, color='r', marker='o', s=180, zorder=10)

    plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/points/noise.png")
    plt.close(fig)

def plot_los(cx, cy, px, py):

    lx = [cx, px]
    ly = [cy, py]
    plt.plot(lx, ly, color='g', linewidth=2, linestyle='--')


def plot_rays(cx,cy,px,py):
    
    vx = cx-px
    vy = cy-py
    vxs=vx/(vx**2+vy**2)**0.5
    vys=vy/(vx**2+vy**2)**0.5

    rx = [px, px-0.5*vxs]
    ry = [py, py-0.5*vys]
    plt.plot(rx, ry, color='g', linewidth=2)


def plot_visibility(px,py):
    ### cameras
    c1x = -1.6
    c1y = 1.85
    plt.plot(c1x, c1y, color='g', marker=(3, 0, 0), markersize=25)

    c2x = 1.25
    c2y = 1.85
    plt.plot(c2x, c2y, color='g', marker=(3, 0, 0), markersize=25)

    ### visibility
    plot_los(c1x, c1y, px[2], py[2])
    plot_los(c1x, c1y, px[3], py[3])
    plot_los(c1x, c1y, px[4], py[4])
    plot_los(c2x, c2y, px[5], py[5])
    plot_los(c2x, c2y, px[6], py[6])
    plot_los(c2x, c2y, px[7], py[7])
    plot_los(c2x, c2y, px[8], py[8])

    # plot_los(c1x, c1y, px[11], py[11])

    plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/los.png")

    plot_rays(c1x, c1y, px[2], py[2])
    plot_rays(c1x, c1y, px[3], py[3])
    plot_rays(c1x, c1y, px[4], py[4])
    plot_rays(c2x, c2y, px[5], py[5])
    plot_rays(c2x, c2y, px[6], py[6])
    plot_rays(c2x, c2y, px[7], py[7])
    plot_rays(c2x, c2y, px[8], py[8])

    # plot_rays(c1x, c1y, px[11], py[11])

    plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/rays.png")


### real surface

fx = np.arange(-4,5.1,0.1)
fy = []
for x in fx:
    fy.append(function(x))



# plt.figure(figsize=(7,7))
# plt.tight_layout()
# plt.xlim(-2.575,2.58)
# plt.ylim(-0.5,2.05)
# plt.axis('off')
# plt.plot(fx, fy, color='g',linewidth=5)
# plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/surface.pdf")


### wrong surface

# px = [-3.7,-3,-2.5,-1.5,-1,0.05,0.5,1.9,2.5,3.3,6]
# plot_points(px,0)
# px = [-3.7,-3,-2.5,-1.5,-1,0.5,0.05,1.9,2.5,3.3,6]
# plot_points(px,1)
# px = [-3.7,-3,-2.5,-1.5,-1,0.5,1.9,2.5,0.05]
# plot_points(px,2)
# px = [-3.7,-3,-2.5,-1.5,0.05,-1,0.5,1.9,2.5,3.3,6]
# plot_points(px,3)
# px = [-3.7,-3,-2.5,-1.5,0.5,-1,0.05,1.9,2.5,3.3,6]
# plot_points(px,4)
# px = [-3.7,-3,-2.5,-1.5,0.5,-1,0.05,1.9,2.5,3.3,6]
# plot_points(px,4,True)




### correct surface

# layout with rays
plt.figure(figsize=(4,4),dpi=1000)
plt.tight_layout()
plt.xlim(-2.7,2.7)
plt.ylim(-0.5,2.05)
plt.axis('off')

# layout without rays
# fig = plt.figure(figsize=(4, 2), dpi=1000)
# plt.tight_layout()
# plt.xlim(-2.7, 2.7)
# plt.ylim(-0.10, 1.1)
# plt.axis('off')

### for voxels
# ax = plt.gca()
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_frame_on(False)
# ax.tick_params(tick1On=False)
# plt.grid(color='0.75', linestyle='-', linewidth=2)



# implicit surface
# gx = np.linspace(-2.7, 2.7, 100)
# gy = np.linspace(-0.10, 1.1, 100)
# cx, cy = np.meshgrid(gx,gy)
# cz = sdf(cx,cy)
#
# cz = gaussian_filter(cz, sigma=3.0)
#
# plt.contour(cx,cy,cz, 1000, cmap=my_cmap)
#
# plt.contour(cx,cy,cz, [0], colors='Red', linewidths=[4.5])
# plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/isosurface.png")

# plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/field.png")



px = [-3.7,-3,-2.5,-1.5,-1,0.05,0.5,1.9,2.5,3.3,6]
py = []
for x in px:
    py.append(function(x,1))

# px.append(-2.2)
# py.append(0.2)
# px.append(-2.15)
# py.append(-0.2)
px.append(0)
py.append(0.5)


points=np.stack((px,py),axis=1)
plt.scatter(px, py, color='r', marker='o',s=180,zorder=10)
# plt.grid()
plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/points/scatter.png")
# px.pop(6)
# py.pop(6)
# plt.plot(px, py, color='r', marker='o', markersize=10, linewidth=3)

# add noise

### correct surface
# px=[-5, -2.5, 6*np.pi/4, 10*np.pi/4, 14*np.pi/4, 4*np.pi, 7*np.pi, 9*np.pi]
# py=[-1, -1.07, -0.95, 1, -1, 0, 0, 2.3]
# points=np.stack((px,py),axis=1)
# py=np.sin(px)
# plt.plot(px,py,color='g',marker='o')



### visibility

# plot_visibility(px,py)


### delaunay
tri = Delaunay(points)
centers = np.sum(points[tri.simplices], axis=1)/3.0

fcolors=[]
for i,c in enumerate(centers):
    ### get the function index
    if c[1] < function(c[0]):
        fcolors.append(1.0)
    else:
        fcolors.append(0)
fcolors=np.array(fcolors)





plt.triplot(points[:,0], points[:,1], tri.simplices.copy(), color='0.75', linewidth=2)
plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/delaunay.png")
#
plt.tripcolor(points[:,0], points[:,1], tri.simplices.copy(), facecolors=fcolors, edgecolors='k', linewidth=2, cmap=my_cmap)
plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/delaunay_colored.png")
#
plt.plot(px[:-1], py[:-1], color='r', marker='o', markersize=10, linewidth=3)
plt.savefig("/home/adminlocal/PhD/cpp/surfaceReconstruction/presentation/delaunay_interface.png")

# plt.scatter(centers[:,0], centers[:,1], marker='o', s=50)

plt.show(block=True)





a = 5
