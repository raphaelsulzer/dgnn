import numpy as np

def pit(v0, v1, v2, v3):

    s = np.random.rand()
    t = np.random.rand()
    u = np.random.rand()

    if(s+t>1.0): # cut and fold the cube into a prism
        s = 1.0 - s
        t = 1.0 - t

    if(t+u>1.0): # cut'n fold the prism into a tetrahedron
        tmp = u
        u = 1.0 - s - t
        t = 1.0 - tmp
    elif(s+t+u>1.0):
        tmp = u
        u = s + t + u - 1.0
        s = 1 - t - tmp

    a=1-s-t-u # a,s,t,u are the barycentric coordinates of the random point.

    return v0*a + v1*s + v2*t + v3*u
