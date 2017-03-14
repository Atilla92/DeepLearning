import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA


#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import numpy as np
from itertools import product, combinations
#from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d




pix_x = 160 # Amount of pixels horizontally
pix_y = 20 # Amount of pixels vertically
r =1  # radius of unit circumpherence
pos_begin =np.array( [1,0,0])

heading = 0
# Define sphere and begin position, velocity
#pos_begin = np.array([0,0,0])
spx= [2]#,-5]#, 3, 2]
spy= [2]#,-3]#, -2, 2]
spz= [0]#,-6]#, 3, -2]
rsp = [0.5]#,3]#, 0.5, 0.5]
V = [1,0,0]

# draw a vector
class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)



az, el = np.mgrid[-np.pi:np.pi:160j,-np.pi/2 :np.pi/2:20j]
unit_mat_x = np.array(np.cos(az)*np.sin(el+np.pi/2))*r
unit_mat_y = np.array(np.sin(az)*np.sin(el+np.pi/2))*r
unit_mat_z = np.array(np.cos(el+np.pi/2))*r
unit_vec_x = unit_mat_x.flatten()
unit_vec_y = unit_mat_y.flatten()
unit_vec_z= unit_mat_z.flatten()

# Plot x, y values, sanity check
t = np.arange(len(unit_vec_z))
plotx = plt.plot(t,unit_vec_x, t,unit_vec_y)
plt.show()
plt.figure()
q= plt.quiver(az,el,unit_mat_x,unit_mat_y,unit_mat_z, scale = 10)
plt.show()

# create 3D vecotr plot
# include pos begin


figVec = False

if figVec == True:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    ax.plot_wireframe(x, y, z, color="r")
    for i in range(1,len(z),1):
        # print(x[i], y[i], z[i])
        a = Arrow3D([pos_begin[0]+unit_vec_x[i], 2*unit_vec_x[i]+pos_begin[0]],
                    [pos_begin[1]+unit_vec_y[i], 2*unit_vec_y[i]+pos_begin[1]], 
                    [pos_begin[2]+unit_vec_z[i],pos_begin[2]+ 2*unit_vec_z[i]], mutation_scale=20,
                lw=1, arrowstyle="-|>", color="k")
        #norm =LA.norm([x[i], y[i], z[i]])
        #print (norm)
        q = ax.add_artist(a)
        ax.set_xlim([-1, 5])
        ax.set_ylim([-1, 5])
        ax.set_zlim([-1,5])
    
    plt.show()

#### Rotation #### 

alpha = np.ones(len(unit_vec_x)) * heading 
rot_vec_x = []
rot_vec_y = []
rot_vec_z = []

for i in range(len(alpha)):
    alpha_val = alpha[i]
    rot = np.array([np.cos(alpha_val), -np.sin(alpha_val), 0,np.sin(alpha_val), np.cos(alpha_val),0, 0,0,1])
    rot = rot.reshape(3,3)
    mat = np.array([unit_vec_x[i],unit_vec_y[i], unit_vec_z[i]])
    mat_rot = np.dot(rot, mat)
    rot_vec_x.append(mat_rot[0])
    rot_vec_y.append(mat_rot[1])
    rot_vec_z.append(mat_rot[2])
# print(mat_rot)
pos_rot_mat =np.vstack([np.array(rot_vec_x), np.array(rot_vec_y), np.array(rot_vec_y)])
pos_rot2 = pos_rot_mat.reshape(3,pix_x,pix_y)
#q = plt.quiver(u,v,pos_rot2[0,:,:] , pos_rot2[1,:,:],pos_rot2[2,:,:],  scale = 10)
q = plt.quiver(az,el,rot_vec_x,rot_vec_y,rot_vec_z,  scale = 10)
plt.show()


##### 

## source > http://paulbourke.net/geometry/circlesphere/ 
##source code: http://paulbourke.net/geometry/circlesphere/sphere_line_intersection.py

def sphere_line_intersection(l1, l2, sp, r):

    def square(f):
        return f * f
    from math import sqrt

    # l1[0],l1[1],l1[2]  P1 coordinates (point of line)
    # l2[0],l2[1],l2[2]  P2 coordinates (point of line)
    # sp[0],sp[1],sp[2], r  P3 coordinates and radius (sphere)
    # x,y,z   intersection coordinates
    #
    # This function returns a pointer array which first index indicates
    # the number of intersection point, followed by coordinate pairs.

    p1 = p2 = None

    a = square(l2[0] - l1[0]) + square(l2[1] - l1[1]) + square(l2[2] - l1[2])
    b = 2.0 * ((l2[0] - l1[0]) * (l1[0] - sp[0]) +
               (l2[1] - l1[1]) * (l1[1] - sp[1]) +
               (l2[2] - l1[2]) * (l1[2] - sp[2]))

    c = (square(sp[0]) + square(sp[1]) + square(sp[2]) + square(l1[0]) +
            square(l1[1]) + square(l1[2]) -
            2.0 * (sp[0] * l1[0] + sp[1] * l1[1] + sp[2] * l1[2]) - square(r))

    i = b * b - 4.0 * a * c

    if i < 0.0:
        pass  # no intersections
    elif i == 0.0:
        # one intersection
        p[0] = 1.0

        mu = -b / (2.0 * a)
        p1 = (l1[0] + mu * (l2[0] - l1[0]),
              l1[1] + mu * (l2[1] - l1[1]),
              l1[2] + mu * (l2[2] - l1[2]),
              )

    elif i > 0.0:
        # first intersection
        mu = (-b + sqrt(i)) / (2.0 * a)
        p1 = (l1[0] + mu * (l2[0] - l1[0]),
              l1[1] + mu * (l2[1] - l1[1]),
              l1[2] + mu * (l2[2] - l1[2]),
              )

        # second intersection
        mu = (-b - sqrt(i)) / (2.0 * a)
        p2 = (l1[0] + mu * (l2[0] - l1[0]),
              l1[1] + mu * (l2[1] - l1[1]),
              l1[2] + mu * (l2[2] - l1[2]),
              )

    return p1, p2



# Distance to intersection point

def distance_intersection(p1,p2,l2):
    distance=[] 
    distance2 = []
    if p1!=None:
        distance.append(LA.norm(np.array(np.array(p1)-l2)))
    if p2!=None:
        distance.append(LA.norm(np.array(np.array(p2)-l2)))
    if len(distance)>0:
        distance2= np.min(distance)
    return distance2


def estimateDistance(spx, spy, spz, rsp, pos_rot_X,pos_rot_Y, pos_rot_Z, pos_begin, noInter, V):
    d_store= []
    d_pos = []
    d_vec=[]
    i_loc=[]
    d_pos2 = []
    OF_pix = []
    OF_x= []
    OF_y = []
    OF_z = []
    for i in range (len(pos_rot_X)):
        # Define two point along vector line
        l2 = np.array([pos_rot_X[i] + pos_begin[0], pos_rot_Y[i]+pos_begin[1], pos_rot_Z[i]+pos_begin[2]])
        l1 = pos_begin
        for j in range(len(spx)):
            #takes two points along the unit vector of one pixel, and finds its intersection point with a sphere
            p1, p2 = sphere_line_intersection(pos_begin,  # begin position point, all point
                            l2,
                            [spx[j], spy[j], spz[j]], # sphere coordinates
                                              rsp[j]) # radius sphere
            # This computes the minimum distance to the sphere for one sphere
            if p1!=None or p2!=None:
                p1Vec =  np.array(p1) - pos_begin
                p2Vec =  np.array(p2) - pos_begin
                l2Vec = l2 - pos_begin
                if (np.sum((p1Vec >= 0 )*1-(l2Vec>= 0)*1)) == 0:
                    d_vec.append(distance_intersection(p1, p2, l2))
                if (np.sum((p1Vec >= 0 )*1-(l2Vec>= 0)*1)) == 0:
                    d_vec.append(distance_intersection(p1, p2, l2))
                #print (d_vec)
                #print(d_vec)
        if d_vec== []:
            d_store.append(noInter)
            d_pos.append(l2)
            OF_pix.append([0,0,0])
            OF_x.append(0)
            OF_y.append(0)
            OF_z.append(0)
        if d_vec!= [] :
            D = np.min(d_vec)
            d_store.append(np.min(d_vec))
            d_pos.append(l2)
            d_vec=[]
            # Ccompute Optic Flow
            OF = -np.divide((V - np.multiply(np.dot(V,[pos_rot_X[i], pos_rot_Y[i], pos_rot_Z[i]]),[pos_rot_X[i]
                                                                            , pos_rot_Y[i], pos_rot_Z[i]])), D)
            OF_x.append(OF[0])
            OF_y.append(OF[1])
            OF_z.append(OF[2])
            OF_pix.append(OF)
    OF_pix = np.vstack((np.array(OF_x), np.array(OF_y), np.array(OF_z)))
    return d_store, OF_pix, OF_x, OF_y, OF_z

distances, OF_stack, OF_vec_x, OF_vec_y, OF_vec_z= estimateDistance(spx, spy, spz, rsp, rot_vec_x,rot_vec_y, rot_vec_z, pos_begin, False, V)
print(np.shape(np.array(OF_stack)))

OF_mat= OF_stack.reshape(3,pix_x,pix_y)

print(np.shape(OF_mat))


######


q = plt.quiver(az,el, OF_mat[0,:,:], OF_mat[1,:,:],
                #              OF2[2,:,:] , 
                np.linalg.norm(OF_mat[:,:,:],axis=0)/100 , 
               scale = 4)
plt.show()

####

# create 3D vecotr plot

figVec = True

if figVec == True:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    #ax.plot_wireframe(x, y, z, color="r")
    for i in range(1,len(OF_vec_x),1):
        if OF_vec_x[i] != 0:
#             a = Arrow3D([pos_begin[0]+rot_vec_x[i], pos_begin[0]+rot_vec_x[i]
#                          + OF_vec_x[i]*k], 
#                         [pos_begin[1]+rot_vec_y[i],rot_vec_y[i]+pos_begin[1]+ OF_vec_y[i]*k],
#                         [pos_begin[2]+rot_vec_z[i] ,pos_begin[2]+ OF_vec_z[i]*k+rot_vec_z[i]],
#                         mutation_scale=1,
#                 lw=0.5, arrowstyle="-|>")#, color="k")
            k = 1

            a = Arrow3D([pos_begin[0]+rot_vec_x[i], #pos_begin[0]+rot_vec_x[i] +
                         OF_vec_x[i]*k], 
                        [pos_begin[1]+rot_vec_y[i],#rot_vec_y[i]+pos_begin[1]+
                         OF_vec_y[i]*k],
                        [pos_begin[2]+rot_vec_z[i] ,#pos_begin[2]+rot_vec_z[i]]+,
                         OF_vec_z[i]*k],
                        mutation_scale=10,
                lw=0.5, arrowstyle="-|>")#, color="k")
            
            q = ax.add_artist(a)
#         k = 10
#         ax.set_xlim([-0.20*k, 0.20*k])
#         ax.set_ylim([-0.20*k, 0.20*k])
#         ax.set_zlim([-0.20*k,0.20*k])
    kVec = 2
#     a = Arrow3D([pos_begin[0], pos_begin[0]+V[0]*kVec], 
#                 [pos_begin[1],pos_begin[1]+ V[1]*kVec],
#                 [pos_begin[2],pos_begin[2]+V[2]*kVec], mutation_scale=10,lw=1, arrowstyle="-|>", color="r")
    a = Arrow3D([pos_begin[0],V[0]*kVec], 
                [pos_begin[1], V[1]*kVec],
                [pos_begin[2],V[2]*kVec], mutation_scale=10,lw=1, arrowstyle="-|>", color="r")
    q = ax.add_artist(a)
    for i in range(len(rsp)):
 
        u1, v1 = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        r = rsp[i]
        center_x = spx[i]
        center_y = spy[i]
        center_z = spz[i]
        x2 = r * np.outer(np.cos(u1), np.sin(v1)) + center_x
        y2 = r * np.outer(np.sin(u1), np.sin(v1)) + center_y
        z2 = r * np.outer(np.ones(np.size(u1)), np.cos(v1)) + center_z
        q = ax.plot_wireframe(x2, y2, z2, color="r")

    r = 1
    center_x = pos_begin[0]
    center_y = pos_begin[1]
    center_z = pos_begin[2]
    x2 = r * np.outer(np.cos(u1), np.sin(v1)) + center_x
    y2 = r * np.outer(np.sin(u1), np.sin(v1)) + center_y
    z2 = r * np.outer(np.ones(np.size(u1)), np.cos(v1)) + center_z
    q = ax.plot_wireframe(x2, y2, z2, color="g")
    #ax.add_artist(a)
    plt.show()