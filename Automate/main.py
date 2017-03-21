
import numpy as np
import matplotlib.pylab as plt
import time 
import pickle
from numpy import linalg as LA
from random import randint
from graphics import *
from computations import *
from intersection import sphere_line_intersection

# Initial restrictions on obstacles
area = 10
numObj = 10
minRadius = 2
maxRadius = 4
maxDis= 10
area = 10
#Safety factor for intersection between obstalces
SF1 = 1.5


# Set boundaries velocity and heading
maxVel = 20
maxHeading = np.pi*2
SF2 = 1 # safety factor

# Set eye parameters:

fov_x  = np.pi/2 #half FOV along azimuth in radians
fov_y = np.pi/4 # half FOV along elevation in radians
res_x = 100 #amount of pixels along azimuth
res_y = 21 # amount of pixels along elevation


# Do one sequence
## 1 Create Obstacles
start_time = time.time()
obstacles = rand_obstacle(maxDis,maxRadius, numObj, SF1)
velocity = rand_velocity(maxVel)
heading = rand_heading(maxHeading)
position = rand_position(obstacles, area, SF2)
obstacles_rot = object_rotation(heading, obstacles)
velocity_rot = velocity_rotation(heading, velocity)
u, a, b , layout= unit_vectors(fov_x, fov_y, res_x, res_y)
intersection_points = intersection_obstacles(u, obstacles_rot, position )
ofx, ofy = optic_flow(u, intersection_points, a,b, velocity_rot)
heading_new = delanauy_heading(position, obstacles, heading)
end_time = time.time()-start_time
print('finished', end_time)
# display_vectors(u.T, u.T)
#display_vectors(u.T, b.T)

#plt.show()
print(obstacles, obstacles_rot.shape, velocity, velocity_rot,u.shape , a.shape, len(intersection_points), 
	len(ofx), len(ofy),
			heading_new)
# plt.quiver(layout[:,0], layout[:,1], 
#            ofx, ofy, scale =5)
# plt.show()

# plt.plot(np.array(intersection_points))
# plt.show()
# plt.close()