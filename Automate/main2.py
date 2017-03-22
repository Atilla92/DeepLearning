
import numpy as np
import matplotlib.pylab as plt
import time 
import pickle
from numpy import linalg as LA
from random import randint
from graphics import *
from computations import *
from intersection import sphere_line_intersection
import glob, os, os.path
from safeFiles import *
#############################################################
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
SF2 = 1.5 # safety factor

# Set eye parameters:
fov_x  = np.pi/2 #half FOV along azimuth in radians
fov_y = np.pi/4 # half FOV along elevation in radians
res_x = 100 #amount of pixels along azimuth
res_y = 21 # amount of pixels along elevation
############################################################################
############################################################################
numSamples = 10
# Do one sequence
## 1 Create Obstacles
start_time = time.time()

###########################################################################
obstacles = rand_obstacle(maxDis,maxRadius, numObj, SF1)
u, a, b , layout= unit_vectors(fov_x, fov_y, res_x, res_y)
############################################################################
os.chdir("/home/atilla/Documents/DeepLearning/Automate/")
files = [file for file in glob.glob("*.hdf5")]
print(files)
#print(obstacles.shape)
test =  'mytestfile.hdf5'
save_obstacles(test, files, np.array(obstacles),res_x, res_y, numObj, fov_x, fov_y )

###########################################################################
samples = rand_samples(maxVel, maxHeading, obstacles, area, SF2, numSamples)

print( len(samples))


###########################################################################
	# velocity = rand_velocity(maxVel)
	# heading = rand_heading(maxHeading)
	# position = rand_position(obstacles, area, SF2)
num=1
for sam in samples:
	position = sam[0]
	velocity = sam[1]
	heading = sam[2]
	obstacles_rot = object_rotation(heading, obstacles)
	velocity_rot = velocity_rotation(heading, velocity)
	intersection_points = intersection_obstacles(u, obstacles_rot, position )
	ofx, ofy = optic_flow(u, intersection_points, a,b, velocity_rot)
	#print(ofx.shape)
	heading_new = delanauy_heading(position, obstacles, heading)
	save_results(num, test, ofx, ofy, heading_new,position, velocity,heading )
	num=num+1


############################################################################
end_time = time.time()-start_time
print('finished', end_time)


############################################################################
f = h5py.File(test)
listf = [key for key in f.keys() if 'tryout' in key]
# xOF = [f[x]['ofx'][:] for x in listf]
# print(listf,np.array(xOF).shape, np.array(xOF[1]).shape)

#### cuidado q trayout nombre lo tienes que camvbiar en varios sitios
#####  need to store layout as well i suppose

####### SANITY CHECK ######

# print(obstacles, obstacles_rot.shape, velocity, velocity_rot,u.shape , a.shape, len(intersection_points), 
# 	len(ofx), len(ofy),
# 			heading_new)



# display_vectors(u.T, u.T)
#display_vectors(u.T, b.T)
#plt.show()

# plt.quiver(layout[:,0], layout[:,1], 
#            ofx, ofy, scale =5)
# plt.show()

# plt.plot(np.array(intersection_points))
# plt.show()
# plt.close()