
import numpy as np
from random import randint
from numpy import linalg as LA
from scipy.spatial import ConvexHull
import matplotlib.path as mltp

def rand_obstacle(maxDis,maxRadius, numObj, SF1):
	s =[]
	s.append(np.array([randint(-maxDis, maxDis), randint(-maxDis, maxDis),0, np.random.sample()*maxRadius]))

	stopObj =False
	i = 1
	while stopObj == False: 
	    intersect = False
	    s_2 = np.array([randint(-maxDis, maxDis), randint(-maxDis, maxDis),0, np.random.sample()*maxRadius])
	    for num in s:
	        if LA.norm([num[0:2]-s_2[0:2]])< (num[3]+s_2[3])+SF1:
	            intersect = True

	    if intersect == False:
	        s.append(s_2)
	        i = i+1
	        if i>numObj-1:
	            stopObj = True
	obstacle =s
	return obstacle

def rand_velocity(maxVel):
	velocity = [randint(-maxVel, maxVel),randint(-maxVel,maxVel),0]
	velocity = np.divide(velocity, LA.norm(velocity))
	return velocity

def rand_heading(maxHeading):
	heading = np.random.sample(1)*maxHeading
	return heading

def rand_position(s, area, SF2):
	# Convec hull of points
	s1 =np.array(s)
	s1 = s1[:,0:2]
	hull = ConvexHull(s1)
	path =  mltp.Path(s1[hull.vertices])
	# Randomize position, check whether it lies within convex hull of obstacles, and whether it does not
	#intersect with objects
	stopPos = False
	while stopPos == False:
	    intersect = True
	    pos = np.array([randint(0,area),randint(0,area),0])
	    if path.contains_point(pos) ==True:
	        #print(pos, 'True')
	        for num in s:                
	            if LA.norm([num[0:2] -pos[0:2]])< (num[3]+SF2) and intersect== True:
	                position = pos
	                intersect = False
	                #print(num, intersect)
	        if intersect == True:
	            position = pos
	            stopPos = True
	return position


def object_rotation(heading, obstacle ):
	obstacles = np.array(obstacle) 
	alpha= -heading[0] # negative rotation about z axis
	rot = np.array([np.cos(alpha), -np.sin(alpha), 0,np.sin(alpha), np.cos(alpha),0, 0,0,1])
	rot = rot.reshape(3,3)
	mat_rot= np.array([[np.dot(rot, i)] for i in obstacles[:,0:3]])
	[r,d,c] = mat_rot.shape
	obstacles_rot = mat_rot.reshape(r,c)
	obstacles[:,0:3]=obstacles_rot
	return obstacles

def velocity_rotation(heading, velocity):
	#  Velocity rotation
	alpha= -heading[0]
	rot = np.array([np.cos(alpha), -np.sin(alpha), 0,np.sin(alpha), np.cos(alpha),0, 0,0,1])
	rot = rot.reshape(3,3)
	vel_rot = np.dot(rot, velocity)
	return vel_rot