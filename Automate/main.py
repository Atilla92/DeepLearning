
import numpy as np
import matplotlib.pylab as plt
import time 
import pickle
from numpy import linalg as LA
from random import randint
from graphics import *
from computations import *

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



# Do one sequence
## 1 Create Obstacles
obstacles = rand_obstacle(maxDis,maxRadius, numObj, SF1)
velocity = rand_velocity(maxVel)
heading = rand_heading(maxHeading)
position = rand_position(obstacles, area, SF2)
obstacles_rot = object_rotation(heading, obstacles)
velocity_rot = velocity_rotation(heading, velocity)
print(obstacles, obstacles_rot, velocity, velocity_rot)