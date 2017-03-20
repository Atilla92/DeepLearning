
import numpy as np
from random import randint
from numpy import linalg as LA
def obstacle(maxDis,maxRadius, numObj, SF1):
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
