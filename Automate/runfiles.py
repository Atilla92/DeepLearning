
from computations import *
from safeFiles import *
def create_samples(maxDis, maxRadius, numObj, SF1, fov_x, fov_y, res_x, res_y, pathFiles, test, maxVel, maxHeading, area, SF2, numSamples, group_name ):
	obstacles = rand_obstacle(maxDis,maxRadius, numObj, SF1)
	u, a, b , layout= unit_vectors(fov_x, fov_y, res_x, res_y)
	files= find_files(pathFiles)
	save_obstacles(test, files, np.array(obstacles),res_x, res_y, numObj, fov_x, fov_y, group_name )
	samples = rand_samples(maxVel, maxHeading, obstacles, area, SF2, numSamples)
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
		save_results(num, test, ofx, ofy, heading_new,position, velocity,heading, group_name )
		num=num+1

