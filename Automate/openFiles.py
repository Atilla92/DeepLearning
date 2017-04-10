
import time
import glob, os, os.path
import h5py
import numpy as np
from safeFiles import *
from dataPreprocessing import *
from computations import *


safe_Files = False
pathFiles = "/home/atilla/Documents/DeepLearning/Test/"
os.chdir(pathFiles)

#files = find_files(pathFiles)
files = [file for file in glob.glob("*.hdf5") if 'test6_' in file]

print(files)
start_time = time.time()

# For newer datasets, set gaps_status to true and you will also get the three gaps data,
# still need to extraxt training set and storing it properly for the gaps
train_x, train_y, gaps_y, mat2 = extract_files(files, gaps_status= True)


end_time = time.time()-start_time
print('finished', end_time)



# fov_x  = np.pi #half FOV along azimuth in radians
# fov_y = np.pi/4 # half FOV along elevation in radians
# res_x = 360 #amount of pixels along azimuth
# res_y = 30 # amount of pixels along elevation



# phis   = np.linspace(-fov_x, fov_x, res_x)
# thetas = np.linspace(-fov_y, fov_y, res_y)


# layout = np.array([[phi, theta] 
#                    for theta in thetas
#                    for phi in phis])

# print(np.sum(train_x[0][0][...]), np.sum(mat2[0][0]), 'sums')
# print(train_x[0][0][...].shape,'train flat', mat2[0][0] ,'train mat')
# print(train_x.shape)
# ofx = mat2[0][0]
# ofy = mat2[0][1]
# ofx = train_x[0][0][...]
# #ofx = ofx.flatten()
# print(np.sum(ofx), np.sum(mat2[0][0]))
# # ofy = train_x[0][1][...].flatten()
# plt.quiver(layout[:,0], layout[:,1], 
#            ofx, ofy)
# plt.show()




#Normalize dataX and Categorize dataY. Could t
train_norm = train_x
train_norm_x = normalize_data(train_norm)
#print(np.sum(train_x[0][0][...]), np.sum(mat2[0][0]), 'sums2')
train_cate_y = categorize_data(train_y)
gaps_cate_y = categorize_data(gaps_y)
gaps_cate_y = np.sum(gaps_cate_y, axis = 0)
gaps_cate_y = gaps_cate_y.reshape(1, len(gaps_cate_y))
#print(gaps_cate_y.shape, train_cate_y.shape)

train_norm_rand_x, train_cate_rand_y = randomize_data(train_norm_x, train_cate_y)
train_norm_rand_x, gaps_cate_rand_y = randomize_data(train_norm_x, gaps_cate_y)
 
#Store datasets 

if safe_Files == True:
	file_name = 'test3_1_'
	f = h5py.File(file_name + "compact.hdf5", "w")
	f.create_dataset('data_x', data =train_x, dtype='f', chunks = True )
	f.create_dataset('data_y', data = train_y, dtype = 'f', chunks = True )
	f.create_dataset('gaps_y', data = gaps_y, dtype = 'f', chunks = True )
	#test = file_name+time.strftime('%F-%T')+'.hdf5'
	f = h5py.File(file_name + "compact_rand.hdf5", "w")
	f.create_dataset('data_x', data =train_norm_rand_x, dtype='f', chunks = True )
	f.create_dataset('data_y', data = train_cate_rand_y, dtype = 'f', chunks = True )
	f.create_dataset('gaps_y', data = gaps_cate_rand_y, dtype = 'f', chunks = True )
	f = h5py.File(file_name + "compact_norm.hdf5", "w")
	f.create_dataset('data_x', data= train_norm_x, dtype = 'f', chunks = True)
	f.create_dataset('data_y', data = train_cate_y, dtype = 'f', chunks = True)
	f.create_dataset('gaps_y', data = gaps_cate_y, dtype = 'f', chunks = True )




# import matplotlib.pylab as plt
# import numpy as np
# bins = np.linspace(-np.pi, np.pi, 360)
# plt.hist(train_y, bins)

# plt.show()

fov_x  = np.pi #half FOV along azimuth in radians
fov_y = np.pi/4 # half FOV along elevation in radians
res_x = 360 #amount of pixels along azimuth
res_y = 30 # amount of pixels along elevation



phis   = np.linspace(-fov_x, fov_x, res_x)
thetas = np.linspace(-fov_y, fov_y, res_y)


layout = np.array([[phi, theta] 
                   for theta in thetas
                   for phi in phis])

ofx = mat2[0][0]
ofy = mat2[0][1]
ofx = train_x[0][0][...]
#ofx = ofx.flatten()
#print(np.sum(ofx), np.sum(mat2[0][0]))
# ofy = train_x[0][1][...].flatten()
plt.quiver(layout[:,0], layout[:,1], 
           ofx, ofy)
plt.show()