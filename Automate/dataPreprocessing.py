import numpy as np
import h5py
from keras.utils.np_utils import to_categorical
import matplotlib.pylab as plt
def extract_files(files, gaps_status, ):

	matx = []
	test_mat = []
	maty = []
	gaps = []
	files = np.array(files)
	mat2 = []

	for file in files:
		f = h5py.File(file)
		
		group_name =f['obstacles'].attrs['Nomenclature']
		list_groups = [key for key in f.keys() if group_name in key]
		resolution = [f['obstacles'].attrs['Resolution'][1],f['obstacles'].attrs['Resolution'][0] ]
		#resolution = resolution[1], resolution
		#print(list_groups)
		mat = np.array([np.stack((f[group]['ofx'][...].reshape(
										resolution)
					,f[group]['ofy'][...].reshape(
										resolution)))
					for group in list_groups])
		mat1 = np.array([np.stack((f[group]['ofx'][...]
					,f[group]['ofy'][...]))
					for group in list_groups])

		#plt.scatter(mat1[0][0], mat[0][1])
		#plt.scatter(mat[0][0][...].flatten(), mat[0][1][...].flatten())
		#plt.plot(mat[0][1][...].flatten)
		#plt.show()

		#print(mat.shape)
		heading = np.array([f[group]['new_heading'][...]
					for group in list_groups])
		if gaps_status == True:
			gap = np.array([f[group]['gaps'][...]
						for group in list_groups])
			gaps.append(gap) 
		matx.append(mat)
		maty.append(heading)
		mat2.append(mat1)
		
	matx = np.array(matx)
	train_x = np.vstack(matx)
	train_y = np.vstack(maty)
	mat2 = np.vstack(np.array(mat2))
	#print(train_x, mat2)
	if gaps_status == True:
		train_gaps = np.vstack(gaps)
		#print('this one')
		return train_x, train_y, train_gaps, mat2
	if gaps_status == False:
		return train_x, train_y, mat2

def normalize_data(train_norm):
	for b, exa in enumerate(train_norm):
		#print(exa.shape)
	    for a, sample in enumerate(exa):
	        train_norm[b][a]= np.divide(np.subtract(sample,sample.mean()), sample.std())
	        #train_norm[b][a]= np.divide(np.subtract(sample, np.min(sample)),np.subtract(np.max(sample), np.min(sample)))
	        print(sample.shape, 'sample shape', a)
	return train_norm

def categorize_data(train_y):
	train_y = np.round(train_y*180/np.pi+180)
	categorical_labels = to_categorical(train_y, 361)
	return categorical_labels

def randomize_data(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

