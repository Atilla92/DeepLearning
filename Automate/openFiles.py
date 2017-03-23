

import glob, os, os.path
import h5py
pathFiles = "/home/atilla/Documents/DeepLearning/Automate/"
os.chdir(pathFiles)
files = [file for file in glob.glob("*.hdf5") if 'test2_' in file]
print(files)


f = h5py.File(files[1])
a =f['obstacles'].attrs
group_name =f['obstacles'].attrs['Nomenclature']
list_groups = [key for key in f.keys() if group_name in key]#[1:]# if group_name in key]
lista = [key for key in a.keys()]
print(list_groups)
print(f['obstacles'][:])
print(f[list_groups[1]]['ofx'][:])